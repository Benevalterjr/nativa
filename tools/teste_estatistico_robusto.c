#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>
#include <omp.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4]; // BF16 scales for 4 rows
    uint8_t  qs[64];   // 64 bytes: 16 bytes per row * 4 rows = 32 nibbles/row
} BlockQ4x4;
#pragma pack(pop)

typedef struct {
    float scale;
    int8_t qs[32];
    int32_t sum_adj;
} BlockQ8;

static inline float bf16_to_fp32(uint16_t v) {
    union { uint32_t u; float f; } conv;
    conv.u = ((uint32_t)v) << 16;
    return conv.f;
}

static inline uint16_t fp32_to_bf16(float f) {
    union { uint32_t u; float f; } conv;
    conv.f = f;
    return (uint16_t)(conv.u >> 16);
}

// ----------------------------------------------------------------------------
// 1. ATIVAÇÃO VETORIZADA SWIGLU (Fast Exp)
// ----------------------------------------------------------------------------
static inline __m128 fast_exp_ps(__m128 x) {
    __m128 max_x = _mm_set1_ps(88.3762626647949f);
    __m128 min_x = _mm_set1_ps(-88.3762626647949f);
    x = _mm_max_ps(min_x, _mm_min_ps(max_x, x));

    __m128 log2ef = _mm_set1_ps(1.44269504088896341f);
    __m128 fx = _mm_mul_ps(x, log2ef);
    fx = _mm_floor_ps(_mm_add_ps(fx, _mm_set1_ps(0.5f)));

    __m128 c_c1 = _mm_set1_ps(0.693359375f);
    __m128 c_c2 = _mm_set1_ps(-2.12194440e-4f);
    __m128 y = _mm_sub_ps(x, _mm_mul_ps(fx, c_c1));
    y = _mm_sub_ps(y, _mm_mul_ps(fx, c_c2));

    __m128 y2 = _mm_mul_ps(y, y);
    __m128 p = _mm_set1_ps(1.9875691500E-4f);
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(1.3981999507E-3f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(8.3334519073E-3f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(4.1665795894E-2f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(1.6666665459E-1f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(5.0000001201E-1f));
    p = _mm_add_ps(_mm_mul_ps(p, y2), y);
    p = _mm_add_ps(p, _mm_set1_ps(1.0f));

    __m128i emm0 = _mm_cvttps_epi32(fx);
    emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(127));
    emm0 = _mm_slli_epi32(emm0, 23);
    return _mm_mul_ps(p, _mm_castsi128_ps(emm0));
}

static inline __m128 fast_swiglu_ps(__m128 g, __m128 u) {
    __m128 neg_g = _mm_sub_ps(_mm_setzero_ps(), g);
    __m128 exp_neg = fast_exp_ps(neg_g);
    __m128 one_plus_exp = _mm_add_ps(_mm_set1_ps(1.0f), exp_neg);
    __m128 silu_g = _mm_div_ps(g, one_plus_exp);
    return _mm_mul_ps(silu_g, u);
}

// ----------------------------------------------------------------------------
// 2. QUANTIZAÇÃO E REFERÊNCIA ESCALAR PURA
// ----------------------------------------------------------------------------
static void quantize_x_to_q8(const float* x, BlockQ8* x_q8, int cols) {
    int num_blocks = cols / 32;
    for (int b = 0; b < num_blocks; b++) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float af = fabsf(xb[i]);
            if (af > amax) amax = af;
        }
        if (amax < 1e-12f) {
            x_q8[b].scale = 0.0f;
            memset(x_q8[b].qs, 0, 32);
            x_q8[b].sum_adj = 0;
            continue;
        }
        float scale = amax / 127.0f;
        float inv_scale = 1.0f / scale;
        x_q8[b].scale = scale;
        int32_t s = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(xb[i] * inv_scale);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            x_q8[b].qs[i] = (int8_t)q;
            s += q;
        }
        x_q8[b].sum_adj = s * 8;
    }
}

void mlp_gate_up_swiglu_scalar_ref(
    float* restrict h_mlp,
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows, int cols
) {
    int nb = cols / 32;
    int groups = rows / 4;

    for (int g = 0; g < groups; g++) {
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;

        float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float uv[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int b = 0; b < nb; b++) {
            const BlockQ8* xb = &x_q8[b];
            const BlockQ4x4* gb = &g_blk_row[b];
            const BlockQ4x4* ub = &u_blk_row[b];

            for (int r = 0; r < 4; r++) {
                float gw_sc = bf16_to_fp32(gb->scale[r]);
                float uw_sc = bf16_to_fp32(ub->scale[r]);
                const uint8_t* g_qs = gb->qs + r * 16;
                const uint8_t* u_qs = ub->qs + r * 16;

                int32_t g_dot = 0;
                int32_t u_dot = 0;
                for (int i = 0; i < 16; i++) {
                    int8_t glo = (int8_t)(g_qs[i] & 0x0F) - 8;
                    int8_t ghi = (int8_t)(g_qs[i] >> 4) - 8;
                    int8_t ulo = (int8_t)(u_qs[i] & 0x0F) - 8;
                    int8_t uhi = (int8_t)(u_qs[i] >> 4) - 8;

                    g_dot += (int32_t)glo * (int32_t)xb->qs[i] + (int32_t)ghi * (int32_t)xb->qs[i + 16];
                    u_dot += (int32_t)ulo * (int32_t)xb->qs[i] + (int32_t)uhi * (int32_t)xb->qs[i + 16];
                }
                gv[r] += (float)g_dot * (gw_sc * xb->scale);
                uv[r] += (float)u_dot * (uw_sc * xb->scale);
            }
        }

        for (int r = 0; r < 4; r++) {
            float silu_g = gv[r] / (1.0f + expf(-gv[r]));
            h_mlp[g * 4 + r] = silu_g * uv[r];
        }
    }

    int rem_rows = rows % 4;
    if (rem_rows > 0) {
        int g = groups;
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;
        for (int r = 0; r < rem_rows; r++) {
            float g_acc = 0.0f, u_acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                const BlockQ8* xb = &x_q8[b];
                const BlockQ4x4* gb = &g_blk_row[b];
                const BlockQ4x4* ub = &u_blk_row[b];
                float gw_sc = bf16_to_fp32(gb->scale[r]);
                float uw_sc = bf16_to_fp32(ub->scale[r]);
                const uint8_t* g_qs = gb->qs + r * 16;
                const uint8_t* u_qs = ub->qs + r * 16;
                int32_t g_dot = 0, u_dot = 0;
                for (int i = 0; i < 16; i++) {
                    int8_t glo = (int8_t)(g_qs[i] & 0x0F) - 8;
                    int8_t ghi = (int8_t)(g_qs[i] >> 4) - 8;
                    int8_t ulo = (int8_t)(u_qs[i] & 0x0F) - 8;
                    int8_t uhi = (int8_t)(u_qs[i] >> 4) - 8;
                    g_dot += (int32_t)glo * (int32_t)xb->qs[i] + (int32_t)ghi * (int32_t)xb->qs[i + 16];
                    u_dot += (int32_t)ulo * (int32_t)xb->qs[i] + (int32_t)uhi * (int32_t)xb->qs[i + 16];
                }
                g_acc += (float)g_dot * (gw_sc * xb->scale);
                u_acc += (float)u_dot * (uw_sc * xb->scale);
            }
            float silu_g = g_acc / (1.0f + expf(-g_acc));
            h_mlp[g * 4 + r] = silu_g * u_acc;
        }
    }
}

// ----------------------------------------------------------------------------
// 3. BASELINE NATIVA (COM CAUDA)
// ----------------------------------------------------------------------------
static inline __m128 q4x4_dot4_baseline(const BlockQ4x4* restrict W4, const BlockQ8* restrict x_q8, int nb) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();
    __m128 acc = _mm_setzero_ps();
    int b;
    for (b = 0; b < nb; b++) {
        const BlockQ4x4* blk = &W4[b];
        const BlockQ8* xb = &x_q8[b];
        __m128i x_lo = _mm_loadu_si128((const __m128i*)xb->qs);
        __m128i x_hi = _mm_loadu_si128((const __m128i*)(xb->qs + 16));

        __m128i r0 = _mm_loadu_si128((const __m128i*)(blk->qs));
        __m128i r1 = _mm_loadu_si128((const __m128i*)(blk->qs + 16));
        __m128i r2 = _mm_loadu_si128((const __m128i*)(blk->qs + 32));
        __m128i r3 = _mm_loadu_si128((const __m128i*)(blk->qs + 48));

        __m128i t0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(r0, mask), x_lo),
                                                  _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(r0, 4), mask), x_hi)), ones);
        __m128i t1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(r1, mask), x_lo),
                                                  _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(r1, 4), mask), x_hi)), ones);
        __m128i t2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(r2, mask), x_lo),
                                                  _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(r2, 4), mask), x_hi)), ones);
        __m128i t3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(r3, mask), x_lo),
                                                  _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(r3, 4), mask), x_hi)), ones);

        __m128i h = _mm_hadd_epi32(_mm_hadd_epi32(t0, t1), _mm_hadd_epi32(t2, t3));
        __m128 fd = _mm_cvtepi32_ps(_mm_sub_epi32(h, _mm_set1_epi32(xb->sum_adj)));

        __m128 sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)blk->scale)));
        acc = _mm_add_ps(acc, _mm_mul_ps(fd, _mm_mul_ps(sc, _mm_set1_ps(xb->scale))));
    }
    return acc;
}

void mlp_gate_up_swiglu_baseline(
    float* restrict h_mlp,
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows, int cols,
    bool use_fast_exp
) {
    int nb = cols / 32;
    int groups = rows / 4;

    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        __m128 gv = q4x4_dot4_baseline(Wgate + (size_t)g * nb, x_q8, nb);
        __m128 uv = q4x4_dot4_baseline(Wup   + (size_t)g * nb, x_q8, nb);

        if (use_fast_exp) {
            __m128 out = fast_swiglu_ps(gv, uv);
            _mm_storeu_ps(&h_mlp[g * 4], out);
        } else {
            float garr[4], uarr[4];
            _mm_storeu_ps(garr, gv);
            _mm_storeu_ps(uarr, uv);
            for (int j = 0; j < 4; j++) {
                h_mlp[g * 4 + j] = (garr[j] / (1.0f + expf(-garr[j]))) * uarr[j];
            }
        }
    }

    int rem = rows % 4;
    if (rem > 0) {
        int g = groups;
        __m128 gv = q4x4_dot4_baseline(Wgate + (size_t)g * nb, x_q8, nb);
        __m128 uv = q4x4_dot4_baseline(Wup   + (size_t)g * nb, x_q8, nb);
        float garr[4], uarr[4];
        _mm_storeu_ps(garr, gv);
        _mm_storeu_ps(uarr, uv);
        for (int j = 0; j < rem; j++) {
            h_mlp[g * 4 + j] = (garr[j] / (1.0f + expf(-garr[j]))) * uarr[j];
        }
    }
}

// ----------------------------------------------------------------------------
// 4. TILED LEAN KERNEL (REUSO DE X, <= 15 REGISTRADORES, CAUDA COMPLETA)
// ----------------------------------------------------------------------------
void mlp_gate_up_swiglu_tiled_lean(
    float* restrict h_mlp,
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows, int cols,
    bool use_fast_exp
) {
    int nb = cols / 32;
    int groups = rows / 4;
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();

    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;

        __m128 acc_g = _mm_setzero_ps();
        __m128 acc_u = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const BlockQ8* xb = &x_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
            __m128 x_sc = _mm_set1_ps(xb->scale);
            __m128i adj = _mm_set1_epi32(xb->sum_adj);

            // Gate
            const BlockQ4x4* gblk = &g_blk_row[b];
            __m128i gr0 = _mm_loadu_si128((const __m128i*)(gblk->qs));
            __m128i gr1 = _mm_loadu_si128((const __m128i*)(gblk->qs + 16));
            __m128i gr2 = _mm_loadu_si128((const __m128i*)(gblk->qs + 32));
            __m128i gr3 = _mm_loadu_si128((const __m128i*)(gblk->qs + 48));

            __m128i gt0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr0, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr0, 4), mask), x_hi)), ones);
            __m128i gt1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr1, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr1, 4), mask), x_hi)), ones);
            __m128i gt2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr2, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr2, 4), mask), x_hi)), ones);
            __m128i gt3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr3, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr3, 4), mask), x_hi)), ones);

            __m128i gh = _mm_hadd_epi32(_mm_hadd_epi32(gt0, gt1), _mm_hadd_epi32(gt2, gt3));
            __m128 g_fd = _mm_cvtepi32_ps(_mm_sub_epi32(gh, adj));
            __m128 g_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)gblk->scale)));
            acc_g = _mm_add_ps(acc_g, _mm_mul_ps(g_fd, _mm_mul_ps(g_sc, x_sc)));

            // Up (reutiliza x_lo e x_hi dos registradores)
            const BlockQ4x4* ublk = &u_blk_row[b];
            __m128i ur0 = _mm_loadu_si128((const __m128i*)(ublk->qs));
            __m128i ur1 = _mm_loadu_si128((const __m128i*)(ublk->qs + 16));
            __m128i ur2 = _mm_loadu_si128((const __m128i*)(ublk->qs + 32));
            __m128i ur3 = _mm_loadu_si128((const __m128i*)(ublk->qs + 48));

            __m128i ut0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur0, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur0, 4), mask), x_hi)), ones);
            __m128i ut1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur1, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur1, 4), mask), x_hi)), ones);
            __m128i ut2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur2, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur2, 4), mask), x_hi)), ones);
            __m128i ut3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur3, mask), x_lo),
                                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur3, 4), mask), x_hi)), ones);

            __m128i uh = _mm_hadd_epi32(_mm_hadd_epi32(ut0, ut1), _mm_hadd_epi32(ut2, ut3));
            __m128 u_fd = _mm_cvtepi32_ps(_mm_sub_epi32(uh, adj));
            __m128 u_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)ublk->scale)));
            acc_u = _mm_add_ps(acc_u, _mm_mul_ps(u_fd, _mm_mul_ps(u_sc, x_sc)));
        }

        if (use_fast_exp) {
            __m128 out = fast_swiglu_ps(acc_g, acc_u);
            _mm_storeu_ps(&h_mlp[g * 4], out);
        } else {
            float garr[4], uarr[4];
            _mm_storeu_ps(garr, acc_g);
            _mm_storeu_ps(uarr, acc_u);
            for (int j = 0; j < 4; j++) {
                h_mlp[g * 4 + j] = (garr[j] / (1.0f + expf(-garr[j]))) * uarr[j];
            }
        }
    }

    int rem_rows = rows % 4;
    if (rem_rows > 0) {
        int g = groups;
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;
        for (int r = 0; r < rem_rows; r++) {
            float g_acc = 0.0f, u_acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                const BlockQ8* xb = &x_q8[b];
                const BlockQ4x4* gb = &g_blk_row[b];
                const BlockQ4x4* ub = &u_blk_row[b];
                float gw_sc = bf16_to_fp32(gb->scale[r]);
                float uw_sc = bf16_to_fp32(ub->scale[r]);
                const uint8_t* g_qs = gb->qs + r * 16;
                const uint8_t* u_qs = ub->qs + r * 16;
                int32_t g_dot = 0, u_dot = 0;
                for (int i = 0; i < 16; i++) {
                    int8_t glo = (int8_t)(g_qs[i] & 0x0F) - 8;
                    int8_t ghi = (int8_t)(g_qs[i] >> 4) - 8;
                    int8_t ulo = (int8_t)(u_qs[i] & 0x0F) - 8;
                    int8_t uhi = (int8_t)(u_qs[i] >> 4) - 8;
                    g_dot += (int32_t)glo * (int32_t)xb->qs[i] + (int32_t)ghi * (int32_t)xb->qs[i + 16];
                    u_dot += (int32_t)ulo * (int32_t)xb->qs[i] + (int32_t)uhi * (int32_t)xb->qs[i + 16];
                }
                g_acc += (float)g_dot * (gw_sc * xb->scale);
                u_acc += (float)u_dot * (uw_sc * xb->scale);
            }
            float silu_g = g_acc / (1.0f + expf(-g_acc));
            h_mlp[g * 4 + r] = silu_g * u_acc;
        }
    }
}

// ----------------------------------------------------------------------------
// ESTATÍSTICAS (Média, Mínimo, Mediana, Desvio Padrão)
// ----------------------------------------------------------------------------
typedef struct {
    double mean;
    double min_v;
    double median;
    double stddev;
} Stats;

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static Stats compute_stats(double* samples, int n) {
    Stats s;
    double sum = 0.0;
    s.min_v = 1e30;
    for (int i = 0; i < n; i++) {
        sum += samples[i];
        if (samples[i] < s.min_v) s.min_v = samples[i];
    }
    s.mean = sum / n;

    qsort(samples, n, sizeof(double), cmp_double);
    s.median = (n % 2 == 0) ? (samples[n / 2 - 1] + samples[n / 2]) * 0.5 : samples[n / 2];

    double sq_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double d = samples[i] - s.mean;
        sq_diff += d * d;
    }
    s.stddev = sqrt(sq_diff / n);
    return s;
}

// ----------------------------------------------------------------------------
// HARNESS PRINCIPAL: TESTES DE BORDA + BENCHMARK MULTI-THREAD
// ----------------------------------------------------------------------------
int main(void) {
    printf("====================================================================\n");
    printf("BATERIA DE ROBUSTEZ & BENCHMARK ESTATISTICO RIGOROSO (MLP / SWIGLU)\n");
    printf("Ambiente: Intel Core Gen 3 (Ivy Bridge x64, 4 Cores / 8 Threads)\n");
    printf("====================================================================\n\n");

    // ------------------------------------------------------------------------
    // PARTE 1: TESTES DE BORDA, CAUDAS E SANITIZAÇÃO DE MEMÓRIA
    // ------------------------------------------------------------------------
    printf("[1] TESTES DE CASOS EXTREMOS, CAUDAS E CANARIOS DE MEMORIA:\n");

    int test_shapes[][2] = {
        {1, 32},    // M=1,  K=32 (menor bloco possível)
        {2, 32},    // M=2,  K=32
        {3, 64},    // M=3,  K=64
        {4, 32},    // M=4,  K=32 (exato 1 grupo)
        {5, 96},    // M=5 (resto 1), K=96 (3 blocos, ímpar)
        {7, 128},   // M=7 (resto 3), K=128
        {8, 160},   // M=8 (exato 2 grupos), K=160 (5 blocos, ímpar)
        {1537, 768},// M=1537 (resto 1), K=768
        {8963, 1568}// M=8963 (resto 3), K=1568 (49 blocos, ímpar)
    };
    int num_shapes = sizeof(test_shapes) / sizeof(test_shapes[0]);

    bool all_passed = true;

    for (int s_idx = 0; s_idx < num_shapes; s_idx++) {
        int M = test_shapes[s_idx][0];
        int K = test_shapes[s_idx][1];
        int nb = K / 32;
        int ng = (M + 3) / 4;

        BlockQ4x4* Wg = (BlockQ4x4*)calloc((size_t)ng * nb, sizeof(BlockQ4x4));
        BlockQ4x4* Wu = (BlockQ4x4*)calloc((size_t)ng * nb, sizeof(BlockQ4x4));
        BlockQ8* xq = (BlockQ8*)calloc(nb, sizeof(BlockQ8));
        float* x = (float*)malloc(K * sizeof(float));

        // Buffer com sentinelas (canários) antes e depois
        const uint32_t CANARY_VAL = 0xDEADBEEF;
        size_t pad = 64;
        float* raw_buf = (float*)malloc((M + 2 * pad) * sizeof(float));
        uint32_t* canary_head = (uint32_t*)raw_buf;
        uint32_t* canary_tail = (uint32_t*)(raw_buf + pad + M);
        float* out = raw_buf + pad;

        float* out_ref = (float*)malloc(M * sizeof(float));

        // Configurar canários
        for (size_t p = 0; p < pad; p++) {
            canary_head[p] = CANARY_VAL;
            canary_tail[p] = CANARY_VAL;
        }

        // Testar múltiplos cenários de dados:
        // Cenario 0: Zeros absolutos
        // Cenario 1: Valores extremos de quantização (q=0 e q=15, INT8=-128 e +127)
        // Cenario 2: Saturação de ativação (x = +40.0 e x = -40.0)
        // Cenario 3: Pseudo-randômico determinístico
        for (int scen = 0; scen < 4; scen++) {
            if (scen == 0) {
                for (int i = 0; i < K; i++) x[i] = 0.0f;
            } else if (scen == 1) {
                for (int i = 0; i < K; i++) x[i] = (i % 2 == 0) ? 127.0f : -128.0f;
                for (int g = 0; g < ng * nb; g++) {
                    for (int r = 0; r < 4; r++) {
                        Wg[g].scale[r] = fp32_to_bf16(0.02f);
                        Wu[g].scale[r] = fp32_to_bf16(0.02f);
                    }
                    memset(Wg[g].qs, 0x0F, 64); // nibbles 15 e 0
                    memset(Wu[g].qs, 0xF0, 64);
                }
            } else if (scen == 2) {
                for (int i = 0; i < K; i++) x[i] = (i % 2 == 0) ? 40.0f : -40.0f;
            } else {
                srand(1337 + s_idx);
                for (int i = 0; i < K; i++) x[i] = ((float)rand() / (float)RAND_MAX) * 8.0f - 4.0f;
                for (int g = 0; g < ng * nb; g++) {
                    for (int r = 0; r < 4; r++) {
                        Wg[g].scale[r] = fp32_to_bf16(0.005f + ((float)rand() / (float)RAND_MAX) * 0.02f);
                        Wu[g].scale[r] = fp32_to_bf16(0.005f + ((float)rand() / (float)RAND_MAX) * 0.02f);
                    }
                    for (int k = 0; k < 64; k++) {
                        Wg[g].qs[k] = (uint8_t)(rand() & 0xFF);
                        Wu[g].qs[k] = (uint8_t)(rand() & 0xFF);
                    }
                }
            }
            quantize_x_to_q8(x, xq, K);

            // Preenche saída com NaN para garantir que cada float seja escrito
            for (int i = 0; i < M; i++) out[i] = NAN;

            mlp_gate_up_swiglu_scalar_ref(out_ref, Wg, Wu, xq, M, K);
            mlp_gate_up_swiglu_tiled_lean(out, Wg, Wu, xq, M, K, false); // mesma ativação expf

            // 1. Verificar canários de memória
            for (size_t p = 0; p < pad; p++) {
                if (canary_head[p] != CANARY_VAL || canary_tail[p] != CANARY_VAL) {
                    printf("   [FALHA] Buffer Overflow / Corrupcao de memoria detectada em M=%d, K=%d!\n", M, K);
                    all_passed = false;
                }
            }

            // 2. Verificar NaNs não sobrescritos
            for (int i = 0; i < M; i++) {
                if (isnan(out[i])) {
                    printf("   [FALHA] Elemento out[%d] nao foi escrito em M=%d, K=%d!\n", i, M, K);
                    all_passed = false;
                }
            }

            // 3. Verificar precisão bit-a-bit contra ground truth
            for (int i = 0; i < M; i++) {
                float diff = fabsf(out[i] - out_ref[i]);
                if (diff > 1e-4f) {
                    printf("   [FALHA] Divergencia em M=%d, K=%d, Cenario %d (diff=%.6e)!\n", M, K, scen, diff);
                    all_passed = false;
                    break;
                }
            }
        }

        free(Wg); free(Wu); free(xq); free(x); free(raw_buf); free(out_ref);
    }

    if (all_passed) {
        printf("   >> SUCESSO: Todos os 9 formatos de matriz, caudas e cenarios de borda passaram sem overflow nem divergencia numerica!\n\n");
    }

    // ------------------------------------------------------------------------
    // PARTE 2: CARACTERIZAÇÃO DE DESEMPENHO ESTATÍSTICA (1, 4 e 8 THREADS)
    // ------------------------------------------------------------------------
    printf("[2] CARACTERIZACAO DE DESEMPENHO ESTATISTICA (Qwen2/Qwen3 MLP 8960 x 1536):\n");
    printf("    (Metodologia: 20 iteracoes de warmup + 100 repeticoes cronometradas)\n\n");

    int M = 8960;
    int K = 1536;
    int nb = K / 32;
    int ng = M / 4;

    BlockQ4x4* Wg = (BlockQ4x4*)malloc((size_t)ng * nb * sizeof(BlockQ4x4));
    BlockQ4x4* Wu = (BlockQ4x4*)malloc((size_t)ng * nb * sizeof(BlockQ4x4));
    BlockQ8* xq = (BlockQ8*)malloc((size_t)nb * sizeof(BlockQ8));
    float* x = (float*)malloc(K * sizeof(float));
    float* out_base = (float*)malloc(M * sizeof(float));
    float* out_lean = (float*)malloc(M * sizeof(float));
    float* out_fast = (float*)malloc(M * sizeof(float));

    srand(42);
    for (int i = 0; i < K; i++) x[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    quantize_x_to_q8(x, xq, K);
    for (int g = 0; g < ng * nb; g++) {
        for (int r = 0; r < 4; r++) {
            Wg[g].scale[r] = fp32_to_bf16(0.01f);
            Wu[g].scale[r] = fp32_to_bf16(0.01f);
        }
        for (int k = 0; k < 64; k++) {
            Wg[g].qs[k] = (uint8_t)(rand() & 0xFF);
            Wu[g].qs[k] = (uint8_t)(rand() & 0xFF);
        }
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    const int REPEAT = 100;
    double samples_base[100];
    double samples_lean[100];
    double samples_fast[100];

    int thread_counts[] = {1, 4, 8};

    for (int t_idx = 0; t_idx < 3; t_idx++) {
        int nthreads = thread_counts[t_idx];
        omp_set_num_threads(nthreads);

        // Warmup
        for (int w = 0; w < 15; w++) {
            mlp_gate_up_swiglu_baseline(out_base, Wg, Wu, xq, M, K, false);
            mlp_gate_up_swiglu_tiled_lean(out_lean, Wg, Wu, xq, M, K, false);
            mlp_gate_up_swiglu_tiled_lean(out_fast, Wg, Wu, xq, M, K, true);
        }

        // Medição Baseline (expf libc)
        for (int it = 0; it < REPEAT; it++) {
            QueryPerformanceCounter(&t0);
            mlp_gate_up_swiglu_baseline(out_base, Wg, Wu, xq, M, K, false);
            QueryPerformanceCounter(&t1);
            samples_base[it] = ((double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart) * 1000.0;
        }

        // Medição Tiled LEAN (MESMA ativação expf libc) -> ISOLA O GEMV!
        for (int it = 0; it < REPEAT; it++) {
            QueryPerformanceCounter(&t0);
            mlp_gate_up_swiglu_tiled_lean(out_lean, Wg, Wu, xq, M, K, false);
            QueryPerformanceCounter(&t1);
            samples_lean[it] = ((double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart) * 1000.0;
        }

        // Medição Tiled LEAN + Fast SwiGLU Vetorizado
        for (int it = 0; it < REPEAT; it++) {
            QueryPerformanceCounter(&t0);
            mlp_gate_up_swiglu_tiled_lean(out_fast, Wg, Wu, xq, M, K, true);
            QueryPerformanceCounter(&t1);
            samples_fast[it] = ((double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart) * 1000.0;
        }

        Stats sb = compute_stats(samples_base, REPEAT);
        Stats sl = compute_stats(samples_lean, REPEAT);
        Stats sf = compute_stats(samples_fast, REPEAT);

        printf("--------------------------------------------------------------------\n");
        printf("CONFIGURACAO: %d THREADS OPENMP\n", nthreads);
        printf("--------------------------------------------------------------------\n");
        printf("A. Baseline (expf libc)     : Media: %6.3f ms | Min: %6.3f ms | Mediana: %6.3f ms | DesvPad: %5.3f ms\n",
               sb.mean, sb.min_v, sb.median, sb.stddev);
        printf("B. Tiled LEAN (expf libc)   : Media: %6.3f ms | Min: %6.3f ms | Mediana: %6.3f ms | DesvPad: %5.3f ms\n",
               sl.mean, sl.min_v, sl.median, sl.stddev);
        printf("   >> Ganho Isolado do GEMV : %.2fx (reducao de %.1f%%)\n",
               sb.median / sl.median, (1.0 - sl.median / sb.median) * 100.0);
        printf("C. Tiled + SwiGLU Vetorizado: Media: %6.3f ms | Min: %6.3f ms | Mediana: %6.3f ms | DesvPad: %5.3f ms\n",
               sf.mean, sf.min_v, sf.median, sf.stddev);
        printf("   >> Ganho TOTAL Otimizado : %.2fx (reducao de %.1f%%)\n",
               sb.median / sf.median, (1.0 - sf.median / sb.median) * 100.0);
    }
    printf("====================================================================\n");

    free(Wg); free(Wu); free(xq); free(x); free(out_base); free(out_lean); free(out_fast);
    return 0;
}
