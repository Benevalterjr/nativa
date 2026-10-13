#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>
#include <omp.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t scale;
    uint8_t qs[16];
} BlockQ4;
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

void mlp_gate_up_swiglu_fast(float* restrict h_mlp, const BlockQ4* restrict Wgate, const BlockQ4* restrict Wup, const BlockQ8* restrict x_q8, int rows, int cols) {
    int num_blocks = cols / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i += 4) {
        if (i + 3 < rows) {
            const BlockQ4* g_row0 = Wgate + (size_t)i * num_blocks;
            const BlockQ4* g_row1 = Wgate + (size_t)(i + 1) * num_blocks;
            const BlockQ4* g_row2 = Wgate + (size_t)(i + 2) * num_blocks;
            const BlockQ4* g_row3 = Wgate + (size_t)(i + 3) * num_blocks;

            const BlockQ4* u_row0 = Wup + (size_t)i * num_blocks;
            const BlockQ4* u_row1 = Wup + (size_t)(i + 1) * num_blocks;
            const BlockQ4* u_row2 = Wup + (size_t)(i + 2) * num_blocks;
            const BlockQ4* u_row3 = Wup + (size_t)(i + 3) * num_blocks;

            __m128 acc_g = _mm_setzero_ps();
            __m128 acc_u = _mm_setzero_ps();

            for (int b = 0; b < num_blocks; b++) {
                const BlockQ8* xblk = &x_q8[b];
                __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
                __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

                // Gate
                const BlockQ4* g0 = &g_row0[b];
                const BlockQ4* g1 = &g_row1[b];
                const BlockQ4* g2 = &g_row2[b];
                const BlockQ4* g3 = &g_row3[b];

                __m128i gr0 = _mm_loadu_si128((const __m128i*)g0->qs);
                __m128i gr1 = _mm_loadu_si128((const __m128i*)g1->qs);
                __m128i gr2 = _mm_loadu_si128((const __m128i*)g2->qs);
                __m128i gr3 = _mm_loadu_si128((const __m128i*)g3->qs);

                __m128i gtot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(gr0, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr0, 4), mask), x_hi), ones));
                __m128i gtot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(gr1, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr1, 4), mask), x_hi), ones));
                __m128i gtot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(gr2, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr2, 4), mask), x_hi), ones));
                __m128i gtot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(gr3, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr3, 4), mask), x_hi), ones));

                // Up
                const BlockQ4* u0 = &u_row0[b];
                const BlockQ4* u1 = &u_row1[b];
                const BlockQ4* u2 = &u_row2[b];
                const BlockQ4* u3 = &u_row3[b];

                __m128i ur0 = _mm_loadu_si128((const __m128i*)u0->qs);
                __m128i ur1 = _mm_loadu_si128((const __m128i*)u1->qs);
                __m128i ur2 = _mm_loadu_si128((const __m128i*)u2->qs);
                __m128i ur3 = _mm_loadu_si128((const __m128i*)u3->qs);

                __m128i utot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(ur0, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur0, 4), mask), x_hi), ones));
                __m128i utot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(ur1, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur1, 4), mask), x_hi), ones));
                __m128i utot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(ur2, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur2, 4), mask), x_hi), ones));
                __m128i utot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(ur3, mask), x_lo), ones),
                                             _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur3, 4), mask), x_hi), ones));

                // 3 instructions reduce all 4 rows
                __m128i gh_all = _mm_hadd_epi32(_mm_hadd_epi32(gtot0, gtot1), _mm_hadd_epi32(gtot2, gtot3));
                __m128i uh_all = _mm_hadd_epi32(_mm_hadd_epi32(utot0, utot1), _mm_hadd_epi32(utot2, utot3));

                __m128i adj = _mm_set1_epi32(xblk->sum_adj);
                __m128 g_fdots = _mm_cvtepi32_ps(_mm_sub_epi32(gh_all, adj));
                __m128 u_fdots = _mm_cvtepi32_ps(_mm_sub_epi32(uh_all, adj));

                float x_sc = xblk->scale;
                __m128 g_scales = _mm_set_ps(bf16_to_fp32(g3->scale) * x_sc,
                                             bf16_to_fp32(g2->scale) * x_sc,
                                             bf16_to_fp32(g1->scale) * x_sc,
                                             bf16_to_fp32(g0->scale) * x_sc);
                __m128 u_scales = _mm_set_ps(bf16_to_fp32(u3->scale) * x_sc,
                                             bf16_to_fp32(u2->scale) * x_sc,
                                             bf16_to_fp32(u1->scale) * x_sc,
                                             bf16_to_fp32(u0->scale) * x_sc);

                acc_g = _mm_add_ps(acc_g, _mm_mul_ps(g_fdots, g_scales));
                acc_u = _mm_add_ps(acc_u, _mm_mul_ps(u_fdots, u_scales));
            }

            float g_arr[4], u_arr[4];
            _mm_storeu_ps(g_arr, acc_g);
            _mm_storeu_ps(u_arr, acc_u);

            h_mlp[i]     = (g_arr[0] / (1.0f + expf(-g_arr[0]))) * u_arr[0];
            h_mlp[i + 1] = (g_arr[1] / (1.0f + expf(-g_arr[1]))) * u_arr[1];
            h_mlp[i + 2] = (g_arr[2] / (1.0f + expf(-g_arr[2]))) * u_arr[2];
            h_mlp[i + 3] = (g_arr[3] / (1.0f + expf(-g_arr[3]))) * u_arr[3];
        }
    }
}

int main(void) {
    int rows = 8960;
    int cols = 1536;
    int num_blocks = cols / 32;

    BlockQ4* Wgate = (BlockQ4*)malloc(rows * num_blocks * sizeof(BlockQ4));
    BlockQ4* Wup   = (BlockQ4*)malloc(rows * num_blocks * sizeof(BlockQ4));
    BlockQ8* x_q8  = (BlockQ8*)malloc(num_blocks * sizeof(BlockQ8));
    float* x = (float*)malloc(cols * sizeof(float));
    float* h_mlp = (float*)malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    for (int i = 0; i < rows * num_blocks; i++) {
        Wgate[i].scale = fp32_to_bf16(0.01f);
        Wup[i].scale   = fp32_to_bf16(0.01f);
        for (int k = 0; k < 16; k++) {
            Wgate[i].qs[k] = 0x88;
            Wup[i].qs[k]   = 0x88;
        }
    }
    quantize_x_to_q8(x, x_q8, cols);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    mlp_gate_up_swiglu_fast(h_mlp, Wgate, Wup, x_q8, rows, cols);

    QueryPerformanceCounter(&t0);
    int iters = 200;
    for (int it = 0; it < iters; it++) {
        mlp_gate_up_swiglu_fast(h_mlp, Wgate, Wup, x_q8, rows, cols);
    }
    QueryPerformanceCounter(&t1);

    double dt = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double per_iter_ms = (dt / iters) * 1000.0;
    printf("Tempo medio Gate+Up+SwiGLU Fused Fast (8960x1536): %.3f ms\n", per_iter_ms);

    free(Wgate);
    free(Wup);
    free(x_q8);
    free(x);
    free(h_mlp);
    return 0;
}
