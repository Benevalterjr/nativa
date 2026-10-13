// ============================================================================
// NATIVA NATIVE C RUNNER - QWEN2.5-1.5B-INSTRUCT IN INT4 (ULTRAFAST Q4_0 x Q8_0)
// ============================================================================
// Motor de inferência nativo de altíssima performance para Qwen2.5-1.5B INT4:
//   - Zero-Copy Win32 mmap do modelo quantizado (828 MB)
//   - Kernel SSSE3 4-Row Fused Vector Reduction (_mm_maddubs + _mm_hadd_epi32)
//   - Redução total de overhead de descompactação e memória
//   - OpenMP multi-core com active wait policy
//   - Protocolo interativo nativo compatível com Chat_Com_Qwen_1.5B.bat
// ============================================================================

#define _CRT_SECURE_NO_WARNINGS 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include <omp.h>

#include "qwen_offsets_1.5b_q4.h"
#include "nativa_sampling.h"

#define MAX_SEQ_LEN 1024

#pragma pack(push, 1)
typedef struct {
    uint16_t scale; // BF16 scale
    uint8_t  qs[16]; // 16 bytes contendo 32 nibbles
} BlockQ4;
#pragma pack(pop)

typedef struct {
    float scale;
    int8_t qs[32];
    int32_t sum_adj; // 8 * sum(qs[0..31])
} BlockQ8;

// Layout Q4x4: 4 linhas intercaladas por bloco (mesmo tamanho total que 4x BlockQ4)
#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4]; // escalas BF16 das 4 linhas, contiguas
    uint8_t  qs[64];   // 4 x 16 bytes de nibbles (linha 0, 1, 2, 3)
} BlockQ4x4;
#pragma pack(pop)

// ----------------------------------------------------------------------------
// VOCABULÁRIO NATIVO
// ----------------------------------------------------------------------------
typedef struct {
    uint16_t len;
    char* str;
} VocabEntry;

static VocabEntry* g_vocab = NULL;
static uint32_t g_vocab_size = 0;

static bool load_vocab(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "rb");
    if (!f) return false;
    if (fread(&g_vocab_size, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    g_vocab = (VocabEntry*)malloc(g_vocab_size * sizeof(VocabEntry));
    if (!g_vocab) { fclose(f); return false; }

    uint32_t i;
    for (i = 0; i < g_vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) {
            g_vocab[i].len = 0;
            g_vocab[i].str = "";
            continue;
        }
        g_vocab[i].len = len;
        g_vocab[i].str = (char*)malloc(len + 1);
        if (len > 0) {
            fread(g_vocab[i].str, 1, len, f);
        }
        g_vocab[i].str[len] = '\0';
    }
    fclose(f);
    return true;
}

static inline void print_token(int token_id) {
    if (token_id >= 0 && (uint32_t)token_id < g_vocab_size && g_vocab[token_id].len > 0) {
        fwrite(g_vocab[token_id].str, 1, g_vocab[token_id].len, stdout);
        fflush(stdout);
    }
}

// ----------------------------------------------------------------------------
// TIMER WINDOWS
// ----------------------------------------------------------------------------
static LARGE_INTEGER g_timer_freq;

static inline void timer_init(void) {
    QueryPerformanceFrequency(&g_timer_freq);
}

static inline double timer_now(void) {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)g_timer_freq.QuadPart;
}

// ----------------------------------------------------------------------------
// MEMORY ARENA GLOBAL
// ----------------------------------------------------------------------------
static float g_k_cache[QWEN15_NUM_LAYERS][MAX_SEQ_LEN][QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];
static float g_v_cache[QWEN15_NUM_LAYERS][MAX_SEQ_LEN][QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];

static float g_x[QWEN15_DIM];
static float g_xn[QWEN15_DIM];
static float g_q[QWEN15_DIM];
static float g_k[QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];
static float g_v[QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];
static float g_attn_out[QWEN15_DIM];
static float g_gate[QWEN15_INTERMEDIATE_DIM];
static float g_up[QWEN15_INTERMEDIATE_DIM];
static float g_h_mlp[QWEN15_INTERMEDIATE_DIM];
static float g_x_final[QWEN15_DIM];
static float g_logits[QWEN15_VOCAB_SIZE];
static float g_scores[QWEN15_NUM_HEADS][MAX_SEQ_LEN];

static BlockQ8 g_xn_q8[QWEN15_DIM / 32];
static BlockQ8 g_attn_q8[QWEN15_DIM / 32];
static BlockQ8 g_h_mlp_q8[QWEN15_INTERMEDIATE_DIM / 32];
static BlockQ8 g_x_final_q8[QWEN15_DIM / 32];

static float g_rope_freqs[64];

static void init_rope_freqs(void) {
    int i;
    for (i = 0; i < 64; i++) {
        g_rope_freqs[i] = 1.0f / powf(QWEN15_ROPE_THETA, (float)i / 64.0f);
    }
}

// ----------------------------------------------------------------------------
// OPERAÇÕES MATEMÁTICAS SIMD & QUANTIZAÇÃO DE ATIVAÇÕES
// ----------------------------------------------------------------------------
static inline float bf16_to_fp32(uint16_t v) {
    union { uint32_t u; float f; } conv;
    conv.u = ((uint32_t)v) << 16;
    return conv.f;
}

static inline void quantize_x_to_q8(const float* restrict x, BlockQ8* restrict x_q8, int cols) {
    int num_blocks = cols / 32;
    int b, i;
    for (b = 0; b < num_blocks; b++) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (i = 0; i < 32; i++) {
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
        for (i = 0; i < 32; i++) {
            int q = (int)roundf(xb[i] * inv_scale);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            x_q8[b].qs[i] = (int8_t)q;
            s += q;
        }
        x_q8[b].sum_adj = s * 8;
    }
}

static inline void rmsnorm(float* restrict out, const float* restrict x, const uint16_t* restrict weight, int dim, float eps) {
    float sum_sq = 0.0f;
    int i;
    for (i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (i = 0; i < dim; i++) {
        union { uint32_t u; float f; } wconv;
        wconv.u = ((uint32_t)weight[i]) << 16;
        out[i] = x[i] * inv_rms * wconv.f;
    }
}

static inline void apply_rope(float* vec, int pos, int heads) {
    int h, i;
    for (h = 0; h < heads; h++) {
        float* head_ptr = vec + h * 128;
        for (i = 0; i < 64; i++) {
            float angle = (float)pos * g_rope_freqs[i];
            float cos_v = cosf(angle);
            float sin_v = sinf(angle);
            float x1 = head_ptr[i];
            float x2 = head_ptr[i + 64];
            head_ptr[i] = x1 * cos_v - x2 * sin_v;
            head_ptr[i + 64] = x1 * sin_v + x2 * cos_v;
        }
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ULTRAFAST 4-ROW VECTOR REDUCTION KERNELS (ZERO SCALAR OVERHEAD)
// ----------------------------------------------------------------------------
static inline void matvec_q4_q8_fast4(float* restrict y, const BlockQ4* restrict W, const BlockQ8* restrict x_q8, const uint16_t* restrict bias, int rows, int cols) {
    int num_blocks = cols / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i += 4) {
        if (i + 3 < rows) {
            const BlockQ4* row0 = W + (size_t)i * num_blocks;
            const BlockQ4* row1 = W + (size_t)(i + 1) * num_blocks;
            const BlockQ4* row2 = W + (size_t)(i + 2) * num_blocks;
            const BlockQ4* row3 = W + (size_t)(i + 3) * num_blocks;
            __m128 acc_rows = _mm_setzero_ps();
            int b;

            for (b = 0; b < num_blocks; b++) {
                const BlockQ8* xblk = &x_q8[b];
                __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
                __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

                const BlockQ4* blk0 = &row0[b];
                __m128i raw0 = _mm_loadu_si128((const __m128i*)blk0->qs);
                __m128i tot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw0, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask), x_hi), ones));

                const BlockQ4* blk1 = &row1[b];
                __m128i raw1 = _mm_loadu_si128((const __m128i*)blk1->qs);
                __m128i tot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw1, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask), x_hi), ones));

                const BlockQ4* blk2 = &row2[b];
                __m128i raw2 = _mm_loadu_si128((const __m128i*)blk2->qs);
                __m128i tot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw2, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw2, 4), mask), x_hi), ones));

                const BlockQ4* blk3 = &row3[b];
                __m128i raw3 = _mm_loadu_si128((const __m128i*)blk3->qs);
                __m128i tot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw3, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw3, 4), mask), x_hi), ones));

                // Reducao vetorial das 4 linhas em 1 registrador
                __m128i h01 = _mm_hadd_epi32(tot0, tot1);
                __m128i h23 = _mm_hadd_epi32(tot2, tot3);
                __m128i h_all = _mm_hadd_epi32(h01, h23);

                __m128i adj = _mm_set1_epi32(xblk->sum_adj);
                __m128 fdots = _mm_cvtepi32_ps(_mm_sub_epi32(h_all, adj));

                // Desempacotamento vetorial SIMD BF16 -> FP32 (sem acessos a pilha)
                __m128i s_u16 = _mm_set_epi16(0, 0, 0, 0, blk3->scale, blk2->scale, blk1->scale, blk0->scale);
                __m128i s_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), s_u16);
                __m128 scales = _mm_mul_ps(_mm_castsi128_ps(s_u32), _mm_set1_ps(xblk->scale));

                acc_rows = _mm_add_ps(acc_rows, _mm_mul_ps(fdots, scales));
            }

            if (bias) {
                __m128i bs_u16 = _mm_set_epi16(0, 0, 0, 0, bias[i + 3], bias[i + 2], bias[i + 1], bias[i]);
                __m128i bs_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), bs_u16);
                acc_rows = _mm_add_ps(acc_rows, _mm_castsi128_ps(bs_u32));
            }
            _mm_storeu_ps(&y[i], acc_rows);
        } else {
            int r;
            for (r = i; r < rows; r++) {
                const BlockQ4* row_r = W + (size_t)r * num_blocks;
                float sum_r = 0.0f;
                int b;
                for (b = 0; b < num_blocks; b++) {
                    const BlockQ8* xblk = &x_q8[b];
                    const BlockQ4* blk = &row_r[b];
                    __m128i raw = _mm_loadu_si128((const __m128i*)blk->qs);
                    __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
                    __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));
                    __m128i tot = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw, mask), x_lo), ones),
                                                _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), x_hi), ones));
                    __m128i h = _mm_hadd_epi32(_mm_hadd_epi32(tot, tot), _mm_hadd_epi32(tot, tot));
                    sum_r += (float)(_mm_cvtsi128_si32(h) - xblk->sum_adj) * (bf16_to_fp32(blk->scale) * xblk->scale);
                }
                if (bias) sum_r += bf16_to_fp32(bias[r]);
                y[r] = sum_r;
            }
        }
    }
}

static inline void matvec_q4_q8_fast4_add(float* restrict y, const BlockQ4* restrict W, const BlockQ8* restrict x_q8, int rows, int cols) {
    int num_blocks = cols / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i += 4) {
        if (i + 3 < rows) {
            const BlockQ4* row0 = W + (size_t)i * num_blocks;
            const BlockQ4* row1 = W + (size_t)(i + 1) * num_blocks;
            const BlockQ4* row2 = W + (size_t)(i + 2) * num_blocks;
            const BlockQ4* row3 = W + (size_t)(i + 3) * num_blocks;
            __m128 acc_rows = _mm_setzero_ps();
            int b;

            for (b = 0; b < num_blocks; b++) {
                const BlockQ8* xblk = &x_q8[b];
                __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
                __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

                const BlockQ4* blk0 = &row0[b];
                __m128i raw0 = _mm_loadu_si128((const __m128i*)blk0->qs);
                __m128i tot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw0, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask), x_hi), ones));

                const BlockQ4* blk1 = &row1[b];
                __m128i raw1 = _mm_loadu_si128((const __m128i*)blk1->qs);
                __m128i tot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw1, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask), x_hi), ones));

                const BlockQ4* blk2 = &row2[b];
                __m128i raw2 = _mm_loadu_si128((const __m128i*)blk2->qs);
                __m128i tot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw2, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw2, 4), mask), x_hi), ones));

                const BlockQ4* blk3 = &row3[b];
                __m128i raw3 = _mm_loadu_si128((const __m128i*)blk3->qs);
                __m128i tot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw3, mask), x_lo), ones),
                                            _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw3, 4), mask), x_hi), ones));

                __m128i h01 = _mm_hadd_epi32(tot0, tot1);
                __m128i h23 = _mm_hadd_epi32(tot2, tot3);
                __m128i h_all = _mm_hadd_epi32(h01, h23);

                __m128i adj = _mm_set1_epi32(xblk->sum_adj);
                __m128 fdots = _mm_cvtepi32_ps(_mm_sub_epi32(h_all, adj));

                __m128i s_u16 = _mm_set_epi16(0, 0, 0, 0, blk3->scale, blk2->scale, blk1->scale, blk0->scale);
                __m128i s_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), s_u16);
                __m128 scales = _mm_mul_ps(_mm_castsi128_ps(s_u32), _mm_set1_ps(xblk->scale));

                acc_rows = _mm_add_ps(acc_rows, _mm_mul_ps(fdots, scales));
            }

            __m128 cur = _mm_loadu_ps(&y[i]);
            _mm_storeu_ps(&y[i], _mm_add_ps(cur, acc_rows));
        } else {
            int r;
            for (r = i; r < rows; r++) {
                const BlockQ4* row_r = W + (size_t)r * num_blocks;
                float sum_r = 0.0f;
                int b;
                for (b = 0; b < num_blocks; b++) {
                    const BlockQ8* xblk = &x_q8[b];
                    const BlockQ4* blk = &row_r[b];
                    __m128i raw = _mm_loadu_si128((const __m128i*)blk->qs);
                    __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
                    __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));
                    __m128i tot = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw, mask), x_lo), ones),
                                                _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), x_hi), ones));
                    __m128i h = _mm_hadd_epi32(_mm_hadd_epi32(tot, tot), _mm_hadd_epi32(tot, tot));
                    sum_r += (float)(_mm_cvtsi128_si32(h) - xblk->sum_adj) * (bf16_to_fp32(blk->scale) * xblk->scale);
                }
                y[r] += sum_r;
            }
        }
    }
}

// Projecoes Q, K, V fundidas em 1 unica regiao paralela (2048 linhas)
static inline void compute_qkv_fused(float* restrict q, float* restrict k, float* restrict v,
                                     const BlockQ4* restrict wq, const BlockQ4* restrict wk, const BlockQ4* restrict wv,
                                     const uint16_t* restrict bq, const uint16_t* restrict bk, const uint16_t* restrict bv,
                                     const BlockQ8* restrict xn_q8, int dim) {
    int num_blocks = dim / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);

    int idx;
    #pragma omp parallel for schedule(static)
    for (idx = 0; idx < 2048; idx += 4) {
        float* y;
        const BlockQ4* W_base;
        const uint16_t* b_base;
        int r;

        if (idx < 1536) {
            y = q;
            W_base = wq;
            b_base = bq;
            r = idx;
        } else if (idx < 1792) {
            y = k;
            W_base = wk;
            b_base = bk;
            r = idx - 1536;
        } else {
            y = v;
            W_base = wv;
            b_base = bv;
            r = idx - 1792;
        }

        const BlockQ4* row0 = W_base + (size_t)r * num_blocks;
        const BlockQ4* row1 = W_base + (size_t)(r + 1) * num_blocks;
        const BlockQ4* row2 = W_base + (size_t)(r + 2) * num_blocks;
        const BlockQ4* row3 = W_base + (size_t)(r + 3) * num_blocks;
        __m128 acc_rows = _mm_setzero_ps();
        int b;

        for (b = 0; b < num_blocks; b++) {
            const BlockQ8* xblk = &xn_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

            const BlockQ4* blk0 = &row0[b];
            __m128i raw0 = _mm_loadu_si128((const __m128i*)blk0->qs);
            __m128i tot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw0, mask), x_lo), ones),
                                        _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw0, 4), mask), x_hi), ones));

            const BlockQ4* blk1 = &row1[b];
            __m128i raw1 = _mm_loadu_si128((const __m128i*)blk1->qs);
            __m128i tot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw1, mask), x_lo), ones),
                                        _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw1, 4), mask), x_hi), ones));

            const BlockQ4* blk2 = &row2[b];
            __m128i raw2 = _mm_loadu_si128((const __m128i*)blk2->qs);
            __m128i tot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw2, mask), x_lo), ones),
                                        _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw2, 4), mask), x_hi), ones));

            const BlockQ4* blk3 = &row3[b];
            __m128i raw3 = _mm_loadu_si128((const __m128i*)blk3->qs);
            __m128i tot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(raw3, mask), x_lo), ones),
                                        _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(raw3, 4), mask), x_hi), ones));

            __m128i h01 = _mm_hadd_epi32(tot0, tot1);
            __m128i h23 = _mm_hadd_epi32(tot2, tot3);
            __m128i h_all = _mm_hadd_epi32(h01, h23);

            __m128i adj = _mm_set1_epi32(xblk->sum_adj);
            __m128 fdots = _mm_cvtepi32_ps(_mm_sub_epi32(h_all, adj));

            __m128i s_u16 = _mm_set_epi16(0, 0, 0, 0, blk3->scale, blk2->scale, blk1->scale, blk0->scale);
            __m128i s_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), s_u16);
            __m128 scales = _mm_mul_ps(_mm_castsi128_ps(s_u32), _mm_set1_ps(xblk->scale));

            acc_rows = _mm_add_ps(acc_rows, _mm_mul_ps(fdots, scales));
        }

        if (b_base) {
            __m128i bs_u16 = _mm_set_epi16(0, 0, 0, 0, b_base[r + 3], b_base[r + 2], b_base[r + 1], b_base[r]);
            __m128i bs_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), bs_u16);
            acc_rows = _mm_add_ps(acc_rows, _mm_castsi128_ps(bs_u32));
        }
        _mm_storeu_ps(&y[r], acc_rows);
    }
}

// MLP Fused: Gate (4 linhas) + Up (4 linhas) + SiLU fundidos sem barreiras intermediarias
static inline void compute_mlp_fused(float* restrict h_mlp,
                                     const BlockQ4* restrict wgate_w,
                                     const BlockQ4* restrict wup_w,
                                     const BlockQ8* restrict xn_q8,
                                     int intermediate_dim,
                                     int dim) {
    int num_blocks = dim / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < intermediate_dim; i += 4) {
        // 1. Rajada sequencial de 4 linhas de Gate
        const BlockQ4* g_row0 = wgate_w + (size_t)i * num_blocks;
        const BlockQ4* g_row1 = wgate_w + (size_t)(i + 1) * num_blocks;
        const BlockQ4* g_row2 = wgate_w + (size_t)(i + 2) * num_blocks;
        const BlockQ4* g_row3 = wgate_w + (size_t)(i + 3) * num_blocks;
        __m128 acc_gate = _mm_setzero_ps();
        int b;

        for (b = 0; b < num_blocks; b++) {
            const BlockQ8* xblk = &xn_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

            const BlockQ4* gb0 = &g_row0[b];
            __m128i graw0 = _mm_loadu_si128((const __m128i*)gb0->qs);
            __m128i gtot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(graw0, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(graw0, 4), mask), x_hi), ones));

            const BlockQ4* gb1 = &g_row1[b];
            __m128i graw1 = _mm_loadu_si128((const __m128i*)gb1->qs);
            __m128i gtot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(graw1, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(graw1, 4), mask), x_hi), ones));

            const BlockQ4* gb2 = &g_row2[b];
            __m128i graw2 = _mm_loadu_si128((const __m128i*)gb2->qs);
            __m128i gtot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(graw2, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(graw2, 4), mask), x_hi), ones));

            const BlockQ4* gb3 = &g_row3[b];
            __m128i graw3 = _mm_loadu_si128((const __m128i*)gb3->qs);
            __m128i gtot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(graw3, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(graw3, 4), mask), x_hi), ones));

            __m128i gh01 = _mm_hadd_epi32(gtot0, gtot1);
            __m128i gh23 = _mm_hadd_epi32(gtot2, gtot3);
            __m128i gh_all = _mm_hadd_epi32(gh01, gh23);

            __m128i adj = _mm_set1_epi32(xblk->sum_adj);
            __m128 gfdots = _mm_cvtepi32_ps(_mm_sub_epi32(gh_all, adj));

            __m128i gs_u16 = _mm_set_epi16(0, 0, 0, 0, gb3->scale, gb2->scale, gb1->scale, gb0->scale);
            __m128i gs_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), gs_u16);
            __m128 gscales = _mm_mul_ps(_mm_castsi128_ps(gs_u32), _mm_set1_ps(xblk->scale));

            acc_gate = _mm_add_ps(acc_gate, _mm_mul_ps(gfdots, gscales));
        }

        // 2. Rajada sequencial de 4 linhas de Up
        const BlockQ4* u_row0 = wup_w + (size_t)i * num_blocks;
        const BlockQ4* u_row1 = wup_w + (size_t)(i + 1) * num_blocks;
        const BlockQ4* u_row2 = wup_w + (size_t)(i + 2) * num_blocks;
        const BlockQ4* u_row3 = wup_w + (size_t)(i + 3) * num_blocks;
        __m128 acc_up = _mm_setzero_ps();

        for (b = 0; b < num_blocks; b++) {
            const BlockQ8* xblk = &xn_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

            const BlockQ4* ub0 = &u_row0[b];
            __m128i uraw0 = _mm_loadu_si128((const __m128i*)ub0->qs);
            __m128i utot0 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(uraw0, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(uraw0, 4), mask), x_hi), ones));

            const BlockQ4* ub1 = &u_row1[b];
            __m128i uraw1 = _mm_loadu_si128((const __m128i*)ub1->qs);
            __m128i utot1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(uraw1, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(uraw1, 4), mask), x_hi), ones));

            const BlockQ4* ub2 = &u_row2[b];
            __m128i uraw2 = _mm_loadu_si128((const __m128i*)ub2->qs);
            __m128i utot2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(uraw2, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(uraw2, 4), mask), x_hi), ones));

            const BlockQ4* ub3 = &u_row3[b];
            __m128i uraw3 = _mm_loadu_si128((const __m128i*)ub3->qs);
            __m128i utot3 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(uraw3, mask), x_lo), ones),
                                         _mm_madd_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(uraw3, 4), mask), x_hi), ones));

            __m128i uh01 = _mm_hadd_epi32(utot0, utot1);
            __m128i uh23 = _mm_hadd_epi32(utot2, utot3);
            __m128i uh_all = _mm_hadd_epi32(uh01, uh23);

            __m128i adj = _mm_set1_epi32(xblk->sum_adj);
            __m128 ufdots = _mm_cvtepi32_ps(_mm_sub_epi32(uh_all, adj));

            __m128i us_u16 = _mm_set_epi16(0, 0, 0, 0, ub3->scale, ub2->scale, ub1->scale, ub0->scale);
            __m128i us_u32 = _mm_unpacklo_epi16(_mm_setzero_si128(), us_u16);
            __m128 uscales = _mm_mul_ps(_mm_castsi128_ps(us_u32), _mm_set1_ps(xblk->scale));

            acc_up = _mm_add_ps(acc_up, _mm_mul_ps(ufdots, uscales));
        }

        // 3. Fused SiLU & Up multiply direto para a saida
        float g_vals[4], u_vals[4];
        _mm_storeu_ps(g_vals, acc_gate);
        _mm_storeu_ps(u_vals, acc_up);

        h_mlp[i + 0] = (g_vals[0] / (1.0f + expf(-g_vals[0]))) * u_vals[0];
        h_mlp[i + 1] = (g_vals[1] / (1.0f + expf(-g_vals[1]))) * u_vals[1];
        h_mlp[i + 2] = (g_vals[2] / (1.0f + expf(-g_vals[2]))) * u_vals[2];
        h_mlp[i + 3] = (g_vals[3] / (1.0f + expf(-g_vals[3]))) * u_vals[3];
    }
}

// ----------------------------------------------------------------------------
// KERNELS Q4x4 (pesos intercalados em grupos de 4 linhas)
// ----------------------------------------------------------------------------
// Calcula 4 produtos escalares (4 linhas) contra x_q8. Um unico fluxo de leitura.
static inline __m128 q4x4_dot4(const BlockQ4x4* restrict W4, const BlockQ8* restrict x_q8, int nb) {
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

        // Soma int16 das metades lo/hi antes do madd (|max| = 7680, sem overflow)
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

        // 4 escalas BF16 com uma unica carga de 64 bits
        __m128 sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)blk->scale)));
        acc = _mm_add_ps(acc, _mm_mul_ps(fd, _mm_mul_ps(sc, _mm_set1_ps(xb->scale))));
    }
    return acc;
}

static inline __m128 bf16x4_to_ps(const uint16_t* p) {
    return _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), _mm_loadl_epi64((const __m128i*)p)));
}

// y = W x (+bias) ou y += W x (add=1). rows multiplo de 4.
static inline void matvec_q4x4(float* restrict y, const BlockQ4x4* restrict W, const BlockQ8* restrict x_q8,
                               const uint16_t* restrict bias, int rows, int cols, int add) {
    int nb = cols / 32;
    int groups = rows / 4;
    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        __m128 acc = q4x4_dot4(W + (size_t)g * nb, x_q8, nb);
        if (bias) acc = _mm_add_ps(acc, bf16x4_to_ps(bias + g * 4));
        if (add) acc = _mm_add_ps(acc, _mm_loadu_ps(&y[g * 4]));
        _mm_storeu_ps(&y[g * 4], acc);
    }
}

// Q, K, V em uma unica regiao paralela (512 grupos de 4 linhas)
static inline void compute_qkv_q4x4(float* restrict q, float* restrict k, float* restrict v,
                                    const BlockQ4x4* restrict wq, const BlockQ4x4* restrict wk, const BlockQ4x4* restrict wv,
                                    const uint16_t* restrict bq, const uint16_t* restrict bk, const uint16_t* restrict bv,
                                    const BlockQ8* restrict xn_q8) {
    const int nb = QWEN15_DIM / 32;
    const int gq = QWEN15_DIM / 4;                                   // 384
    const int gk = (QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM) / 4;     // 64
    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < gq + 2 * gk; g++) {
        float* y; const BlockQ4x4* W; const uint16_t* bias; int r;
        if (g < gq)            { y = q; W = wq; bias = bq; r = g; }
        else if (g < gq + gk)  { y = k; W = wk; bias = bk; r = g - gq; }
        else                   { y = v; W = wv; bias = bv; r = g - gq - gk; }
        __m128 acc = q4x4_dot4(W + (size_t)r * nb, xn_q8, nb);
        acc = _mm_add_ps(acc, bf16x4_to_ps(bias + r * 4));
        _mm_storeu_ps(&y[r * 4], acc);
    }
}

// Gate + Up + SiLU em uma unica regiao paralela
static inline void compute_mlp_q4x4(float* restrict h_mlp, const BlockQ4x4* restrict wgate,
                                    const BlockQ4x4* restrict wup, const BlockQ8* restrict xn_q8) {
    const int nb = QWEN15_DIM / 32;
    const int groups = QWEN15_INTERMEDIATE_DIM / 4;
    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        float gv[4], uv[4];
        _mm_storeu_ps(gv, q4x4_dot4(wgate + (size_t)g * nb, xn_q8, nb));
        _mm_storeu_ps(uv, q4x4_dot4(wup + (size_t)g * nb, xn_q8, nb));
        int j;
        for (j = 0; j < 4; j++) {
            h_mlp[g * 4 + j] = (gv[j] / (1.0f + expf(-gv[j]))) * uv[j];
        }
    }
}

// ----------------------------------------------------------------------------
// FORWARD PASS 1.5B INT4
// ----------------------------------------------------------------------------
static double g_prof_attn = 0.0;
static double g_prof_mlp = 0.0;
static double g_prof_lm = 0.0;
static int g_prof_tokens = 0;

static void forward_token(const uint8_t* model_bytes, int token_id, int pos) {
    // 1. Embedding lookup (layout Q4x4: grupo = token/4, linha dentro do grupo = token%4)
    const BlockQ4x4* emb_grp = (const BlockQ4x4*)(model_bytes + g_qwen15_offsets.embed_tokens) + (size_t)(token_id / 4) * (QWEN15_DIM / 32);
    const int lane = token_id % 4;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i off = _mm_set1_epi8(8);

    int b;
    for (b = 0; b < QWEN15_DIM / 32; b++) {
        const BlockQ4x4* blk = &emb_grp[b];
        float scale = bf16_to_fp32(blk->scale[lane]);
        __m128 svec = _mm_set1_ps(scale);

        __m128i raw = _mm_loadu_si128((const __m128i*)(blk->qs + lane * 16));
        __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, mask), off);
        __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), off);

        __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(lo));
        __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 4)));
        __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 12)));

        _mm_storeu_ps(&g_x[b * 32],      _mm_mul_ps(f0, svec));
        _mm_storeu_ps(&g_x[b * 32 + 4],  _mm_mul_ps(f1, svec));
        _mm_storeu_ps(&g_x[b * 32 + 8],  _mm_mul_ps(f2, svec));
        _mm_storeu_ps(&g_x[b * 32 + 12], _mm_mul_ps(f3, svec));

        __m128 f4 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(hi));
        __m128 f5 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
        __m128 f6 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        __m128 f7 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 12)));

        _mm_storeu_ps(&g_x[b * 32 + 16], _mm_mul_ps(f4, svec));
        _mm_storeu_ps(&g_x[b * 32 + 20], _mm_mul_ps(f5, svec));
        _mm_storeu_ps(&g_x[b * 32 + 24], _mm_mul_ps(f6, svec));
        _mm_storeu_ps(&g_x[b * 32 + 28], _mm_mul_ps(f7, svec));
    }

    // 2. Loop através das 28 camadas
    int l;
    for (l = 0; l < QWEN15_NUM_LAYERS; l++) {
        const Qwen15LayerOffsets* lo = &g_qwen15_offsets.layers[l];
        const uint16_t* norm1_w = (const uint16_t*)(model_bytes + lo->norm1);
        const uint16_t* norm2_w = (const uint16_t*)(model_bytes + lo->norm2);
        const BlockQ4x4* wq_w = (const BlockQ4x4*)(model_bytes + lo->wq);
        const uint16_t* bq_w = (const uint16_t*)(model_bytes + lo->bq);
        const BlockQ4x4* wk_w = (const BlockQ4x4*)(model_bytes + lo->wk);
        const uint16_t* bk_w = (const uint16_t*)(model_bytes + lo->bk);
        const BlockQ4x4* wv_w = (const BlockQ4x4*)(model_bytes + lo->wv);
        const uint16_t* bv_w = (const uint16_t*)(model_bytes + lo->bv);
        const BlockQ4x4* wo_w = (const BlockQ4x4*)(model_bytes + lo->wo);
        const BlockQ4x4* wgate_w = (const BlockQ4x4*)(model_bytes + lo->wgate);
        const BlockQ4x4* wup_w = (const BlockQ4x4*)(model_bytes + lo->wup);
        const BlockQ4x4* wdown_w = (const BlockQ4x4*)(model_bytes + lo->wdown);

        double t_a0 = timer_now();
        // Input RMSNorm + Quantizacao Q8
        rmsnorm(g_xn, g_x, norm1_w, QWEN15_DIM, QWEN15_RMS_EPS);
        quantize_x_to_q8(g_xn, g_xn_q8, QWEN15_DIM);

        // Q, K, V em 1 unica regiao paralela (layout Q4x4)
        compute_qkv_q4x4(g_q, g_k, g_v, wq_w, wk_w, wv_w, bq_w, bk_w, bv_w, g_xn_q8);

        // RoPE Positional Embeddings
        apply_rope(g_q, pos, QWEN15_NUM_HEADS);
        apply_rope(g_k, pos, QWEN15_NUM_KV_HEADS);

        // Grava no KV-Cache
        memcpy(&g_k_cache[l][pos][0], g_k, (QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM) * sizeof(float));
        memcpy(&g_v_cache[l][pos][0], g_v, (QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM) * sizeof(float));

        // Grouped Query Attention (GQA) - Vetorizado em SIMD (128 floats = 8x unroll de 16)
        int h;
        for (h = 0; h < QWEN15_NUM_HEADS; h++) {
            int kv_h = h / 6;
            const float* q_h = g_q + h * QWEN15_HEAD_DIM;
            float* out_h = g_attn_out + h * QWEN15_HEAD_DIM;
            float* head_scores = g_scores[h];

            float max_score = -1e30f;
            int s;
            for (s = 0; s <= pos; s++) {
                const float* k_vec = &g_k_cache[l][s][kv_h * QWEN15_HEAD_DIM];
                __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
                __m128 a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
                int d;
                for (d = 0; d < QWEN15_HEAD_DIM; d += 16) {
                    a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(q_h + d),      _mm_loadu_ps(k_vec + d)));
                    a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(q_h + d + 4),  _mm_loadu_ps(k_vec + d + 4)));
                    a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_loadu_ps(q_h + d + 8),  _mm_loadu_ps(k_vec + d + 8)));
                    a3 = _mm_add_ps(a3, _mm_mul_ps(_mm_loadu_ps(q_h + d + 12), _mm_loadu_ps(k_vec + d + 12)));
                }
                __m128 sum128 = _mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                float dot = _mm_cvtss_f32(sum128) * 0.0883883476f; // 1 / sqrt(128)
                head_scores[s] = dot;
                if (dot > max_score) max_score = dot;
            }

            float sum_exp = 0.0f;
            for (s = 0; s <= pos; s++) {
                float exp_v = expf(head_scores[s] - max_score);
                head_scores[s] = exp_v;
                sum_exp += exp_v;
            }
            float inv_sum = 1.0f / sum_exp;

            memset(out_h, 0, QWEN15_HEAD_DIM * sizeof(float));
            for (s = 0; s <= pos; s++) {
                float p = head_scores[s] * inv_sum;
                __m128 pvec = _mm_set1_ps(p);
                const float* v_vec = &g_v_cache[l][s][kv_h * QWEN15_HEAD_DIM];
                int d;
                for (d = 0; d < QWEN15_HEAD_DIM; d += 16) {
                    _mm_storeu_ps(out_h + d,      _mm_add_ps(_mm_loadu_ps(out_h + d),      _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d))));
                    _mm_storeu_ps(out_h + d + 4,  _mm_add_ps(_mm_loadu_ps(out_h + d + 4),  _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d + 4))));
                    _mm_storeu_ps(out_h + d + 8,  _mm_add_ps(_mm_loadu_ps(out_h + d + 8),  _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d + 8))));
                    _mm_storeu_ps(out_h + d + 12, _mm_add_ps(_mm_loadu_ps(out_h + d + 12), _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d + 12))));
                }
            }
        }

        // Residual + Wo
        quantize_x_to_q8(g_attn_out, g_attn_q8, QWEN15_DIM);
        matvec_q4x4(g_x, wo_w, g_attn_q8, NULL, QWEN15_DIM, QWEN15_DIM, 1);
        g_prof_attn += (timer_now() - t_a0);

        double t_m0 = timer_now();
        // Post-Attention RMSNorm + Quantizacao Q8
        rmsnorm(g_xn, g_x, norm2_w, QWEN15_DIM, QWEN15_RMS_EPS);
        quantize_x_to_q8(g_xn, g_xn_q8, QWEN15_DIM);

        // MLP: Gate + Up + SiLU em 1 unica regiao paralela (layout Q4x4)
        compute_mlp_q4x4(g_h_mlp, wgate_w, wup_w, g_xn_q8);

        // Residual + Wdown
        quantize_x_to_q8(g_h_mlp, g_h_mlp_q8, QWEN15_INTERMEDIATE_DIM);
        matvec_q4x4(g_x, wdown_w, g_h_mlp_q8, NULL, QWEN15_DIM, QWEN15_INTERMEDIATE_DIM, 1);
        g_prof_mlp += (timer_now() - t_m0);
    }

    // Final RMSNorm
    const uint16_t* norm_w = (const uint16_t*)(model_bytes + g_qwen15_offsets.norm);
    rmsnorm(g_x_final, g_x, norm_w, QWEN15_DIM, QWEN15_RMS_EPS);
    quantize_x_to_q8(g_x_final, g_x_final_q8, QWEN15_DIM);

    // LM Head
    double t_l0 = timer_now();
    const BlockQ4x4* embed_w = (const BlockQ4x4*)(model_bytes + g_qwen15_offsets.embed_tokens);
    matvec_q4x4(g_logits, embed_w, g_x_final_q8, NULL, QWEN15_VOCAB_SIZE, QWEN15_DIM, 0);
    g_prof_lm += (timer_now() - t_l0);
    g_prof_tokens++;
}

// ----------------------------------------------------------------------------
// SAMPLING ALGORITHMS
// ----------------------------------------------------------------------------
// (Funcoes de amostragem em nativa_sampling.h)


// ----------------------------------------------------------------------------
// INTERFACE PRINCIPAL
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    timer_init();
    init_rope_freqs();

    const char* model_path = "data/qwen2.5_1.5b_q4x4.bin"; // layout intercalado (tools/repack_q4x4.py)
    const char* vocab_path = "data/qwen_vocab.bin";

    bool interactive_mode = (argc > 1 && strcmp(argv[1], "--interactive") == 0);

    if (!load_vocab(vocab_path)) {
        fprintf(stderr, "[ERRO] Nao foi possivel carregar o vocabulario em %s\n", vocab_path);
        return 1;
    }

    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ERRO] Nao foi possivel abrir %s (Codigo: %lu)\n", model_path, GetLastError());
        return 1;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "[ERRO] Falha ao criar mapeamento de arquivo\n");
        CloseHandle(hFile);
        return 1;
    }

    const uint8_t* model_bytes = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!model_bytes) {
        fprintf(stderr, "[ERRO] Falha ao mapear arquivo na memoria\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }

    if (interactive_mode) {
        printf("[NATIVA_CHAT_READY]\n");
        fflush(stdout);

        int cur_seq_pos = 0;
        char* line_buf = (char*)malloc(131072);

        while (fgets(line_buf, 131072, stdin)) {
            if (strncmp(line_buf, "QUIT", 4) == 0) break;
            if (strncmp(line_buf, "INPUT", 5) != 0) continue;

            char* ptr = line_buf + 5;
            int num_in = 0, max_gen = 60;
            if (sscanf(ptr, "%d %d", &num_in, &max_gen) < 1 || num_in <= 0) continue;
            if (max_gen <= 0) max_gen = 60;

            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            while (*ptr && (*ptr >= '0' && *ptr <= '9')) ptr++;
            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            while (*ptr && (*ptr >= '0' && *ptr <= '9')) ptr++;
            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;

            int t_idx;
            for (t_idx = 0; t_idx < num_in && cur_seq_pos < MAX_SEQ_LEN; t_idx++) {
                int tok_val = 0;
                if (sscanf(ptr, "%d", &tok_val) != 1) break;
                forward_token(model_bytes, tok_val, cur_seq_pos);
                cur_seq_pos++;
                while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
                while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            }

            int gen_step = 0;
            double t_sample_total = 0.0;
            double t_print_total = 0.0;
            double t_fwd_total = 0.0;
            g_prof_attn = g_prof_mlp = g_prof_lm = 0.0;
            g_prof_tokens = 0;

            double t_turn0 = timer_now();
            while (cur_seq_pos < MAX_SEQ_LEN - 1 && gen_step < max_gen) {
                double t_s0 = timer_now();
                int next_tok = sample_temperature(g_logits, QWEN15_VOCAB_SIZE, 0.7f, 0.9f);
                double t_s1 = timer_now();
                t_sample_total += (t_s1 - t_s0);

                if (next_tok == 151645 || next_tok == 151643) {
                    break;
                }

                double t_p0 = timer_now();
                print_token(next_tok);
                double t_p1 = timer_now();
                t_print_total += (t_p1 - t_p0);

                double t_f0 = timer_now();
                forward_token(model_bytes, next_tok, cur_seq_pos);
                double t_f1 = timer_now();
                t_fwd_total += (t_f1 - t_f0);

                cur_seq_pos++;
                gen_step++;
            }
            double t_turn1 = timer_now();
            double turn_sec = t_turn1 - t_turn0;
            double turn_tok_s = (turn_sec > 0.0) ? (gen_step / turn_sec) : 0.0;

            printf("\n[TURN_DONE tokens=%d time=%.2f speed=%.2f fwd=%.1fms attn=%.1fms mlp=%.1fms lm=%.1fms samp=%.1fms print=%.1fms]\n",
                   gen_step, turn_sec, turn_tok_s,
                   (gen_step > 0) ? (t_fwd_total / gen_step * 1000.0) : 0.0,
                   (g_prof_tokens > 0) ? (g_prof_attn / g_prof_tokens * 1000.0) : 0.0,
                   (g_prof_tokens > 0) ? (g_prof_mlp / g_prof_tokens * 1000.0) : 0.0,
                   (g_prof_tokens > 0) ? (g_prof_lm / g_prof_tokens * 1000.0) : 0.0,
                   (gen_step > 0) ? (t_sample_total / gen_step * 1000.0) : 0.0,
                   (gen_step > 0) ? (t_print_total / gen_step * 1000.0) : 0.0);
            fflush(stdout);
        }

        free(line_buf);
    } else {
        // Benchmark mode
        int prompt_tokens_buf[1024];
        int prompt_len = 14;
        int num_tokens_generate = 25;
        bool custom_prompt = false;

        int default_prompt[] = { 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 77091, 198 };
        memcpy(prompt_tokens_buf, default_prompt, sizeof(default_prompt));

        bool bench_mode = false;
        int a;
        for (a = 1; a < argc; a++) {
            if (strcmp(argv[a], "--bench") == 0 && a + 1 < argc) {
                bench_mode = true;
                num_tokens_generate = atoi(argv[a + 1]);
                if (num_tokens_generate <= 0) num_tokens_generate = 128;
                a++;
                continue;
            }
            if (strcmp(argv[a], "--tokens") == 0 && a + 1 < argc) {
                int n_toks = atoi(argv[a + 1]);
                if (n_toks > 0 && n_toks < 1024 && a + 1 + n_toks < argc) {
                    prompt_len = n_toks;
                    custom_prompt = true;
                    int t;
                    for (t = 0; t < n_toks; t++) {
                        prompt_tokens_buf[t] = atoi(argv[a + 2 + t]);
                    }
                    a += 1 + n_toks;
                    if (a + 1 < argc) {
                        num_tokens_generate = atoi(argv[a + 1]);
                    }
                }
            }
        }

        printf("====================================================================\n");
        printf("  NATIVA C99 ENGINE BENCHMARK - QWEN2.5-1.5B (INT4 Q4_0 x Q8_0)\n");
        printf("  Parametros: 1.54 Bilhoes | Camadas: 28 | Memoria: 828 MB\n");
        printf("====================================================================\n\n");

        if (custom_prompt) {
            printf("[*] Prompt customizado (%d tokens):\n\"", prompt_len);
            int p_i;
            for (p_i = 0; p_i < prompt_len; p_i++) {
                print_token(prompt_tokens_buf[p_i]);
            }
            printf("\"\n\n");
        }

        printf("[*] Executando prefill do prompt (%d tokens)...\n", prompt_len);
        double t_prefill_start = timer_now();
        int p;
        for (p = 0; p < prompt_len - 1; p++) {
            forward_token(model_bytes, prompt_tokens_buf[p], p);
        }
        double t_prefill_end = timer_now();
        printf("    -> Prefill concluido em %.3f s!\n\n", t_prefill_end - t_prefill_start);

        printf("[*] Gerando tokens em tempo real:\n----------------------------------------------------\n");
        int cur_token = prompt_tokens_buf[prompt_len - 1];
        int pos = prompt_len - 1;
        int generated = 0;

        // Zera o perfil para medir apenas a geracao (sem o prefill)
        g_prof_attn = g_prof_mlp = g_prof_lm = 0.0;
        g_prof_tokens = 0;
        uint32_t seq_hash = 2166136261u; // FNV-1a da sequencia gerada

        double t_gen_start = timer_now();
        while (generated < num_tokens_generate && pos < MAX_SEQ_LEN - 1) {
            forward_token(model_bytes, cur_token, pos);
            int next_token = bench_mode ? sample_argmax(g_logits, QWEN15_VOCAB_SIZE)
                                        : sample_temperature(g_logits, QWEN15_VOCAB_SIZE, 0.7f, 0.9f);

            if (!bench_mode && (next_token == 151645 || next_token == 151643)) {
                break;
            }

            seq_hash = (seq_hash ^ (uint32_t)next_token) * 16777619u;
            if (!bench_mode) print_token(next_token);
            cur_token = next_token;
            pos++;
            generated++;
        }
        double t_gen_end = timer_now();
        double gen_sec = t_gen_end - t_gen_start;
        double tok_s = (gen_sec > 0.0) ? (generated / gen_sec) : 0.0;

        printf("\n----------------------------------------------------\n");
        printf("[RESULTADO FINAL]:\n");
        printf("  Tokens gerados   : %d\n", generated);
        printf("  Tempo total      : %.3f s\n", gen_sec);
        printf("  Velocidade       : \x1b[1;32m%.2f tokens/segundo\x1b[0m\n", tok_s);
        if (g_prof_tokens > 0) {
            printf("  [Perfil por Token]:\n");
            printf("    Atenção (28 cam) : %.1f ms\n", (g_prof_attn / g_prof_tokens) * 1000.0);
            printf("    MLP (28 cam)     : %.1f ms\n", (g_prof_mlp / g_prof_tokens) * 1000.0);
            printf("    LM Head (Saída)  : %.1f ms\n", (g_prof_lm / g_prof_tokens) * 1000.0);
        }
        if (bench_mode) printf("  BENCH_HASH=%08x\n", seq_hash);
        printf("====================================================================\n");
    }

    UnmapViewOfFile(model_bytes);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    return 0;
}
