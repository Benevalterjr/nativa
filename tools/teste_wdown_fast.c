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

// Vectorized 4-row kernel using 3-instruction reduction
void matvec_q4_q8_fast4(float* restrict y, const BlockQ4* restrict W, const BlockQ8* restrict x_q8, int rows, int cols) {
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

            for (int b = 0; b < num_blocks; b++) {
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

                // 3 instructions reduce all 4 rows simultaneously
                __m128i h01 = _mm_hadd_epi32(tot0, tot1);
                __m128i h23 = _mm_hadd_epi32(tot2, tot3);
                __m128i h_all = _mm_hadd_epi32(h01, h23);

                __m128i adj = _mm_set1_epi32(xblk->sum_adj);
                __m128 fdots = _mm_cvtepi32_ps(_mm_sub_epi32(h_all, adj));

                float x_sc = xblk->scale;
                __m128 scales = _mm_set_ps(bf16_to_fp32(blk3->scale) * x_sc,
                                           bf16_to_fp32(blk2->scale) * x_sc,
                                           bf16_to_fp32(blk1->scale) * x_sc,
                                           bf16_to_fp32(blk0->scale) * x_sc);

                acc_rows = _mm_add_ps(acc_rows, _mm_mul_ps(fdots, scales));
            }
            _mm_storeu_ps(&y[i], acc_rows);
        } else {
            for (int r = i; r < rows; r++) {
                const BlockQ4* row_r = W + (size_t)r * num_blocks;
                float sum_r = 0.0f;
                for (int b = 0; b < num_blocks; b++) {
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
                y[r] = sum_r;
            }
        }
    }
}

int main(void) {
    // Benchmark wdown: 1536 rows x 8960 cols
    int rows = 1536;
    int cols = 8960;
    int num_blocks = cols / 32;

    BlockQ4* W = (BlockQ4*)malloc(rows * num_blocks * sizeof(BlockQ4));
    BlockQ8* x_q8 = (BlockQ8*)malloc(num_blocks * sizeof(BlockQ8));
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    for (int i = 0; i < rows * num_blocks; i++) {
        W[i].scale = fp32_to_bf16(0.01f);
        for (int k = 0; k < 16; k++) {
            W[i].qs[k] = 0x88;
        }
    }
    quantize_x_to_q8(x, x_q8, cols);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    matvec_q4_q8_fast4(y, W, x_q8, rows, cols);

    QueryPerformanceCounter(&t0);
    int iters = 200;
    for (int it = 0; it < iters; it++) {
        matvec_q4_q8_fast4(y, W, x_q8, rows, cols);
    }
    QueryPerformanceCounter(&t1);

    double dt = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double per_iter_ms = (dt / iters) * 1000.0;
    printf("Tempo medio wdown (1536x8960): %.3f ms\n", per_iter_ms);

    free(W);
    free(x_q8);
    free(x);
    free(y);
    return 0;
}
