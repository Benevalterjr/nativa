#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <windows.h>
#include <omp.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t scale; // BF16 scale
    uint8_t qs[16]; // 16 bytes for 32 nibbles
} BlockQ4;
#pragma pack(pop)

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

static inline float hsum_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matvec_q4(float* restrict y, const BlockQ4* restrict W, const float* restrict x, const uint16_t* restrict bias, int rows, int cols) {
    int num_blocks = cols / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i off = _mm_set1_epi8(8);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i++) {
        const BlockQ4* row_blocks = W + (size_t)i * num_blocks;
        float sum = 0.0f;

        for (int b = 0; b < num_blocks; b++) {
            const BlockQ4* blk = &row_blocks[b];
            float scale = bf16_to_fp32(blk->scale);
            const float* xb = x + b * 32;

            __m128i raw = _mm_loadu_si128((const __m128i*)blk->qs);
            __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, mask), off);
            __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), off);

            __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(lo));
            __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 4)));
            __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 12)));

            __m128 acc0 = _mm_mul_ps(f0, _mm_loadu_ps(xb));
            __m128 acc1 = _mm_mul_ps(f1, _mm_loadu_ps(xb + 4));
            __m128 acc2 = _mm_mul_ps(f2, _mm_loadu_ps(xb + 8));
            __m128 acc3 = _mm_mul_ps(f3, _mm_loadu_ps(xb + 12));

            __m128 f4 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(hi));
            __m128 f5 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
            __m128 f6 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
            __m128 f7 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 12)));

            acc0 = _mm_add_ps(acc0, _mm_mul_ps(f4, _mm_loadu_ps(xb + 16)));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(f5, _mm_loadu_ps(xb + 20)));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(f6, _mm_loadu_ps(xb + 24)));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(f7, _mm_loadu_ps(xb + 28)));

            __m128 blk_acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
            sum += hsum_ps(blk_acc) * scale;
        }

        if (bias) {
            sum += bf16_to_fp32(bias[i]);
        }
        y[i] = sum;
    }
}

int main(void) {
    printf("Testando matvec_q4...\n");
    int rows = 1536;
    int cols = 1536;
    int num_blocks = cols / 32;

    BlockQ4* W = (BlockQ4*)malloc(rows * num_blocks * sizeof(BlockQ4));
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    for (int i = 0; i < rows * num_blocks; i++) {
        W[i].scale = fp32_to_bf16(0.1f);
        for (int k = 0; k < 16; k++) {
            W[i].qs[k] = 0x88; // 8 - 8 = 0
        }
    }
    // Place some non-zero weights in the first block of the first row
    // byte 0: lo=10 (10-8 = 2), hi=6 (6-8 = -2)
    W[0].qs[0] = 0x6A; // lo=10, hi=6

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    // Warmup
    matvec_q4(y, W, x, NULL, rows, cols);
    printf("Result y[0] (expected: 0.0f): %f\n", y[0]); // (2 + -2) * 0.1 = 0.0

    // Benchmark 100 iteracoes
    QueryPerformanceCounter(&t0);
    int iters = 200;
    for (int it = 0; it < iters; it++) {
        matvec_q4(y, W, x, NULL, rows, cols);
    }
    QueryPerformanceCounter(&t1);

    double dt = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double per_iter_ms = (dt / iters) * 1000.0;
    printf("Average time per matvec (1536x1536): %.3f ms\n", per_iter_ms);

    free(W);
    free(x);
    free(y);
    return 0;
}
