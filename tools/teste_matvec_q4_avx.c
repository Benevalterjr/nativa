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

// AVX 256-bit horizontal sum
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehdup_ps(vsum);
    __m128 sums  = _mm_add_ps(vsum, shuf);
    shuf         = _mm_movehl_ps(shuf, sums);
    sums         = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matvec_q4_avx(float* restrict y, const BlockQ4* restrict W, const float* restrict x, const uint16_t* restrict bias, int rows, int cols) {
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

            // 8 floats for elements 0..7
            __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(lo));
            __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 4)));
            __m256 q0 = _mm256_set_m128(f1, f0);

            // 8 floats for elements 8..15
            __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 12)));
            __m256 q1 = _mm256_set_m128(f3, f2);

            // 8 floats for elements 16..23
            __m128 f4 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(hi));
            __m128 f5 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
            __m256 q2 = _mm256_set_m128(f5, f4);

            // 8 floats for elements 24..31
            __m128 f6 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
            __m128 f7 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 12)));
            __m256 q3 = _mm256_set_m128(f7, f6);

            __m256 acc0 = _mm256_mul_ps(q0, _mm256_loadu_ps(xb));
            __m256 acc1 = _mm256_mul_ps(q1, _mm256_loadu_ps(xb + 8));
            __m256 acc2 = _mm256_mul_ps(q2, _mm256_loadu_ps(xb + 16));
            __m256 acc3 = _mm256_mul_ps(q3, _mm256_loadu_ps(xb + 24));

            __m256 blk_acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            sum += hsum256_ps(blk_acc) * scale;
        }

        if (bias) {
            sum += bf16_to_fp32(bias[i]);
        }
        y[i] = sum;
    }
}

int main(void) {
    printf("Testando matvec_q4_avx...\n");
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
            W[i].qs[k] = 0x88;
        }
    }
    W[0].qs[0] = 0x6A;

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    matvec_q4_avx(y, W, x, NULL, rows, cols);
    printf("Resultado y[0]: %f\n", y[0]);

    QueryPerformanceCounter(&t0);
    int iters = 200;
    for (int it = 0; it < iters; it++) {
        matvec_q4_avx(y, W, x, NULL, rows, cols);
    }
    QueryPerformanceCounter(&t1);

    double dt = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double per_iter_ms = (dt / iters) * 1000.0;
    printf("Tempo medio por matvec AVX (1536x1536): %.3f ms\n", per_iter_ms);

    free(W);
    free(x);
    free(y);
    return 0;
}
