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
    uint16_t scale; // BF16 scale
    uint8_t qs[16]; // 16 bytes for 32 nibbles
} BlockQ4;
#pragma pack(pop)

typedef struct {
    float scale;
    int8_t qs[32];
    int32_t sum_lo; // 8 * sum(qs[0..15])
    int32_t sum_hi; // 8 * sum(qs[16..31])
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

// Quantiza vetor x de floats para BlockQ8
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
            x_q8[b].sum_lo = 0;
            x_q8[b].sum_hi = 0;
            continue;
        }

        float scale = amax / 127.0f;
        float inv_scale = 1.0f / scale;
        x_q8[b].scale = scale;

        int32_t slo = 0;
        for (int i = 0; i < 16; i++) {
            int q = (int)roundf(xb[i] * inv_scale);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            x_q8[b].qs[i] = (int8_t)q;
            slo += q;
        }
        x_q8[b].sum_lo = slo * 8;

        int32_t shi = 0;
        for (int i = 16; i < 32; i++) {
            int q = (int)roundf(xb[i] * inv_scale);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            x_q8[b].qs[i] = (int8_t)q;
            shi += q;
        }
        x_q8[b].sum_hi = shi * 8;
    }
}

// Dot product Q4_0 x Q8_0 usando SSSE3 _mm_maddubs_epi16
void matvec_q4_q8(float* restrict y, const BlockQ4* restrict W, const BlockQ8* restrict x_q8, const uint16_t* restrict bias, int rows, int cols) {
    int num_blocks = cols / 32;
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i++) {
        const BlockQ4* row_blocks = W + (size_t)i * num_blocks;
        float sum = 0.0f;

        for (int b = 0; b < num_blocks; b++) {
            const BlockQ4* blk = &row_blocks[b];
            const BlockQ8* xblk = &x_q8[b];
            float scale = bf16_to_fp32(blk->scale) * xblk->scale;

            __m128i raw = _mm_loadu_si128((const __m128i*)blk->qs);
            __m128i lo = _mm_and_si128(raw, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);

            __m128i x_lo = _mm_loadu_si128((const __m128i*)xblk->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xblk->qs + 16));

            // _mm_maddubs_epi16: multiplica unsigned 8-bit por signed 8-bit e soma pares adjacentes em int16!
            __m128i p_lo = _mm_maddubs_epi16(lo, x_lo);
            __m128i p_hi = _mm_maddubs_epi16(hi, x_hi);

            // _mm_madd_epi16: soma pares int16 em int32!
            __m128i s_lo = _mm_madd_epi16(p_lo, ones);
            __m128i s_hi = _mm_madd_epi16(p_hi, ones);

            __m128i total_s = _mm_add_epi32(s_lo, s_hi);

            // Reduz 4x int32 em 1 int32 com SSSE3 phaddd
            __m128i h0 = _mm_hadd_epi32(total_s, total_s);
            __m128i h1 = _mm_hadd_epi32(h0, h0);
            int32_t dot_int = _mm_cvtsi128_si32(h1);

            int32_t dot_adjusted = dot_int - (xblk->sum_lo + xblk->sum_hi);
            sum += (float)dot_adjusted * scale;
        }

        if (bias) {
            sum += bf16_to_fp32(bias[i]);
        }
        y[i] = sum;
    }
}

int main(void) {
    printf("Testando matvec_q4_q8 (SSSE3 maddubs integer kernel)...\n");
    int rows = 1536;
    int cols = 1536;
    int num_blocks = cols / 32;

    BlockQ4* W = (BlockQ4*)malloc(rows * num_blocks * sizeof(BlockQ4));
    BlockQ8* x_q8 = (BlockQ8*)malloc(num_blocks * sizeof(BlockQ8));
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    for (int i = 0; i < rows * num_blocks; i++) {
        W[i].scale = fp32_to_bf16(0.1f);
        for (int k = 0; k < 16; k++) {
            W[i].qs[k] = 0x88; // 8 - 8 = 0
        }
    }
    W[0].qs[0] = 0x6A; // lo=10, hi=6

    quantize_x_to_q8(x, x_q8, cols);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    matvec_q4_q8(y, W, x_q8, NULL, rows, cols);
    printf("Result y[0] (expected: 0.0f): %f\n", y[0]);

    QueryPerformanceCounter(&t0);
    int iters = 500;
    for (int it = 0; it < iters; it++) {
        matvec_q4_q8(y, W, x_q8, NULL, rows, cols);
    }
    QueryPerformanceCounter(&t1);

    double dt = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double per_iter_ms = (dt / iters) * 1000.0;
    printf("Average time per Q4_Q8 matvec (1536x1536): %.3f ms\n", per_iter_ms);

    free(W);
    free(x_q8);
    free(x);
    free(y);
    return 0;
}
