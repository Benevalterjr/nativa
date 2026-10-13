#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t scale;
    uint8_t qs[16];
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

static inline void quantize_block_32(const float* in_vals, BlockQ4* out_block) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float af = fabsf(in_vals[i]);
        if (af > amax) amax = af;
    }
    if (amax < 1e-12f) {
        out_block->scale = 0;
        memset(out_block->qs, 0x88, 16);
        return;
    }
    float scale = amax / 7.0f;
    uint16_t scale_bf16 = fp32_to_bf16(scale);
    float recon_scale = bf16_to_fp32(scale_bf16);
    if (recon_scale < 1e-12f) recon_scale = scale;
    float inv_scale = 1.0f / recon_scale;

    out_block->scale = scale_bf16;
    for (int k = 0; k < 16; k++) {
        int q0 = (int)roundf(in_vals[k] * inv_scale) + 8;
        int q1 = (int)roundf(in_vals[k + 16] * inv_scale) + 8;
        if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
        if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
        out_block->qs[k] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

int main(void) {
    float orig[32];
    for (int i = 0; i < 32; i++) {
        orig[i] = (float)(i - 16) * 0.15f;
    }

    BlockQ4 blk;
    quantize_block_32(orig, &blk);

    // Dequantize SIMD
    __m128i raw = _mm_loadu_si128((const __m128i*)blk.qs);
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i off = _mm_set1_epi8(8);
    __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, mask), off);
    __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), off);

    float dequant[32];
    float scale = bf16_to_fp32(blk.scale);

    __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(lo));
    __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 4)));
    __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 12)));

    _mm_storeu_ps(&dequant[0], _mm_mul_ps(f0, _mm_set1_ps(scale)));
    _mm_storeu_ps(&dequant[4], _mm_mul_ps(f1, _mm_set1_ps(scale)));
    _mm_storeu_ps(&dequant[8], _mm_mul_ps(f2, _mm_set1_ps(scale)));
    _mm_storeu_ps(&dequant[12], _mm_mul_ps(f3, _mm_set1_ps(scale)));

    __m128 f4 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(hi));
    __m128 f5 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
    __m128 f6 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
    __m128 f7 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 12)));

    _mm_storeu_ps(&dequant[16], _mm_mul_ps(f4, _mm_set1_ps(scale)));
    _mm_storeu_ps(&dequant[20], _mm_mul_ps(f5, _mm_set1_ps(scale)));
    _mm_storeu_ps(&dequant[24], _mm_mul_ps(f6, _mm_set1_ps(scale)));
    _mm_storeu_ps(&dequant[28], _mm_mul_ps(f7, _mm_set1_ps(scale)));

    float max_err = 0.0f;
    for (int i = 0; i < 32; i++) {
        float err = fabsf(orig[i] - dequant[i]);
        if (err > max_err) max_err = err;
        printf("i=%2d orig=%7.4f dequant=%7.4f err=%7.4f\n", i, orig[i], dequant[i], err);
    }
    printf("Erro maximo de quantizacao INT4: %f\n", max_err);
    return 0;
}
