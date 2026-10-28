#ifndef QUTRIT_KERNELS_H
#define QUTRIT_KERNELS_H

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define popcnt64(x) ((int64_t)__popcnt64((uint64_t)(x)))
#else
#define popcnt64(x) ((int64_t)__builtin_popcountll((unsigned long long)(x)))
#endif

#ifdef _WIN32
#define QK_API __declspec(dllexport)
#else
#define QK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* 1. INT4 Data Structures and Kernels (Group-wise 32 or 128)                */
/* ------------------------------------------------------------------------- */

typedef struct {
    float scale;        /* float32 scale per group of 32 */
    uint8_t qs[16];     /* 16 bytes = 32 x 4-bit nibbles */
} BlockQ4_32;

QK_API void nat_quantize_int4_g32(const float* W, BlockQ4_32* out_blocks, int64_t M, int64_t K);
QK_API void nat_gemv_int4_g32(const BlockQ4_32* W, const float* x, float* y, int64_t M, int64_t K);

/* ------------------------------------------------------------------------- */
/* 2. Ternary and 5-trits Reference Kernels (BitNet b1.58)                   */
/* ------------------------------------------------------------------------- */

QK_API void nat_gemv_ternary_int8(const int8_t* W, const float* x, float* y, int64_t M, int64_t K, float gamma);
QK_API void nat_pack_5trits_2d(const int8_t* in, uint8_t* out_bytes, int64_t M, int64_t K);
QK_API void nat_gemv_veclut_5trits(const uint8_t* W_packed, const float* x, float* y, int64_t M, int64_t K, float gamma);

/* ------------------------------------------------------------------------- */
/* 3. Qutrit 2-bit dot64 Kernels (4 states: 00, 01, 10, 11)                  */
/* ------------------------------------------------------------------------- */

/* Packing functions */
QK_API void nat_pack_qutrit_baseline(const float* W, uint64_t* W_pos, uint64_t* W_neg, int64_t M, int64_t K, float gamma);
QK_API void nat_pack_qutrit_subquantum(const float* W, uint64_t* W_pos, uint64_t* W_neg, int64_t M, int64_t K, float gamma, float low_ratio);
QK_API void nat_pack_qutrit_prob(const float* W, uint64_t* W_pos, uint64_t* W_neg, int64_t M, int64_t K, float gamma, float low_th, float high_th);

/* Activation quantizer (7 magnitude planes + 1 sign plane for popcount GEMV) */
QK_API void nat_quantize_activations_popcnt(const float* x, uint64_t* x_sign, uint64_t* x_mag, int64_t K, float* act_scale);

/* GEMV with Bit-plane Popcount (Multiplication-free) */
QK_API void nat_qutrit_dot64_baseline_gemv(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const uint64_t* x_sign,
    const uint64_t* x_mag,
    float* y,
    int64_t M,
    int64_t num_blocks,
    float gamma,
    float act_scale
);

QK_API void nat_qutrit_dot64_subquantum_gemv(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const uint64_t* x_sign,
    const uint64_t* x_mag,
    float* y,
    int64_t M,
    int64_t num_blocks,
    float gamma,
    float beta,
    float act_scale
);

QK_API void nat_qutrit_dot64_ctx_gemv(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const uint64_t* x_sign,
    const uint64_t* x_mag,
    float* y,
    int64_t M,
    int64_t num_blocks,
    float gamma,
    float act_scale,
    uint64_t c_mask
);

/* Direct Float32 Activation GEMV Kernels (Continuous x) */
QK_API void nat_qutrit_dot64_subquantum_gemv_f32(
    const uint64_t* W_pos,
    const uint64_t* W_neg,
    const float* x,
    float* y,
    int64_t M,
    int64_t K,
    float gamma,
    float beta
);

/* ------------------------------------------------------------------------- */
/* 4. veclut_4qutrits (LUT256 for 4 qutrits per byte = 2.0 bits/param)       */
/* ------------------------------------------------------------------------- */

QK_API void nat_pack_4qutrits(const float* W, uint8_t* out_bytes, int64_t M, int64_t K, float gamma, float low_ratio);
QK_API void nat_gemv_veclut_4qutrits(
    const uint8_t* W_packed,
    const float* x,
    float* y,
    int64_t M,
    int64_t K,
    float gamma,
    float beta
);

#ifdef __cplusplus
}
#endif

#endif /* QUTRIT_KERNELS_H */
