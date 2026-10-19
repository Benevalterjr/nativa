/* ========================================================================= */
/* Unified C99 Runtime for the Nativa Language (nativa_rt.h)                 */
/* Shared between AOTKernelEngine (ctypes DLL) and CTranspiler (nativa)      */
/* ========================================================================= */
#ifndef NATIVA_RT_H
#define NATIVA_RT_H

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

#ifndef NAT_API
#ifdef NATIVA_BUILD_DLL
#ifdef _WIN32
#define NAT_API __declspec(dllexport)
#else
#define NAT_API __attribute__((visibility("default")))
#endif
#else
#define NAT_API static inline
#endif
#endif

#ifndef NAT_PI
#define NAT_PI 3.14159265358979323846
#endif
#ifndef NAT_E
#define NAT_E  2.71828182845904523536
#endif

/* ------------------------------------------------------------------------- */
/* 1. Float32 / BitNet b1.58 AOT Kernels (used via FFI and C99)              */
/* ------------------------------------------------------------------------- */

NAT_API void nat_c_gemv_ternary(const int8_t* W, const float* x, float* y, int64_t M, int64_t K) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc = 0.0f;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc;
    }
}

static const uint8_t DEC_01[256] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3
};

static const uint8_t DEC_23[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1
};

static const uint8_t DEC_4[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

NAT_API void nat_c_gemv_veclut_5trits(const uint8_t* W_packed, const float* x, float* y, int64_t M, int64_t K, float gamma) {
    int64_t num_groups = (K + 4) / 5;
    float* lut_full = (float*)malloc(num_groups * 256 * sizeof(float));
    if (!lut_full) return;

    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 5;
        float x0 = (base < K) ? x[base] : 0.0f;
        float x1 = (base + 1 < K) ? x[base + 1] : 0.0f;
        float x2 = (base + 2 < K) ? x[base + 2] : 0.0f;
        float x3 = (base + 3 < K) ? x[base + 3] : 0.0f;
        float x4 = (base + 4 < K) ? x[base + 4] : 0.0f;

        float sub01[9];
        float sub23[9];
        float sub4[3];
        for (int t1 = 0; t1 < 3; t1++) {
            for (int t0 = 0; t0 < 3; t0++) {
                sub01[t0 + 3 * t1] = (float)(t0 - 1) * x0 + (float)(t1 - 1) * x1;
                sub23[t0 + 3 * t1] = (float)(t0 - 1) * x2 + (float)(t1 - 1) * x3;
            }
        }
        for (int t4 = 0; t4 < 3; t4++) {
            sub4[t4] = (float)(t4 - 1) * x4;
        }

        float* g_lut = lut_full + g * 256;
        for (int b = 0; b < 243; b++) {
            g_lut[b] = sub01[DEC_01[b]] + sub23[DEC_23[b]] + sub4[DEC_4[b]];
        }
        for (int b = 243; b < 256; b++) {
            g_lut[b] = 0.0f;
        }
    }

    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const uint8_t* row = W_packed + i * num_groups;
        const float* lut_ptr = lut_full;
        int64_t g = 0;

        for (; g + 3 < num_groups; g += 4) {
            acc0 += lut_ptr[row[g]];
            acc1 += (lut_ptr + 256)[row[g + 1]];
            acc2 += (lut_ptr + 512)[row[g + 2]];
            acc3 += (lut_ptr + 768)[row[g + 3]];
            lut_ptr += 1024;
        }
        for (; g < num_groups; g++) {
            acc0 += lut_ptr[row[g]];
            lut_ptr += 256;
        }
        y[i] = (acc0 + acc1 + acc2 + acc3) * gamma;
    }

    free(lut_full);
}

NAT_API void nat_c_pack_5trits_2d(const int8_t* in, uint8_t* out_bytes, int64_t M, int64_t K) {
    int64_t bpr = (K + 4) / 5;
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const int8_t* row_in = in + i * K;
        uint8_t* row_out = out_bytes + i * bpr;
        for (int64_t g = 0; g < bpr; g++) {
            int64_t base = g * 5;
            uint16_t b_val = 0;
            uint16_t mul = 1;
            for (int j = 0; j < 5; j++) {
                int8_t t = (base + j < K) ? (row_in[base + j] + 1) : 1;
                b_val += (uint16_t)(t * mul);
                mul *= 3;
            }
            row_out[g] = (uint8_t)b_val;
        }
    }
}

NAT_API void nat_c_rmsnorm(const float* x, const float* gamma, float* out, int64_t N, float eps) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)N + eps);
    float inv_rms = 1.0f / rms;
    for (int64_t i = 0; i < N; i++) out[i] = (x[i] * inv_rms) * gamma[i];
}

NAT_API void nat_c_silu(const float* x, float* out, int64_t N) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < N; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

NAT_API void nat_c_softmax(const float* x, float* out, int64_t N) {
    float max_val = x[0];
    for (int64_t i = 1; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }
    float inv = 1.0f / sum;
    for (int64_t i = 0; i < N; i++) out[i] *= inv;
}

/* ------------------------------------------------------------------------- */
/* 2. Complex Numbers and O(N log N) Cooley-Tukey Fast Fourier Transform     */
/* ------------------------------------------------------------------------- */

typedef struct {
    double real;
    double imag;
} NatComplex;

static inline NatComplex nat_complex(double re, double im) {
    NatComplex c; c.real = re; c.imag = im; return c;
}
static inline NatComplex nat_complex_add(NatComplex a, NatComplex b) {
    return nat_complex(a.real + b.real, a.imag + b.imag);
}
static inline NatComplex nat_complex_sub(NatComplex a, NatComplex b) {
    return nat_complex(a.real - b.real, a.imag - b.imag);
}
static inline NatComplex nat_complex_mul(NatComplex a, NatComplex b) {
    return nat_complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}
static inline double nat_complex_abs(NatComplex c) {
    return sqrt(c.real * c.real + c.imag * c.imag);
}
static inline double nat_complex_power(NatComplex c) {
    return c.real * c.real + c.imag * c.imag;
}

/* In-place O(N log N) Radix-2 Cooley-Tukey algorithm for powers of 2 */
static inline void nat_fft_cooley_tukey(NatComplex* a, int64_t N, bool inverse) {
    for (int64_t i = 1, j = 0; i < N; i++) {
        int64_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            NatComplex tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    for (int64_t len = 2; len <= N; len <<= 1) {
        double ang = 2.0 * NAT_PI / (double)len * (inverse ? 1.0 : -1.0);
        int64_t half = len >> 1;
        for (int64_t i = 0; i < N; i += len) {
            for (int64_t j = 0; j < half; j++) {
                double theta = ang * (double)j;
                NatComplex w = nat_complex(cos(theta), sin(theta));
                NatComplex u = a[i + j];
                NatComplex v = nat_complex_mul(a[i + j + half], w);
                a[i + j] = nat_complex_add(u, v);
                a[i + j + half] = nat_complex_sub(u, v);
            }
        }
    }
    if (inverse) {
        double inv_n = 1.0 / (double)N;
        for (int64_t i = 0; i < N; i++) {
            a[i].real *= inv_n;
            a[i].imag *= inv_n;
        }
    }
}

static inline void nat_rfft(const double* x, double* out_spec, int64_t N) {
    if (N <= 0) return;
    int64_t M = N / 2 + 1;
    if ((N & (N - 1)) == 0) {
        NatComplex* buf = (NatComplex*)malloc((size_t)N * sizeof(NatComplex));
        if (buf) {
            for (int64_t i = 0; i < N; i++) buf[i] = nat_complex(x[i], 0.0);
            nat_fft_cooley_tukey(buf, N, false);
            for (int64_t k = 0; k < M; k++) {
                out_spec[k * 2]     = buf[k].real;
                out_spec[k * 2 + 1] = buf[k].imag;
            }
            free(buf);
            return;
        }
    }
    for (int64_t k = 0; k < M; k++) {
        double re = 0.0, im = 0.0;
        for (int64_t n = 0; n < N; n++) {
            double angle = 2.0 * NAT_PI * (double)k * (double)n / (double)N;
            re += x[n] * cos(angle);
            im -= x[n] * sin(angle);
        }
        out_spec[k * 2]     = re;
        out_spec[k * 2 + 1] = im;
    }
}

static inline void nat_irfft(const double* spec, double* out_x, int64_t M, int64_t N) {
    if (N <= 0 || M <= 0) return;
    if ((N & (N - 1)) == 0) {
        NatComplex* buf = (NatComplex*)calloc((size_t)N, sizeof(NatComplex));
        if (buf) {
            for (int64_t k = 0; k < M && k < N; k++) {
                buf[k] = nat_complex(spec[k * 2], spec[k * 2 + 1]);
            }
            int64_t max_k = (N - 1) / 2;
            for (int64_t k = 1; k <= max_k && k < M; k++) {
                buf[N - k] = nat_complex(spec[k * 2], -spec[k * 2 + 1]);
            }
            nat_fft_cooley_tukey(buf, N, true);
            for (int64_t n = 0; n < N; n++) out_x[n] = buf[n].real;
            free(buf);
            return;
        }
    }
    double inv_N = 1.0 / (double)N;
    int64_t max_k = (N - 1) / 2;
    for (int64_t n = 0; n < N; n++) {
        double val = spec[0];
        if (N % 2 == 0 && (N / 2) < M) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            val += spec[(N / 2) * 2] * sign;
        }
        for (int64_t k = 1; k <= max_k && k < M; k++) {
            double angle = 2.0 * NAT_PI * (double)k * (double)n / (double)N;
            double re_k = spec[k * 2];
            double im_k = spec[k * 2 + 1];
            val += 2.0 * (re_k * cos(angle) - im_k * sin(angle));
        }
        out_x[n] = val * inv_N;
    }
}

static inline void nat_power_spectrum_spec(const double* spec, double* out_ps, int64_t M) {
    for (int64_t k = 0; k < M; k++) {
        double re = spec[k * 2];
        double im = spec[k * 2 + 1];
        out_ps[k] = re * re + im * im;
    }
}

static inline void nat_power_spectrum_signal(const double* x, double* out_ps, int64_t N) {
    if (N <= 0) return;
    int64_t M = N / 2 + 1;
    double* spec = (double*)malloc((size_t)(M * 2) * sizeof(double));
    if (spec) {
        nat_rfft(x, spec, N);
        nat_power_spectrum_spec(spec, out_ps, M);
        free(spec);
        return;
    }
    for (int64_t k = 0; k < M; k++) {
        double re = 0.0, im = 0.0;
        for (int64_t n = 0; n < N; n++) {
            double angle = 2.0 * NAT_PI * (double)k * (double)n / (double)N;
            re += x[n] * cos(angle);
            im -= x[n] * sin(angle);
        }
        out_ps[k] = re * re + im * im;
    }
}

/* ------------------------------------------------------------------------- */
/* 3. Ternary Quantization, BitNet b1.58 and F64 Math Primitives             */
/* ------------------------------------------------------------------------- */

static inline int8_t* nat_quantize_ternary(const double* v, int64_t length, double threshold) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        if (v[i] > threshold) out[i] = 1;
        else if (v[i] < -threshold) out[i] = -1;
        else out[i] = 0;
    }
    return out;
}
static inline uint8_t* nat_pack_ternary(const int8_t* in, int64_t length) {
    int64_t num_bytes = (length + 3) / 4;
    uint8_t* out = (uint8_t*)calloc(num_bytes, sizeof(uint8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        uint8_t enc = (in[i] == 1) ? 1 : ((in[i] == -1) ? 2 : 0);
        int64_t b = i / 4;
        int shift = (i % 4) * 2;
        out[b] |= (enc << shift);
    }
    return out;
}
static inline int8_t* nat_unpack_ternary(const uint8_t* in, int64_t length) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    static const int8_t map[4] = {0, 1, -1, 0};
    for (int64_t i = 0; i < length; i++) {
        int64_t b = i / 4;
        int shift = (i % 4) * 2;
        uint8_t code = (in[b] >> shift) & 0x03;
        out[i] = map[code];
    }
    return out;
}

static inline double nat_ln(double x) { return log(x); }
static inline double nat_relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double nat_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double nat_silu(double x) { return x / (1.0 + exp(-x)); }

static inline void nat_rmsnorm_f64(const double* x, const double* gamma, double* out, int64_t N, double eps) {
    if (N <= 0) return;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
    double inv_rms = 1.0 / sqrt(sum_sq / (double)N + eps);
    for (int64_t i = 0; i < N; i++) out[i] = (x[i] * inv_rms) * (gamma ? gamma[i] : 1.0);
}

static inline void nat_silu_vec_f64(const double* x, double* out, int64_t N) {
    for (int64_t i = 0; i < N; i++) out[i] = x[i] / (1.0 + exp(-x[i]));
}

static inline void nat_relu_vec_f64(const double* x, double* out, int64_t N) {
    for (int64_t i = 0; i < N; i++) out[i] = x[i] > 0.0 ? x[i] : 0.0;
}

static inline void nat_sigmoid_vec_f64(const double* x, double* out, int64_t N) {
    for (int64_t i = 0; i < N; i++) out[i] = 1.0 / (1.0 + exp(-x[i]));
}

static inline void nat_softmax_f64(const double* x, double* out, int64_t N) {
    if (N <= 0) return;
    double max_val = x[0];
    for (int64_t i = 1; i < N; i++) if (x[i] > max_val) max_val = x[i];
    double sum = 0.0;
    for (int64_t i = 0; i < N; i++) {
        out[i] = exp(x[i] - max_val);
        sum += out[i];
    }
    double inv = sum > 0.0 ? (1.0 / sum) : 0.0;
    for (int64_t i = 0; i < N; i++) out[i] *= inv;
}

static inline void nat_apply_rope_f64(const double* x, double* out, int64_t N, int64_t pos, double base) {
    int64_t half = N / 2;
    for (int64_t i = 0; i < half; i++) {
        double freq = 1.0 / pow(base, (double)i / (double)half);
        double angle = (double)pos * freq;
        double c = cos(angle);
        double s = sin(angle);
        double x1 = x[i];
        double x2 = x[i + half];
        out[i] = x1 * c - x2 * s;
        out[i + half] = x1 * s + x2 * c;
    }
    if (N % 2 != 0) out[N - 1] = x[N - 1];
}

static inline int64_t nat_argmax_f64(const double* x, int64_t N) {
    if (N <= 0) return 0;
    int64_t best_idx = 0;
    double best_val = x[0];
    for (int64_t i = 1; i < N; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static inline double* nat_dup_f64(const double* src, int64_t n) {
    double* dst = (double*)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (dst && src && n > 0) memcpy(dst, src, (size_t)n * sizeof(double));
    return dst;
}

static inline void nat_ternary_matmul_vec(const int8_t* W, const double* x, double* y, int64_t M, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc;
    }
}

static inline void nat_pack_5trits(const int8_t* in, uint8_t* out_bytes, int64_t length) {
    int64_t out_idx = 0;
    for (int64_t i = 0; i < length; i += 5) {
        uint16_t b_val = 0;
        uint16_t mul = 1;
        for (int j = 0; j < 5; j++) {
            int8_t t = (i + j < length) ? (in[i + j] + 1) : 1;
            b_val += (uint16_t)(t * mul);
            mul *= 3;
        }
        out_bytes[out_idx++] = (uint8_t)b_val;
    }
}

static inline void nat_unpack_5trits(const uint8_t* in_bytes, int8_t* out, int64_t length) {
    int64_t out_idx = 0;
    int64_t num_bytes = (length + 4) / 5;
    for (int64_t b = 0; b < num_bytes && out_idx < length; b++) {
        uint8_t val = in_bytes[b];
        for (int step = 0; step < 5 && out_idx < length; step++) {
            out[out_idx++] = (int8_t)(val % 3) - 1;
            val /= 3;
        }
    }
}

static inline void nat_fused_gemv_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {
    int64_t bytes_per_row = (K + 4) / 5;
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const uint8_t* row_packed = W_packed + i * bytes_per_row;
        int64_t k = 0;
        for (int64_t b = 0; b < bytes_per_row && k < K; b++) {
            uint8_t val = row_packed[b];
            for (int step = 0; step < 5 && k < K; step++) {
                int8_t trit = (int8_t)(val % 3) - 1;
                val /= 3;
                if (trit == 1) acc += x[k];
                else if (trit == -1) acc -= x[k];
                k++;
            }
        }
        y[i] = acc * gamma;
    }
}

static inline void nat_gemv_vec_lut_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {
    int64_t num_groups = (K + 4) / 5;
    if (num_groups <= 0) return;
    double* lut = (double*)malloc(num_groups * 21 * sizeof(double));
    if (!lut) return;
    double* lut01 = lut;
    double* lut23 = lut + num_groups * 9;
    double* lut4  = lut + num_groups * 18;
    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 5;
        double x0 = (base < K) ? x[base] : 0.0;
        double x1 = (base + 1 < K) ? x[base + 1] : 0.0;
        double x2 = (base + 2 < K) ? x[base + 2] : 0.0;
        double x3 = (base + 3 < K) ? x[base + 3] : 0.0;
        double x4 = (base + 4 < K) ? x[base + 4] : 0.0;
        for (int t1 = 0; t1 < 3; t1++) {
            for (int t0 = 0; t0 < 3; t0++) {
                lut01[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x0 + (t1 - 1) * x1;
                lut23[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x2 + (t1 - 1) * x3;
            }
        }
        for (int t4 = 0; t4 < 3; t4++) {
            lut4[g * 3 + t4] = (t4 - 1) * x4;
        }
    }
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const uint8_t* row = W_packed + i * num_groups;
        for (int64_t g = 0; g < num_groups; g++) {
            uint8_t b = row[g];
            uint8_t idx01 = b % 9;
            uint8_t idx23 = (b / 9) % 9;
            uint8_t idx4  = b / 81;
            acc += lut01[g * 9 + idx01] + lut23[g * 9 + idx23] + lut4[g * 3 + idx4];
        }
        y[i] = acc * gamma;
    }
    free(lut);
}

static inline uint8_t* nat_pack_ternary_5trits(const int8_t* in, int64_t length) {
    int64_t num_bytes = (length + 4) / 5;
    uint8_t* out = (uint8_t*)malloc(num_bytes * sizeof(uint8_t));
    if (out) nat_pack_5trits(in, out, length);
    return out;
}
static inline int8_t* nat_unpack_ternary_5trits(const uint8_t* in, int64_t length) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (out) nat_unpack_5trits(in, out, length);
    return out;
}
static inline double* nat_gemv_vec_lut(const uint8_t* W_packed, const double* x, int64_t M, int64_t K) {
    double* y = (double*)malloc(M * sizeof(double));
    if (y) nat_gemv_vec_lut_5trits(W_packed, x, y, M, K, 1.0);
    return y;
}
static inline double nat_tensor_absmean(const double* W, int64_t length) {
    if (length <= 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < length; i++) s += fabs(W[i]);
    return s / (double)length;
}
static inline int8_t* nat_quantize_bitnet(const double* W, int64_t length) {
    double gamma = nat_tensor_absmean(W, length);
    double inv = 1.0 / (gamma + 1e-7);
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        double sc = W[i] * inv;
        double r = round(sc);
        if (r > 1.0) r = 1.0; else if (r < -1.0) r = -1.0;
        out[i] = (int8_t)r;
    }
    return out;
}
static inline double nat_mae(const double* a, const double* b, int64_t length) {
    if (length <= 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < length; i++) s += fabs(a[i] - b[i]);
    return s / (double)length;
}
static inline double nat_relative_error(const double* a, const double* b, int64_t length) {
    double norm_diff = 0.0, norm_a = 0.0;
    for (int64_t i = 0; i < length; i++) {
        double d = a[i] - b[i];
        norm_diff += d * d; norm_a += a[i] * a[i];
    }
    return norm_a > 0.0 ? sqrt(norm_diff / norm_a) : 0.0;
}
static inline double nat_pearson_corr(const double* a, const double* b, int64_t length) {
    if (length < 2) return 1.0;
    double ma = 0.0, mb = 0.0;
    for (int64_t i = 0; i < length; i++) { ma += a[i]; mb += b[i]; }
    ma /= (double)length; mb /= (double)length;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int64_t i = 0; i < length; i++) {
        double diff_a = a[i] - ma; double diff_b = b[i] - mb;
        num += diff_a * diff_b; da += diff_a * diff_a; db += diff_b * diff_b;
    }
    double den = sqrt(da * db);
    return den > 0.0 ? (num / den) : 0.0;
}
static inline double nat_cosine_sim(const double* a, const double* b, int64_t length) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < length; i++) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    double den = sqrt(na) * sqrt(nb);
    return den > 0.0 ? (dot / den) : 0.0;
}

static inline double* nat_fill_f64(double* arr, int64_t n, double val) {
    for (int64_t i = 0; i < n; i++) arr[i] = val; return arr;
}
static inline int64_t* nat_fill_i64(int64_t* arr, int64_t n, int64_t val) {
    for (int64_t i = 0; i < n; i++) arr[i] = val; return arr;
}
static inline double* nat_zeros_f64(double* arr, int64_t n) {
    memset(arr, 0, n * sizeof(double)); return arr;
}
static inline double* nat_ones_f64(double* arr, int64_t n) {
    for (int64_t i = 0; i < n; i++) arr[i] = 1.0; return arr;
}
static inline double nat_dot(const double* a, const double* b, int64_t n) {
    double acc = 0.0; for (int64_t i = 0; i < n; i++) acc += a[i] * b[i]; return acc;
}
static inline double nat_sum(const double* a, int64_t n) {
    double acc = 0.0; for (int64_t i = 0; i < n; i++) acc += a[i]; return acc;
}
static inline double nat_mean(const double* a, int64_t n) {
    return n > 0 ? (nat_sum(a, n) / (double)n) : 0.0;
}
static inline double nat_min(const double* a, int64_t n) {
    if (n <= 0) return 0.0;
    double m = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
static inline double nat_max(const double* a, int64_t n) {
    if (n <= 0) return 0.0;
    double m = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}

static inline void nat_gemv_f64(const double* A, const double* x, double* y, int64_t M, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const double* row = A + i * K;
        for (int64_t k = 0; k < K; k++) acc += row[k] * x[k];
        y[i] = acc;
    }
}
static inline void nat_gemv_scaled_f64(const double* A, const double* x, double* y, int64_t M, int64_t K, double scale) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const double* row = A + i * K;
        for (int64_t k = 0; k < K; k++) acc += row[k] * x[k];
        y[i] = acc * scale;
    }
}
static inline void nat_ternary_matmul_vec_scaled(const int8_t* W, const double* x, double* y, int64_t M, int64_t K, double scale) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc * scale;
    }
}
static inline void nat_gemv_trans_f64(const double* A, const double* x, double* y, int64_t M, int64_t N) {
    for (int64_t j = 0; j < N; j++) {
        double acc = 0.0;
        for (int64_t i = 0; i < M; i++) acc += x[i] * A[i * N + j];
        y[j] = acc;
    }
}
static inline void nat_gemm_f64(const double* A, const double* B, double* C, int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}
static inline void nat_random_ternary(int8_t* out, int64_t length, uint64_t seed) {
    uint64_t state = (seed != 0) ? seed : 123456789ULL;
    for (int64_t i = 0; i < length; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int r = (int)((state >> 32) % 3);
        out[i] = (int8_t)(r - 1);
    }
}

/* ------------------------------------------------------------------------- */
/* 4. Contiguous C99 KVCache for Autoregressive LLM Inference                */
/* ------------------------------------------------------------------------- */

typedef struct {
    double* k;
    double* v;
    int64_t max_seq_len;
    int64_t dim;
    int64_t seq_len;
} NatKVCache;

static inline NatKVCache nat_kvcache_init(int64_t max_seq_len, int64_t dim) {
    NatKVCache c;
    c.max_seq_len = max_seq_len > 0 ? max_seq_len : 1;
    c.dim = dim > 0 ? dim : 1;
    c.seq_len = 0;
    c.k = (double*)calloc((size_t)(c.max_seq_len * c.dim), sizeof(double));
    c.v = (double*)calloc((size_t)(c.max_seq_len * c.dim), sizeof(double));
    return c;
}

static inline void nat_kvcache_update(NatKVCache* c, const double* k_new, const double* v_new, int64_t n_tokens) {
    if (!c || !c->k || !c->v) return;
    if (n_tokens <= 0) n_tokens = 1;
    if (c->seq_len + n_tokens <= c->max_seq_len) {
        memcpy(c->k + c->seq_len * c->dim, k_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        memcpy(c->v + c->seq_len * c->dim, v_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        c->seq_len += n_tokens;
    } else if (n_tokens >= c->max_seq_len) {
        int64_t offset = (n_tokens - c->max_seq_len) * c->dim;
        memcpy(c->k, k_new + offset, (size_t)(c->max_seq_len * c->dim) * sizeof(double));
        memcpy(c->v, v_new + offset, (size_t)(c->max_seq_len * c->dim) * sizeof(double));
        c->seq_len = c->max_seq_len;
    } else {
        int64_t shift = c->seq_len + n_tokens - c->max_seq_len;
        int64_t keep = c->seq_len - shift;
        memmove(c->k, c->k + shift * c->dim, (size_t)(keep * c->dim) * sizeof(double));
        memmove(c->v, c->v + shift * c->dim, (size_t)(keep * c->dim) * sizeof(double));
        memcpy(c->k + keep * c->dim, k_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        memcpy(c->v + keep * c->dim, v_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        c->seq_len = c->max_seq_len;
    }
}

static inline void nat_kvcache_clear(NatKVCache* c) {
    if (c) c->seq_len = 0;
}

/* ------------------------------------------------------------------------- */
/* 5. Formatted Printing, Result[T] System and Safe C99 I/O                  */
/* ------------------------------------------------------------------------- */

static inline void nat_print_vec_f64(const double* v, int64_t n) {
    printf("[");
    for (int64_t i = 0; i < n; i++) printf("%g%s", v[i], (i < n - 1) ? ", " : "");
    printf("]");
}
static inline void nat_print_mat_f64(const double* A, int64_t M, int64_t N) {
    printf("[");
    for (int64_t i = 0; i < M; i++) {
        printf("[");
        for (int64_t j = 0; j < N; j++) printf("%g%s", A[i * N + j], (j < N - 1) ? ", " : "");
        printf("]%s", (i < M - 1) ? ", " : "");
    }
    printf("]");
}

typedef struct {
    const char* val_str;
    double val_f64;
    int64_t val_i64;
    const char* err;
    bool is_ok;
} NatResult;

static inline NatResult nat_ok_str(const char* val) {
    NatResult r; memset(&r, 0, sizeof(r)); r.val_str = val; r.is_ok = true; return r;
}
static inline NatResult nat_ok_i64(int64_t val) {
    NatResult r; memset(&r, 0, sizeof(r)); r.val_i64 = val; r.val_f64 = (double)val; r.is_ok = true; return r;
}
static inline NatResult nat_ok_f64(double val) {
    NatResult r; memset(&r, 0, sizeof(r)); r.val_f64 = val; r.val_i64 = (int64_t)val; r.is_ok = true; return r;
}
static inline NatResult nat_err(const char* err_msg) {
    NatResult r; memset(&r, 0, sizeof(r)); r.err = err_msg; r.is_ok = false; return r;
}
static inline bool nat_is_ok(NatResult r) { return r.is_ok; }
static inline bool nat_is_err(NatResult r) { return !r.is_ok; }
static inline const char* nat_unwrap_str(NatResult r) {
    if (!r.is_ok) {
        fprintf(stderr, "Runtime error: attempted to unwrap a Result containing an error: %s\n", r.err ? r.err : "unknown");
        exit(1);
    }
    return r.val_str ? r.val_str : "";
}
static inline int64_t nat_unwrap_i64(NatResult r) {
    if (!r.is_ok) {
        fprintf(stderr, "Runtime error: attempted to unwrap a Result containing an error: %s\n", r.err ? r.err : "unknown");
        exit(1);
    }
    return r.val_i64;
}
static inline double nat_unwrap_f64(NatResult r) {
    if (!r.is_ok) {
        fprintf(stderr, "Runtime error: attempted to unwrap a Result containing an error: %s\n", r.err ? r.err : "unknown");
        exit(1);
    }
    return r.val_f64;
}
static inline const char* nat_unwrap_or_str(NatResult r, const char* def) {
    return r.is_ok ? (r.val_str ? r.val_str : "") : def;
}
static inline int64_t nat_unwrap_or_i64(NatResult r, int64_t def) {
    return r.is_ok ? r.val_i64 : def;
}
static inline double nat_unwrap_or_f64(NatResult r, double def) {
    return r.is_ok ? r.val_f64 : def;
}
static inline const char* nat_error_message(NatResult r) {
    return (r.is_ok || !r.err) ? "" : r.err;
}
static inline bool nat_exists_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}
static inline NatResult nat_read_file_safe(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nat_err("file not found or inaccessible");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return nat_err("memory allocation failed for reading"); }
    size_t rd = fread(buf, 1, sz, f);
    buf[rd] = '\0';
    fclose(f);
    return nat_ok_str(buf);
}
static inline const char* nat_read_file(const char* path) {
    NatResult r = nat_read_file_safe(path);
    return nat_unwrap_str(r);
}

#endif /* NATIVA_RT_H */
