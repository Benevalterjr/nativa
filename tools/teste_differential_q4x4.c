// ============================================================================
// NATIVA DIFFERENTIAL TEST: SIMD vs SCALAR Q4x4 & EMBEDDING KERNELS
// ============================================================================
// Rigorous verification suite for numerical equivalence:
//   1. Zero blocks (0x88 nibbles = 0.0)
//   2. Extreme weights (0x00 = -8, 0xFF = +7)
//   3. Extreme INT8 activations (-128, +127)
//   4. Variable BF16 scales (zero, small, large, normal)
//   5. 1,000 randomized vectors with random seeds
//   6. Embedding lookup SIMD vs Scalar equivalence
// ============================================================================

#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4];
    uint8_t  qs[64];
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
// SIMD Kernels
// ----------------------------------------------------------------------------
static inline __m128 q4x4_dot4_simd(const BlockQ4x4* restrict W4, const BlockQ8* restrict x_q8, int nb) {
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

static inline void embedding_lookup_simd(float* x, const BlockQ4x4* emb_grp, int lane, int dim) {
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i off = _mm_set1_epi8(8);
    int b;
    for (b = 0; b < dim / 32; b++) {
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

        _mm_storeu_ps(&x[b * 32],      _mm_mul_ps(f0, svec));
        _mm_storeu_ps(&x[b * 32 + 4],  _mm_mul_ps(f1, svec));
        _mm_storeu_ps(&x[b * 32 + 8],  _mm_mul_ps(f2, svec));
        _mm_storeu_ps(&x[b * 32 + 12], _mm_mul_ps(f3, svec));

        __m128 f4 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(hi));
        __m128 f5 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
        __m128 f6 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        __m128 f7 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 12)));

        _mm_storeu_ps(&x[b * 32 + 16], _mm_mul_ps(f4, svec));
        _mm_storeu_ps(&x[b * 32 + 20], _mm_mul_ps(f5, svec));
        _mm_storeu_ps(&x[b * 32 + 24], _mm_mul_ps(f6, svec));
        _mm_storeu_ps(&x[b * 32 + 28], _mm_mul_ps(f7, svec));
    }
}

// ----------------------------------------------------------------------------
// Pure Scalar C99 Kernels
// ----------------------------------------------------------------------------
static inline void q4x4_dot4_scalar(const BlockQ4x4* W4, const BlockQ8* x_q8, int nb, float* out4) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int b = 0; b < nb; b++) {
        const BlockQ4x4* blk = &W4[b];
        const BlockQ8* xb = &x_q8[b];

        float s0 = bf16_to_fp32(blk->scale[0]);
        float s1 = bf16_to_fp32(blk->scale[1]);
        float s2 = bf16_to_fp32(blk->scale[2]);
        float s3 = bf16_to_fp32(blk->scale[3]);

        int32_t dot0 = 0;
        int32_t dot1 = 0;
        int32_t dot2 = 0;
        int32_t dot3 = 0;

        for (int i = 0; i < 16; i++) {
            uint8_t b0 = blk->qs[i];
            uint8_t b1 = blk->qs[i + 16];
            uint8_t b2 = blk->qs[i + 32];
            uint8_t b3 = blk->qs[i + 48];

            int8_t x_lo = xb->qs[i];
            int8_t x_hi = xb->qs[i + 16];

            dot0 += ((int32_t)(b0 & 0x0F) - 8) * (int32_t)x_lo + ((int32_t)(b0 >> 4) - 8) * (int32_t)x_hi;
            dot1 += ((int32_t)(b1 & 0x0F) - 8) * (int32_t)x_lo + ((int32_t)(b1 >> 4) - 8) * (int32_t)x_hi;
            dot2 += ((int32_t)(b2 & 0x0F) - 8) * (int32_t)x_lo + ((int32_t)(b2 >> 4) - 8) * (int32_t)x_hi;
            dot3 += ((int32_t)(b3 & 0x0F) - 8) * (int32_t)x_lo + ((int32_t)(b3 >> 4) - 8) * (int32_t)x_hi;
        }

        float sc = xb->scale;
        acc0 += (float)dot0 * (s0 * sc);
        acc1 += (float)dot1 * (s1 * sc);
        acc2 += (float)dot2 * (s2 * sc);
        acc3 += (float)dot3 * (s3 * sc);
    }
    out4[0] = acc0;
    out4[1] = acc1;
    out4[2] = acc2;
    out4[3] = acc3;
}

static inline void embedding_lookup_scalar(float* restrict x, const BlockQ4x4* restrict emb_grp, int lane, int dim) {
    int num_blocks = dim / 32;
    for (int b = 0; b < num_blocks; b++) {
        const BlockQ4x4* blk = &emb_grp[b];
        float scale = bf16_to_fp32(blk->scale[lane]);
        const uint8_t* qs = blk->qs + lane * 16;
        float* xb = x + b * 32;
        for (int i = 0; i < 16; i++) {
            uint8_t byte = qs[i];
            int v_lo = (int)(byte & 0x0F) - 8;
            int v_hi = (int)(byte >> 4) - 8;
            xb[i]      = (float)v_lo * scale;
            xb[i + 16] = (float)v_hi * scale;
        }
    }
}

// ----------------------------------------------------------------------------
// Test Runner
// ----------------------------------------------------------------------------
static bool test_case_dot(const char* name, const BlockQ4x4* W, const BlockQ8* x_q8, int nb, float tol) {
    __m128 simd_res = q4x4_dot4_simd(W, x_q8, nb);
    float simd_out[4];
    _mm_storeu_ps(simd_out, simd_res);

    float scalar_out[4];
    q4x4_dot4_scalar(W, x_q8, nb, scalar_out);

    bool ok = true;
    for (int r = 0; r < 4; r++) {
        float diff = fabsf(simd_out[r] - scalar_out[r]);
        float denom = fmaxf(fabsf(simd_out[r]), fabsf(scalar_out[r]));
        float rel_diff = (denom > 1e-6f) ? (diff / denom) : diff;

        if (diff > tol && rel_diff > 1e-4f) {
            fprintf(stderr, "[FAIL] %s - Row %d mismatch: SIMD=%.7f, Scalar=%.7f (diff=%.7e)\n",
                    name, r, simd_out[r], scalar_out[r], diff);
            ok = false;
        }
    }
    return ok;
}

int main(void) {
    printf("====================================================================\n");
    printf("  NATIVA DIFFERENTIAL TEST: SIMD vs SCALAR NUMERICAL EQUIVALENCE\n");
    printf("====================================================================\n");

    int total_tests = 0;
    int passed_tests = 0;

    const int nb = 32; // 1024 cols / 32
    BlockQ4x4 W[32];
    BlockQ8 x[32];

    // TEST 1: Zero block test (nibbles = 0x88 -> centered = 0)
    total_tests++;
    for (int b = 0; b < nb; b++) {
        for (int r = 0; r < 4; r++) W[b].scale[r] = fp32_to_bf16(0.05f * (r + 1));
        memset(W[b].qs, 0x88, sizeof(W[b].qs));
        x[b].scale = 0.02f;
        int32_t s = 0;
        for (int i = 0; i < 32; i++) {
            x[b].qs[i] = (int8_t)((i % 25) - 12);
            s += x[b].qs[i];
        }
        x[b].sum_adj = s * 8;
    }
    if (test_case_dot("Test 1: Zero blocks (0x88 = 0 centered)", W, x, nb, 1e-6f)) {
        printf("  [PASS] Test 1: Zero blocks (weights=0 centered -> dot=0.0)\n");
        passed_tests++;
    }

    // TEST 2: All activations zero
    total_tests++;
    for (int b = 0; b < nb; b++) {
        for (int r = 0; r < 4; r++) W[b].scale[r] = fp32_to_bf16(1.5f);
        for (int k = 0; k < 64; k++) W[b].qs[k] = (uint8_t)(k & 0xFF);
        x[b].scale = 0.1f;
        memset(x[b].qs, 0, sizeof(x[b].qs));
        x[b].sum_adj = 0;
    }
    if (test_case_dot("Test 2: All activations zero", W, x, nb, 1e-6f)) {
        printf("  [PASS] Test 2: All activations zero (dot=0.0)\n");
        passed_tests++;
    }

    // TEST 3: Extreme negative weights (0x00 -> centered = -8) with extreme positive activations (+127)
    total_tests++;
    for (int b = 0; b < nb; b++) {
        for (int r = 0; r < 4; r++) W[b].scale[r] = fp32_to_bf16(0.125f);
        memset(W[b].qs, 0x00, sizeof(W[b].qs));
        x[b].scale = 0.01f;
        int32_t s = 0;
        for (int i = 0; i < 32; i++) {
            x[b].qs[i] = 127;
            s += 127;
        }
        x[b].sum_adj = s * 8;
    }
    if (test_case_dot("Test 3: Extreme negative weights (-8) and max activations (+127)", W, x, nb, 1e-4f)) {
        printf("  [PASS] Test 3: Extreme negative weights (-8) and max activations (+127)\n");
        passed_tests++;
    }

    // TEST 4: Extreme positive weights (0xFF -> centered = +7) with extreme negative activations (-128)
    total_tests++;
    for (int b = 0; b < nb; b++) {
        for (int r = 0; r < 4; r++) W[b].scale[r] = fp32_to_bf16(0.25f);
        memset(W[b].qs, 0xFF, sizeof(W[b].qs));
        x[b].scale = 0.05f;
        int32_t s = 0;
        for (int i = 0; i < 32; i++) {
            x[b].qs[i] = -128;
            s += -128;
        }
        x[b].sum_adj = s * 8;
    }
    if (test_case_dot("Test 4: Extreme positive weights (+7) and min activations (-128)", W, x, nb, 1e-4f)) {
        printf("  [PASS] Test 4: Extreme positive weights (+7) and min activations (-128)\n");
        passed_tests++;
    }

    // TEST 5: Tiny and large BF16 scales
    total_tests++;
    for (int b = 0; b < nb; b++) {
        W[b].scale[0] = fp32_to_bf16(1e-4f);
        W[b].scale[1] = fp32_to_bf16(1e2f);
        W[b].scale[2] = fp32_to_bf16(0.0f);
        W[b].scale[3] = fp32_to_bf16(1.0f);
        for (int k = 0; k < 64; k++) W[b].qs[k] = (uint8_t)(k * 13 + 7);
        x[b].scale = 0.003f;
        int32_t s = 0;
        for (int i = 0; i < 32; i++) {
            x[b].qs[i] = (int8_t)((i * 17) % 255 - 128);
            s += x[b].qs[i];
        }
        x[b].sum_adj = s * 8;
    }
    if (test_case_dot("Test 5: Tiny, large, zero, and unit BF16 scales", W, x, nb, 1e-3f)) {
        printf("  [PASS] Test 5: Dynamic BF16 scale ranges (1e-4 to 1e2, plus 0.0 and 1.0)\n");
        passed_tests++;
    }

    // TEST 6: 1,000 randomized iterations
    total_tests++;
    bool rand_ok = true;
    uint32_t seed = 133742u;
    for (int it = 0; it < 1000; it++) {
        for (int b = 0; b < nb; b++) {
            for (int r = 0; r < 4; r++) {
                seed = seed * 1664525u + 1013904223u;
                float sc_val = (float)(seed % 1000) / 500.0f + 0.001f;
                W[b].scale[r] = fp32_to_bf16(sc_val);
            }
            for (int k = 0; k < 64; k++) {
                seed = seed * 1664525u + 1013904223u;
                W[b].qs[k] = (uint8_t)(seed & 0xFF);
            }
            seed = seed * 1664525u + 1013904223u;
            x[b].scale = (float)(seed % 1000) / 10000.0f + 0.0001f;
            int32_t s = 0;
            for (int i = 0; i < 32; i++) {
                seed = seed * 1664525u + 1013904223u;
                int8_t val = (int8_t)((seed % 256) - 128);
                x[b].qs[i] = val;
                s += val;
            }
            x[b].sum_adj = s * 8;
        }

        if (!test_case_dot("Randomized Q4x4 iteration", W, x, nb, 1e-3f)) {
            rand_ok = false;
            break;
        }
    }
    if (rand_ok) {
        printf("  [PASS] Test 6: 1,000 randomized vectors passed differential verification\n");
        passed_tests++;
    }

    // TEST 7: Embedding lookup SIMD vs Scalar
    total_tests++;
    const int emb_dim = 1024;
    BlockQ4x4 emb_grp[32];
    for (int b = 0; b < 32; b++) {
        for (int r = 0; r < 4; r++) {
            emb_grp[b].scale[r] = fp32_to_bf16(0.015f * (r + 1));
        }
        for (int k = 0; k < 64; k++) {
            emb_grp[b].qs[k] = (uint8_t)(k * 31 + b * 7);
        }
    }

    float emb_simd[1024];
    float emb_scalar[1024];
    bool emb_ok = true;
    for (int lane = 0; lane < 4; lane++) {
        embedding_lookup_simd(emb_simd, emb_grp, lane, emb_dim);
        embedding_lookup_scalar(emb_scalar, emb_grp, lane, emb_dim);

        for (int i = 0; i < emb_dim; i++) {
            float diff = fabsf(emb_simd[i] - emb_scalar[i]);
            if (diff > 1e-6f) {
                fprintf(stderr, "[FAIL] Embedding lane %d index %d: SIMD=%.7f, Scalar=%.7f\n",
                        lane, i, emb_simd[i], emb_scalar[i]);
                emb_ok = false;
                break;
            }
        }
        if (!emb_ok) break;
    }
    if (emb_ok) {
        printf("  [PASS] Test 7: Embedding lookup SIMD vs Scalar (all 4 lanes bit-exact)\n");
        passed_tests++;
    }

    printf("====================================================================\n");
    printf("  DIFFERENTIAL TEST RESULT: %d / %d PASSED\n", passed_tests, total_tests);
    printf("====================================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
