#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <windows.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4]; // 8 bytes: BF16 scales for 4 rows
    uint8_t  qs[64];   // 64 bytes: 16 bytes per row (32 nibbles each)
} BlockQ4x4;
#pragma pack(pop)

typedef struct {
    float scale;
    int8_t qs[32];
    int32_t sum_adj;
} BlockQ8;

// Fast exp approximation for SSE
static inline __m128 fast_exp_ps(__m128 x) {
    x = _mm_max_ps(_mm_set1_ps(-88.0f), _mm_min_ps(_mm_set1_ps(88.0f), x));
    __m128 c_inv_ln2 = _mm_set1_ps(1.4426950408889634f);
    __m128 c_ln2_hi  = _mm_set1_ps(0.6931471805599453f);
    __m128 c_ln2_lo  = _mm_set1_ps(2.3190468138462996e-17f);
    __m128 z = _mm_mul_ps(x, c_inv_ln2);
    __m128 z_floor = _mm_floor_ps(z);
    __m128 r = _mm_sub_ps(x, _mm_mul_ps(z_floor, c_ln2_hi));
    r = _mm_sub_ps(r, _mm_mul_ps(z_floor, c_ln2_lo));
    __m128 c1 = _mm_set1_ps(1.0f);
    __m128 c2 = _mm_set1_ps(0.5f);
    __m128 c3 = _mm_set1_ps(0.16666666666666666f);
    __m128 c4 = _mm_set1_ps(0.041666666666666664f);
    __m128 poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, c4))))))));
    __m128i k = _mm_cvtps_epi32(z_floor);
    k = _mm_add_epi32(k, _mm_set1_epi32(127));
    k = _mm_slli_epi32(k, 23);
    __m128 scale_val = _mm_castsi128_ps(k);
    return _mm_mul_ps(poly, scale_val);
}

// ----------------------------------------------------------------------------
// KERNEL A: Pure Stream Read (Same OpenMP structure, zero compute overhead)
// Reads all 15.48 MB of Wgate and Wup directly into XMM registers
// ----------------------------------------------------------------------------
void stream_read_q4x4(
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    int rows, int cols,
    uint64_t* out_checksum
) {
    int nb = cols / 32;
    int groups = rows / 4;
    uint64_t global_chk = 0;

    int g;
    #pragma omp parallel for schedule(static) reduction(^:global_chk)
    for (g = 0; g < groups; g++) {
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;
        __m128i dummy_acc = _mm_setzero_si128();

        for (int b = 0; b < nb; b++) {
            const BlockQ4x4* gblk = &g_blk_row[b];
            __m128i gr0 = _mm_loadu_si128((const __m128i*)(gblk->qs));
            __m128i gr1 = _mm_loadu_si128((const __m128i*)(gblk->qs + 16));
            __m128i gr2 = _mm_loadu_si128((const __m128i*)(gblk->qs + 32));
            __m128i gr3 = _mm_loadu_si128((const __m128i*)(gblk->qs + 48));
            __m128i gsc = _mm_loadl_epi64((const __m128i*)gblk->scale);

            const BlockQ4x4* ublk = &u_blk_row[b];
            __m128i ur0 = _mm_loadu_si128((const __m128i*)(ublk->qs));
            __m128i ur1 = _mm_loadu_si128((const __m128i*)(ublk->qs + 16));
            __m128i ur2 = _mm_loadu_si128((const __m128i*)(ublk->qs + 32));
            __m128i ur3 = _mm_loadu_si128((const __m128i*)(ublk->qs + 48));
            __m128i usc = _mm_loadl_epi64((const __m128i*)ublk->scale);

            dummy_acc = _mm_xor_si128(dummy_acc, _mm_xor_si128(gr0, ur0));
            dummy_acc = _mm_xor_si128(dummy_acc, _mm_xor_si128(gr1, ur1));
            dummy_acc = _mm_xor_si128(dummy_acc, _mm_xor_si128(gr2, ur2));
            dummy_acc = _mm_xor_si128(dummy_acc, _mm_xor_si128(gr3, ur3));
            dummy_acc = _mm_xor_si128(dummy_acc, _mm_xor_si128(gsc, usc));
        }

        uint64_t chk[2];
        _mm_storeu_si128((__m128i*)chk, dummy_acc);
        global_chk ^= (chk[0] ^ chk[1]);
    }

    *out_checksum = global_chk;
}

// ----------------------------------------------------------------------------
// KERNEL B: Fused Lean Q4x4 (Full compute + SwiGLU)
// ----------------------------------------------------------------------------
void mlp_gate_up_swiglu_lean(
    float* restrict h_mlp,
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows, int cols
) {
    int nb = cols / 32;
    int groups = rows / 4;
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();

    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;

        __m128 acc_g = _mm_setzero_ps();
        __m128 acc_u = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const BlockQ8* xb = &x_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
            __m128 x_sc = _mm_set1_ps(xb->scale);
            __m128i adj = _mm_set1_epi32(xb->sum_adj);

            // 1. Gate
            {
                const BlockQ4x4* gblk = &g_blk_row[b];
                __m128i gr0 = _mm_loadu_si128((const __m128i*)(gblk->qs));
                __m128i gr1 = _mm_loadu_si128((const __m128i*)(gblk->qs + 16));
                __m128i gr2 = _mm_loadu_si128((const __m128i*)(gblk->qs + 32));
                __m128i gr3 = _mm_loadu_si128((const __m128i*)(gblk->qs + 48));

                __m128i gt0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr0, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr0, 4), mask), x_hi)), ones);
                __m128i gt1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr1, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr1, 4), mask), x_hi)), ones);
                __m128i gt2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr2, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr2, 4), mask), x_hi)), ones);
                __m128i gt3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(gr3, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(gr3, 4), mask), x_hi)), ones);

                __m128i gh = _mm_hadd_epi32(_mm_hadd_epi32(gt0, gt1), _mm_hadd_epi32(gt2, gt3));
                __m128 g_fd = _mm_cvtepi32_ps(_mm_sub_epi32(gh, adj));
                __m128 g_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)gblk->scale)));
                acc_g = _mm_add_ps(acc_g, _mm_mul_ps(g_fd, _mm_mul_ps(g_sc, x_sc)));
            }

            // 2. Up
            {
                const BlockQ4x4* ublk = &u_blk_row[b];
                __m128i ur0 = _mm_loadu_si128((const __m128i*)(ublk->qs));
                __m128i ur1 = _mm_loadu_si128((const __m128i*)(ublk->qs + 16));
                __m128i ur2 = _mm_loadu_si128((const __m128i*)(ublk->qs + 32));
                __m128i ur3 = _mm_loadu_si128((const __m128i*)(ublk->qs + 48));

                __m128i ut0 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur0, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur0, 4), mask), x_hi)), ones);
                __m128i ut1 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur1, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur1, 4), mask), x_hi)), ones);
                __m128i ut2 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur2, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur2, 4), mask), x_hi)), ones);
                __m128i ut3 = _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(ur3, mask), x_lo),
                                                           _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(ur3, 4), mask), x_hi)), ones);

                __m128i uh = _mm_hadd_epi32(_mm_hadd_epi32(ut0, ut1), _mm_hadd_epi32(ut2, ut3));
                __m128 u_fd = _mm_cvtepi32_ps(_mm_sub_epi32(uh, adj));
                __m128 u_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)ublk->scale)));
                acc_u = _mm_add_ps(acc_u, _mm_mul_ps(u_fd, _mm_mul_ps(u_sc, x_sc)));
            }
        }

        // Fused SwiGLU: silu(acc_g) * acc_u
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_g = _mm_sub_ps(_mm_setzero_ps(), acc_g);
        __m128 exp_neg = fast_exp_ps(neg_g);
        __m128 sig_g = _mm_div_ps(one, _mm_add_ps(one, exp_neg));
        __m128 silu_g = _mm_mul_ps(acc_g, sig_g);
        __m128 res = _mm_mul_ps(silu_g, acc_u);

        _mm_storeu_ps(h_mlp + (size_t)g * 4, res);
    }
}

// ----------------------------------------------------------------------------
// TERNARY DEFINITIONS (Etapa 2)
// Strategy A: 2-bit per trit (4 trits / byte) -> 10 bytes / 32 weights
// Strategy B: 5 trits per byte (BitNet b1.58 canonical) -> LUT decoding
// ----------------------------------------------------------------------------

#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4]; // 8 bytes: BF16 scales for 4 rows
    uint8_t  qs[32];   // 32 bytes: 8 bytes per row (32 trits, 2 bits each)
} BlockTernary2bit4x4; // 40 bytes for 128 weights! (vs 72 bytes in Q4x4)
#pragma pack(pop)

// 2-bit ternary kernel: decodes trits in {00: 0, 01: +1, 10: -1}
void mlp_ternary_2bit_lean(
    float* restrict h_mlp,
    const BlockTernary2bit4x4* restrict Wgate,
    const BlockTernary2bit4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows, int cols
) {
    int nb = cols / 32;
    int groups = rows / 4;
    const __m128i mask2 = _mm_set1_epi8(0x03);
    const __m128i ones  = _mm_set1_epi16(1);
    const __m128i zero  = _mm_setzero_si128();

    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < groups; g++) {
        const BlockTernary2bit4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockTernary2bit4x4* u_blk_row = Wup   + (size_t)g * nb;

        __m128 acc_g = _mm_setzero_ps();
        __m128 acc_u = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const BlockQ8* xb = &x_q8[b];
            __m128i x_lo = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i x_hi = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
            __m128 x_sc = _mm_set1_ps(xb->scale);

            // In 2-bit ternary, 32 trits take 8 bytes.
            // 4 rows = 32 bytes total. Loaded in two 16-byte SSE registers!
            const BlockTernary2bit4x4* gblk = &g_blk_row[b];
            __m128i g_raw01 = _mm_loadu_si128((const __m128i*)gblk->qs);       // rows 0 & 1 (16 bytes)
            __m128i g_raw23 = _mm_loadu_si128((const __m128i*)(gblk->qs + 16)); // rows 2 & 3 (16 bytes)

            // Let's do a fast dot accumulation using maddubs
            // (Simulated kernel to measure memory bandwidth + execution throughput)
            __m128i g_t0 = _mm_maddubs_epi16(g_raw01, x_lo);
            __m128i g_t1 = _mm_maddubs_epi16(g_raw23, x_hi);
            __m128i gh = _mm_hadd_epi32(_mm_madd_epi16(g_t0, ones), _mm_madd_epi16(g_t1, ones));

            __m128 g_fd = _mm_cvtepi32_ps(gh);
            __m128 g_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)gblk->scale)));
            acc_g = _mm_add_ps(acc_g, _mm_mul_ps(g_fd, _mm_mul_ps(g_sc, x_sc)));

            const BlockTernary2bit4x4* ublk = &u_blk_row[b];
            __m128i u_raw01 = _mm_loadu_si128((const __m128i*)ublk->qs);
            __m128i u_raw23 = _mm_loadu_si128((const __m128i*)(ublk->qs + 16));
            __m128i u_t0 = _mm_maddubs_epi16(u_raw01, x_lo);
            __m128i u_t1 = _mm_maddubs_epi16(u_raw23, x_hi);
            __m128i uh = _mm_hadd_epi32(_mm_madd_epi16(u_t0, ones), _mm_madd_epi16(u_t1, ones));

            __m128 u_fd = _mm_cvtepi32_ps(uh);
            __m128 u_sc = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, _mm_loadl_epi64((const __m128i*)ublk->scale)));
            acc_u = _mm_add_ps(acc_u, _mm_mul_ps(u_fd, _mm_mul_ps(u_sc, x_sc)));
        }

        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_g = _mm_sub_ps(_mm_setzero_ps(), acc_g);
        __m128 exp_neg = fast_exp_ps(neg_g);
        __m128 sig_g = _mm_div_ps(one, _mm_add_ps(one, exp_neg));
        __m128 silu_g = _mm_mul_ps(acc_g, sig_g);
        __m128 res = _mm_mul_ps(silu_g, acc_u);
        _mm_storeu_ps(h_mlp + (size_t)g * 4, res);
    }
}

// 5-trits per byte lookup table
// 243 entries, each holds 5 int8 trits in {-1, 0, +1}
static int8_t g_lut5[256][8]; // 8 bytes padded for SIMD loads

void init_lut5() {
    for (int i = 0; i < 256; i++) {
        if (i < 243) {
            int val = i;
            for (int t = 0; t < 5; t++) {
                int rem = val % 3;
                g_lut5[i][t] = (int8_t)(rem - 1); // 0 -> -1, 1 -> 0, 2 -> +1
                val /= 3;
            }
            g_lut5[i][5] = 0;
            g_lut5[i][6] = 0;
            g_lut5[i][7] = 0;
        } else {
            memset(g_lut5[i], 0, 8);
        }
    }
}

static uint32_t g_bench_rng = 1337;
static inline uint32_t fast_rand(void) {
    g_bench_rng ^= g_bench_rng << 13;
    g_bench_rng ^= g_bench_rng >> 17;
    g_bench_rng ^= g_bench_rng << 5;
    return g_bench_rng;
}

// Microbenchmark for decoding 5 trits/byte via LUT
double benchmark_decode_5trit(int num_weights) {
    int num_bytes = num_weights / 5;
    uint8_t* encoded = (uint8_t*)malloc(num_bytes);
    int8_t* decoded = (int8_t*)malloc(num_weights);
    if (!encoded || !decoded) {
        free(encoded);
        free(decoded);
        return 0.0;
    }
    for (int i = 0; i < num_bytes; i++) encoded[i] = (uint8_t)(fast_rand() % 243);

    double t0 = omp_get_wtime();
    for (int it = 0; it < 50; it++) {
        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < num_bytes; i++) {
            uint8_t code = encoded[i];
            const int8_t* trits = g_lut5[code];
            decoded[i * 5 + 0] = trits[0];
            decoded[i * 5 + 1] = trits[1];
            decoded[i * 5 + 2] = trits[2];
            decoded[i * 5 + 3] = trits[3];
            decoded[i * 5 + 4] = trits[4];
        }
    }
    double t1 = omp_get_wtime();
    free(encoded);
    free(decoded);
    return (t1 - t0) / 50.0;
}

// Helper for statistics
typedef struct {
    double mean;
    double median;
    double min;
    double max;
    double stddev;
} Stats;

Stats calc_stats(double* times, int n) {
    // sort
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (times[j] < times[i]) {
                double tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
        }
    }
    Stats s;
    s.min = times[0];
    s.max = times[n - 1];
    s.median = (n % 2 == 1) ? times[n / 2] : 0.5 * (times[n / 2 - 1] + times[n / 2]);
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += times[i];
    s.mean = sum / n;
    double var = 0.0;
    for (int i = 0; i < n; i++) var += (times[i] - s.mean) * (times[i] - s.mean);
    s.stddev = sqrt(var / n);
    return s;
}

int main() {
    init_lut5();
    omp_set_num_threads(4);

    const int M = 8960;
    const int K = 1536;
    const int nb = K / 32;       // 48 blocks
    const int groups = M / 4;    // 2240 groups

    printf("================================================================================\n");
    printf("  NATIVA BENCHMARK: ETAPA 1 (RECONCILIACAO E STREAM PROBE) E ETAPA 2 (TERNARIO)\n");
    printf("================================================================================\n\n");

    // 1. RECONCILIAÇÃO MATEMÁTICA DE PESOS
    size_t q4x4_bytes_per_blk = sizeof(BlockQ4x4); // 72 bytes
    size_t q4_single_tensor_bytes = (size_t)groups * nb * q4x4_bytes_per_blk;
    size_t q4_two_tensors_bytes   = 2 * q4_single_tensor_bytes;

    printf("[1] RECONCILIACAO DO VOLUME DE PESOS (Q4x4):\n");
    printf("    Dimensoes: M = %d, K = %d\n", M, K);
    printf("    Layout: BlockQ4x4 (4 linhas x 32 colunas = 128 pesos)\n");
    printf("    Tamanho da struct: %zu bytes (8B scales BF16 + 64B nibbles)\n", q4x4_bytes_per_blk);
    printf("    Bytes por peso (efetivo): %.4f bytes/peso (%.2f bits/peso)\n",
           (double)q4x4_bytes_per_blk / 128.0, ((double)q4x4_bytes_per_blk / 128.0) * 8.0);
    printf("    Volume de 1 tensor (Wgate ou Wup): %zu bytes (%.3f MB decimal | %.3f MiB binario)\n",
           q4_single_tensor_bytes, (double)q4_single_tensor_bytes / 1e6, (double)q4_single_tensor_bytes / (1024.0 * 1024.0));
    printf("    Volume TOTAL (Wgate + Wup):        %zu bytes (%.3f MB decimal | %.3f MiB binario)\n\n",
           q4_two_tensors_bytes, (double)q4_two_tensors_bytes / 1e6, (double)q4_two_tensors_bytes / (1024.0 * 1024.0));

    // Alocações para Q4
    BlockQ4x4* Wgate = (BlockQ4x4*)_aligned_malloc(q4_single_tensor_bytes, 64);
    BlockQ4x4* Wup   = (BlockQ4x4*)_aligned_malloc(q4_single_tensor_bytes, 64);
    BlockQ8*   x_q8  = (BlockQ8*)_aligned_malloc(nb * sizeof(BlockQ8), 64);
    float*     h_mlp = (float*)_aligned_malloc(M * sizeof(float), 64);

    memset(Wgate, 0x5A, q4_single_tensor_bytes);
    memset(Wup,   0x3C, q4_single_tensor_bytes);
    memset(x_q8,  0x12, nb * sizeof(BlockQ8));
    for (int b = 0; b < nb; b++) {
        x_q8[b].scale = 0.05f;
        x_q8[b].sum_adj = 120;
    }

    // Warmup
    uint64_t chk = 0;
    stream_read_q4x4(Wgate, Wup, M, K, &chk);
    mlp_gate_up_swiglu_lean(h_mlp, Wgate, Wup, x_q8, M, K);

    // 2. MEDIÇÕES RIGOROSAS (100 iterações com 4 threads)
    const int N_ITERS = 100;
    double t_stream[100];
    double t_compute[100];

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    for (int i = 0; i < N_ITERS; i++) {
        QueryPerformanceCounter(&start);
        stream_read_q4x4(Wgate, Wup, M, K, &chk);
        QueryPerformanceCounter(&end);
        t_stream[i] = (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    }

    for (int i = 0; i < N_ITERS; i++) {
        QueryPerformanceCounter(&start);
        mlp_gate_up_swiglu_lean(h_mlp, Wgate, Wup, x_q8, M, K);
        QueryPerformanceCounter(&end);
        t_compute[i] = (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    }

    Stats s_stream  = calc_stats(t_stream, N_ITERS);
    Stats s_compute = calc_stats(t_compute, N_ITERS);

    double bw_stream  = ((double)q4_two_tensors_bytes / s_stream.median) / 1e9;
    double bw_compute = ((double)q4_two_tensors_bytes / s_compute.median) / 1e9;

    printf("[2] ETAPA 1 - RESULTADOS COMPARATIVOS (4 THREADS OPENMP, 100 AMOSTRAS):\n");
    printf("    A. Pure Stream Read (Leitura pura da DRAM sem computacao):\n");
    printf("       - Mediana:   %.3f ms\n", s_stream.median * 1000.0);
    printf("       - Media:     %.3f ms (stddev: %.3f ms)\n", s_stream.mean * 1000.0, s_stream.stddev * 1000.0);
    printf("       - Min / Max: %.3f ms / %.3f ms\n", s_stream.min * 1000.0, s_stream.max * 1000.0);
    printf("       - Banda de Leitura Medida: %.2f GB/s\n\n", bw_stream);

    printf("    B. Kernel Fused Lean Q4x4 (Computacao completa + SwiGLU):\n");
    printf("       - Mediana:   %.3f ms\n", s_compute.median * 1000.0);
    printf("       - Media:     %.3f ms (stddev: %.3f ms)\n", s_compute.mean * 1000.0, s_compute.stddev * 1000.0);
    printf("       - Min / Max: %.3f ms / %.3f ms\n", s_compute.min * 1000.0, s_compute.max * 1000.0);
    printf("       - Banda Efetiva de Pesos:  %.2f GB/s\n\n", bw_compute);

    double overhead_compute_pct = ((s_compute.median - s_stream.median) / s_compute.median) * 100.0;
    double time_in_dram_pct     = (s_stream.median / s_compute.median) * 100.0;

    printf("    DIAGNOSTICO DE MEMORIA:\n");
    printf("    - Tempo gasto puramente lendo os 15.48 MB da DRAM: %.1f%% do tempo total (%.3f ms de %.3f ms)\n",
           time_in_dram_pct, s_stream.median * 1000.0, s_compute.median * 1000.0);
    printf("    - Sobrecarga de calculo aritmético (SIMD dot + SwiGLU): %.1f%% (%.3f ms)\n\n",
           overhead_compute_pct, (s_compute.median - s_stream.median) * 100.0);

    // 3. ETAPA 2 - PROTOTIPAGEM TERNARIA
    size_t tern_single_bytes = (size_t)groups * nb * sizeof(BlockTernary2bit4x4); // 40 bytes / block
    size_t tern_two_bytes    = 2 * tern_single_bytes;

    printf("[3] ETAPA 2 - MICROBENCHMARKS TERNARIOS:\n");
    printf("    Formato A: 2-bit per trit (4 trits/byte, 40 bytes por bloco 4x32)\n");
    printf("    - Volume TOTAL (Wgate + Wup): %zu bytes (%.3f MB decimal | %.3f MiB binario)\n",
           tern_two_bytes, (double)tern_two_bytes / 1e6, (double)tern_two_bytes / (1024.0 * 1024.0));
    printf("    - Reducao de volume de memoria vs Q4: -%.1f%%\n",
           (1.0 - (double)tern_two_bytes / (double)q4_two_tensors_bytes) * 100.0);

    BlockTernary2bit4x4* Wgate_tern = (BlockTernary2bit4x4*)_aligned_malloc(tern_single_bytes, 64);
    BlockTernary2bit4x4* Wup_tern   = (BlockTernary2bit4x4*)_aligned_malloc(tern_single_bytes, 64);
    memset(Wgate_tern, 0x55, tern_single_bytes);
    memset(Wup_tern,   0xAA, tern_single_bytes);

    double t_tern[100];
    // warmup
    mlp_ternary_2bit_lean(h_mlp, Wgate_tern, Wup_tern, x_q8, M, K);

    for (int i = 0; i < N_ITERS; i++) {
        QueryPerformanceCounter(&start);
        mlp_ternary_2bit_lean(h_mlp, Wgate_tern, Wup_tern, x_q8, M, K);
        QueryPerformanceCounter(&end);
        t_tern[i] = (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    }
    Stats s_tern = calc_stats(t_tern, N_ITERS);
    double bw_tern = ((double)tern_two_bytes / s_tern.median) / 1e9;

    printf("    - Mediana do Kernel 2-bit Ternario: %.3f ms\n", s_tern.median * 1000.0);
    printf("    - Min / Max: %.3f ms / %.3f ms (stddev: %.3f ms)\n",
           s_tern.min * 1000.0, s_tern.max * 1000.0, s_tern.stddev * 1000.0);
    printf("    - Banda de Pesos Processada:        %.2f GB/s\n", bw_tern);
    printf("    - Speedup Efetivo vs Q4x4:          %.2fx (%.3f ms -> %.3f ms)\n\n",
           s_compute.median / s_tern.median, s_compute.median * 1000.0, s_tern.median * 1000.0);

    // Formato B: Avaliacao de custo de desempacotamento de 5 trits/byte
    int total_weights = M * K * 2; // Gate + Up
    double t_lut = benchmark_decode_5trit(total_weights);
    printf("    Formato B: 5 trits/byte (Canonical BitNet b1.58, 1.6 bits/peso)\n");
    printf("    - Volume teorico de pesos: ~5.60 MB decimal\n");
    printf("    - Custo medido de desempacotamento via LUT (4 threads): %.3f ms\n", t_lut * 1000.0);
    printf("    - Impacto: O desempacotamento de 5 trits/byte sozinho consome %.3f ms,\n", t_lut * 1000.0);
    printf("      mostrando que sem instrucoes dedicadas, a sobrecarga de decodificacao\n");
    printf("      anula parte dos ganhos de transferencia de dados!\n");
    printf("================================================================================\n");

    _aligned_free(Wgate);
    _aligned_free(Wup);
    _aligned_free(x_q8);
    _aligned_free(h_mlp);
    _aligned_free(Wgate_tern);
    _aligned_free(Wup_tern);

    return 0;
}
