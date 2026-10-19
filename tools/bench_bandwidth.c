// bench_bandwidth.c - Measures real RAM READ bandwidth
// (the same pattern as decode: sweep weights once, sequentially).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <emmintrin.h>
#include <omp.h>
#include <windows.h>

#define BUF_MB 512
#define REPS 7

static double now_s(void) {
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static int cmp_d(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

// Pure reading: 4 independent accumulators, minimal CPU overhead
static uint64_t read_pass(const __m128i* buf, size_t n_vec) {
    __m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0;
    long long i;
    #pragma omp parallel
    {
        __m128i l0 = _mm_setzero_si128(), l1 = l0, l2 = l0, l3 = l0;
        #pragma omp for schedule(static) nowait
        for (i = 0; i < (long long)n_vec; i += 4) {
            l0 = _mm_xor_si128(l0, _mm_load_si128(buf + i));
            l1 = _mm_xor_si128(l1, _mm_load_si128(buf + i + 1));
            l2 = _mm_xor_si128(l2, _mm_load_si128(buf + i + 2));
            l3 = _mm_xor_si128(l3, _mm_load_si128(buf + i + 3));
        }
        #pragma omp critical
        {
            a0 = _mm_xor_si128(a0, l0); a1 = _mm_xor_si128(a1, l1);
            a2 = _mm_xor_si128(a2, l2); a3 = _mm_xor_si128(a3, l3);
        }
    }
    __m128i r = _mm_xor_si128(_mm_xor_si128(a0, a1), _mm_xor_si128(a2, a3));
    return (uint64_t)_mm_cvtsi128_si32(r);
}

int main(void) {
    size_t bytes = (size_t)BUF_MB * 1024 * 1024;
    __m128i* buf = (__m128i*)_aligned_malloc(bytes, 64);
    if (!buf) { printf("Failed to allocate %d MB\n", BUF_MB); return 1; }
    memset(buf, 1, bytes);   // force commit of the pages
    size_t n_vec = bytes / 16;

    printf("Buffer: %d MB | %d repetitions per configuration\n\n", BUF_MB, REPS);
    printf("Threads |   Best (GB/s)  | Median (GB/s)  | Time for 828 MB (median)\n");
    printf("--------+----------------+----------------+--------------------------\n");

    int cfgs[3] = {1, 2, 4};
    uint64_t sink = 0;
    for (int c = 0; c < 3; c++) {
        omp_set_num_threads(cfgs[c]);
        sink ^= read_pass(buf, n_vec);  // warmup
        double gbs[REPS];
        for (int r = 0; r < REPS; r++) {
            double t0 = now_s();
            sink ^= read_pass(buf, n_vec);
            double dt = now_s() - t0;
            gbs[r] = (bytes / 1e9) / dt;
        }
        qsort(gbs, REPS, sizeof(double), cmp_d);
        double best = gbs[REPS - 1], med = gbs[REPS / 2];
        double ms_828 = (828.31 * 1024 * 1024 / 1e9) / med * 1000.0;
        printf("   %d    |     %6.2f     |     %6.2f     |   %6.1f ms  (ceiling %.1f tok/s)\n",
               cfgs[c], best, med, ms_828, 1000.0 / ms_828);
    }
    printf("\n(sink=%llu)\n", (unsigned long long)sink);
    _aligned_free(buf);
    return 0;
}
