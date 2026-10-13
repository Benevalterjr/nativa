#ifndef CPU_DISPATCH_H
#define CPU_DISPATCH_H

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <intrin.h>
#include <windows.h>
#include <omp.h>

typedef struct {
    char vendor[13];
    char brand[49];
    bool ssse3;
    bool sse41;
    bool sse42;
    bool avx;
    bool fma3;
    bool avx2;
    bool avx512f;
    bool avx512vnni;
    int  logical_threads;
    int  physical_cores;
    int  recommended_omp_threads;
} CPUInfo;

static inline void cpu_detect(CPUInfo* cpu) {
    memset(cpu, 0, sizeof(CPUInfo));

    int info[4];
    __cpuid(info, 0);
    int max_ids = info[0];

    *(int*)&cpu->vendor[0] = info[1];
    *(int*)&cpu->vendor[4] = info[3];
    *(int*)&cpu->vendor[8] = info[2];
    cpu->vendor[12] = '\0';

    if (max_ids >= 1) {
        __cpuid(info, 1);
        cpu->ssse3 = (info[2] & (1 << 9)) != 0;
        cpu->fma3  = (info[2] & (1 << 12)) != 0;
        cpu->sse41 = (info[2] & (1 << 19)) != 0;
        cpu->sse42 = (info[2] & (1 << 20)) != 0;
        cpu->avx   = (info[2] & (1 << 28)) != 0;
    }

    if (max_ids >= 7) {
        __cpuidex(info, 7, 0);
        cpu->avx2       = (info[1] & (1 << 5)) != 0;
        cpu->avx512f    = (info[1] & (1 << 16)) != 0;
        cpu->avx512vnni = (info[2] & (1 << 11)) != 0;
    }

    int ext[4];
    __cpuid(ext, 0x80000000);
    if ((unsigned int)ext[0] >= 0x80000004) {
        __cpuid((int*)&cpu->brand[0],  0x80000002);
        __cpuid((int*)&cpu->brand[16], 0x80000003);
        __cpuid((int*)&cpu->brand[32], 0x80000004);
        cpu->brand[48] = '\0';
    } else {
        strcpy(cpu->brand, "CPU x86_64 Generica");
    }

    SYSTEM_INFO sys;
    GetSystemInfo(&sys);
    cpu->logical_threads = (int)sys.dwNumberOfProcessors;
    // Em CPUs dual-core com HyperThreading, 4 threads costumam dar o melhor pico de barramento
    cpu->physical_cores = (cpu->logical_threads > 1) ? (cpu->logical_threads / 2) : 1;
    cpu->recommended_omp_threads = cpu->logical_threads;
    if (cpu->recommended_omp_threads > 8) cpu->recommended_omp_threads = 8; // saturacao de barramento DDR
}

static inline void cpu_print_info(const CPUInfo* cpu) {
    printf("====================================================================\n");
    printf("  NATIVA CPU - DETECCAO E CONFIGURACAO OPENMP\n");
    printf("====================================================================\n");
    printf("  Processador : %s\n", cpu->brand);
    printf("  Fabricante  : %s | Threads: %d (Cores Fisicos est.: %d)\n",
           cpu->vendor, cpu->logical_threads, cpu->physical_cores);
    printf("  Recursos    : SSSE3=%s | SSE4.1=%s | AVX=%s | AVX2=%s | FMA3=%s | AVX512=%s\n",
           cpu->ssse3 ? "SIM" : "NAO",
           cpu->sse41 ? "SIM" : "NAO",
           cpu->avx   ? "SIM" : "NAO",
           cpu->avx2  ? "SIM" : "NAO",
           cpu->fma3  ? "SIM" : "NAO",
           cpu->avx512f ? "SIM" : "NAO");
    printf("  OpenMP      : recomendação de até %d threads; OMP_NUM_THREADS tem precedência\n", cpu->recommended_omp_threads);
    printf("====================================================================\n");
}

static inline void cpu_apply_auto_tune(const CPUInfo* cpu) {
    // Defina a policy antes da primeira chamada OpenMP, pois o runtime pode
    // ler OMP_WAIT_POLICY durante sua inicializacao.
    DWORD wait_len = GetEnvironmentVariableA("OMP_WAIT_POLICY", NULL, 0);
    if (!wait_len) SetEnvironmentVariableA("OMP_WAIT_POLICY", "PASSIVE");

    // Respeite OMP_NUM_THREADS quando o usuario ou o chamador o definiu.
    // omp_get_max_threads() incorpora esse valor sem precisar reinterpretar
    // variantes validas da sintaxe da OpenMP (por exemplo, listas por nivel).
    DWORD env_len = GetEnvironmentVariableA("OMP_NUM_THREADS", NULL, 0);
    int threads = env_len ? omp_get_max_threads() : cpu->recommended_omp_threads;
    if (threads < 1) threads = cpu->recommended_omp_threads;
    omp_set_num_threads(threads);
}

#endif // CPU_DISPATCH_H
