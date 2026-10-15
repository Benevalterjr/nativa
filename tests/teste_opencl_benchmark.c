// ============================================================================
// TESTE E BENCHMARK PRÁTICO NATIVA OPENCL (CPU & GPU)
// ============================================================================
// Demonstração e validação do backend OpenCL da Linguagem Nativa:
//   1. Carrega OpenCL.dll sem exigir SDK
//   2. Enumera os dispositivos (CPU i7-3770 e iGPU HD Graphics 4000)
//   3. Compila um kernel JIT de tensores (1 Milhão de elementos)
//   4. Executa no Dispositivo 0 (CPU) e valida resultado numérico
//   5. Executa no Dispositivo 1 (GPU) e valida resultado numérico
//   6. Compara tempos de latência e throughput
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "../tools/nativa_opencl.h"

// Kernel OpenCL de operação tensorial fundida (Mul-Add + ReLU): Y = relu(A * B + 0.5f)
static const char* TENSOR_KERNEL_SOURCE = 
"__kernel void tensor_fused_mul_add_relu(\n"
"    __global const float* A,\n"
"    __global const float* B,\n"
"    __global float* C,\n"
"    int N\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i < N) {\n"
"        float val = A[i] * B[i] + 0.5f;\n"
"        C[i] = (val > 0.0f) ? val : 0.0f;\n"
"    }\n"
"}\n";

static double get_time_sec(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static bool init = false;
    if (!init) { QueryPerformanceFrequency(&freq); init = true; }
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static bool run_tensor_test_on_device(NativaCLContext* ctx, int dev_idx, int N) {
    const NativaCLDevice* dev = &ctx->devices[dev_idx];
    printf("\n>>> Testando no Dispositivo [%d]: [%s] %s <<<\n",
           dev_idx, dev->is_gpu ? "GPU" : "CPU", dev->name);

    if (!nativa_cl_activate_device(ctx, dev_idx)) {
        fprintf(stderr, "[ERRO] Falha ao criar contexto/fila no dispositivo %d\n", dev_idx);
        return false;
    }

    // 1. Compila o Kernel JIT para este dispositivo
    char err_buf[2048] = {0};
    double t_build_start = get_time_sec();
    cl_program prog = nativa_cl_build_program_from_source(ctx, TENSOR_KERNEL_SOURCE, err_buf, sizeof(err_buf));
    double t_build = get_time_sec() - t_build_start;

    if (!prog) {
        fprintf(stderr, "[ERRO] Falha ao compilar kernel OpenCL:\n%s\n", err_buf);
        return false;
    }
    printf("    -> Kernel OpenCL compilado via JIT em %.2f ms\n", t_build * 1000.0);

    cl_int err = 0;
    cl_kernel kernel = ctx->clCreateKernel(prog, "tensor_fused_mul_add_relu", &err);
    if (err != CL_SUCCESS || !kernel) {
        fprintf(stderr, "[ERRO] clCreateKernel falhou com codigo %d\n", err);
        ctx->clReleaseProgram(prog);
        return false;
    }

    // 2. Prepara dados na memória do Host
    size_t bytes = (size_t)N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        free(h_A); free(h_B); free(h_C);
        ctx->clReleaseKernel(kernel);
        ctx->clReleaseProgram(prog);
        return false;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 100) * 0.05f;
        h_B[i] = (float)((i + 3) % 100) * 0.02f - 0.5f;
        h_C[i] = 0.0f;
    }

    // 3. Aloca buffers de memória OpenCL
    cl_mem d_A = ctx->clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A, &err);
    cl_mem d_B = ctx->clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B, &err);
    cl_mem d_C = ctx->clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    if (!d_A || !d_B || !d_C) {
        fprintf(stderr, "[ERRO] Falha na alocacao de buffers cl_mem\n");
        free(h_A); free(h_B); free(h_C);
        if (d_A) ctx->clReleaseMemObject(d_A);
        if (d_B) ctx->clReleaseMemObject(d_B);
        if (d_C) ctx->clReleaseMemObject(d_C);
        ctx->clReleaseKernel(kernel);
        ctx->clReleaseProgram(prog);
        return false;
    }

    // 4. Configura argumentos do Kernel
    ctx->clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    ctx->clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    ctx->clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    ctx->clSetKernelArg(kernel, 3, sizeof(int), &N);

    // 5. Executa o Kernel e mede o tempo de computação
    size_t global_size = ((size_t)N + 255) / 256 * 256; // Arredonda para múltiplo do workgroup
    size_t local_size = 256;
    if (local_size > dev->max_workgroup_size) local_size = dev->max_workgroup_size;

    ctx->clFinish(ctx->queue);
    double t_exec_start = get_time_sec();

    ctx->clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    ctx->clFinish(ctx->queue);

    double t_exec = get_time_sec() - t_exec_start;

    // 6. Lê o resultado de volta para o Host
    ctx->clEnqueueReadBuffer(ctx->queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL);

    // 7. Validação numérica rigorosa
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] * h_B[i] + 0.5f;
        if (expected < 0.0f) expected = 0.0f;
        if (fabsf(h_C[i] - expected) > 1e-4f) {
            errors++;
            if (errors <= 3) {
                printf("    [FALHA] Indice %d: obtido=%f esperado=%f\n", i, h_C[i], expected);
            }
        }
    }

    if (errors == 0) {
        double mflops = ((double)N * 3.0) / (t_exec * 1e6);
        printf("    [SUCESSO] %d elementos computados com 100%% de precisao numerica!\n", N);
        printf("    -> Tempo de execucao do Kernel: %.3f ms (%.1f MFLOPS)\n", t_exec * 1000.0, mflops);
    } else {
        printf("    [ERRO] Foram encontrados %d erros numericos!\n", errors);
    }

    // Limpeza
    free(h_A); free(h_B); free(h_C);
    ctx->clReleaseMemObject(d_A);
    ctx->clReleaseMemObject(d_B);
    ctx->clReleaseMemObject(d_C);
    ctx->clReleaseKernel(kernel);
    ctx->clReleaseProgram(prog);

    return (errors == 0);
}

int main(void) {
    printf("====================================================================\n");
    printf("  NATIVA - TESTE E VALIDAÇÃO PRÁTICA DO BACKEND OPENCL\n");
    printf("====================================================================\n");

    NativaCLContext ctx;
    if (!nativa_cl_init(&ctx)) {
        printf("[INFO] OpenCL nao esta disponivel neste ambiente ou OpenCL.dll nao encontrada.\n");
        return 0;
    }

    nativa_cl_print_devices(&ctx);

    const int N = 1000000; // 1 Milhão de elementos
    printf("\nExecutando calculo tensorial em lote (%d tensores float32)...\n", N);

    bool all_ok = true;
    for (int i = 0; i < ctx.device_count; i++) {
        if (!run_tensor_test_on_device(&ctx, i, N)) {
            all_ok = false;
        }
    }

    printf("\n====================================================================\n");
    if (all_ok) {
        printf("  STATUS FINAL: TODOS OS DISPOSITIVOS OPENCL FORAM VALIDADOS COM SUCESSO!\n");
    } else {
        printf("  STATUS FINAL: ALGUNS TESTES APRESENTARAM FALHAS.\n");
    }
    printf("====================================================================\n");

    nativa_cl_release(&ctx);
    return all_ok ? 0 : 1;
}
