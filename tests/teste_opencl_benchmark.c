// ============================================================================
// PRACTICAL NATIVA OPENCL TEST AND BENCHMARK (CPU & GPU)
// ============================================================================
// Demonstration and validation of the Nativa Language OpenCL backend:
//   1. Dynamically loads OpenCL.dll without requiring an SDK
//   2. Enumerates devices (CPU i7-3770 and iGPU HD Graphics 4000)
//   3. Compiles a JIT tensor kernel (1 Million elements)
//   4. Executes on Device 0 (CPU) and validates numerical result
//   5. Executes on Device 1 (GPU) and validates numerical result
//   6. Compares latency and throughput
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "../tools/nativa_opencl.h"

// Fused tensor operation OpenCL kernel (Mul-Add + ReLU): Y = relu(A * B + 0.5f)
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

static void cleanup_host_buffers(float* a, float* b, float* c) {
    if (a) free(a);
    if (b) free(b);
    if (c) free(c);
}

static void cleanup_cl_resources(const NativaCLContext* ctx, cl_mem da, cl_mem db, cl_mem dc, cl_kernel k, cl_program p) {
    if (!ctx) return;
    if (da) ctx->clReleaseMemObject(da);
    if (db) ctx->clReleaseMemObject(db);
    if (dc) ctx->clReleaseMemObject(dc);
    if (k) ctx->clReleaseKernel(k);
    if (p) ctx->clReleaseProgram(p);
}

static int validate_results(const float* h_A, const float* h_B, const float* h_C, int N) {
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] * h_B[i] + 0.5f;
        if (expected < 0.0f) expected = 0.0f;
        if (fabsf(h_C[i] - expected) > 1e-4f) {
            errors++;
            if (errors <= 3) {
                printf("    [FAIL] Index %d: got=%f expected=%f\n", i, h_C[i], expected);
            }
        }
    }
    return errors;
}

static bool run_tensor_test_on_device(NativaCLContext* ctx, int dev_idx, int N) {
    if (!ctx || dev_idx < 0 || dev_idx >= ctx->device_count || N < 2) {
        return false;
    }
    const NativaCLDevice* dev = &ctx->devices[dev_idx];
    printf("\n>>> Testing on Device [%d]: [%s] %s <<<\n",
           dev_idx, dev->is_gpu ? "GPU" : "CPU", dev->name);

    if (!nativa_cl_activate_device(ctx, dev_idx)) {
        fprintf(stderr, "[ERROR] Failed to create context/queue on device %d\n", dev_idx);
        return false;
    }

    // 1. Compile JIT Kernel for this device
    char err_buf[2048] = {0};
    double t_build_start = get_time_sec();
    cl_program prog = nativa_cl_build_program_from_source(ctx, TENSOR_KERNEL_SOURCE, err_buf, sizeof(err_buf));
    double t_build = get_time_sec() - t_build_start;

    if (!prog) {
        fprintf(stderr, "[ERROR] Failed to compile OpenCL kernel:\n%s\n", err_buf);
        return false;
    }
    printf("    -> OpenCL kernel JIT-compiled in %.2f ms\n", t_build * 1000.0);

    cl_int err = 0;
    cl_kernel kernel = ctx->clCreateKernel(prog, "tensor_fused_mul_add_relu", &err);
    if (err != CL_SUCCESS || !kernel) {
        fprintf(stderr, "[ERROR] clCreateKernel failed with code %d\n", err);
        ctx->clReleaseProgram(prog);
        return false;
    }

    // 2. Prepare data in Host memory
    size_t bytes = (size_t)N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        cleanup_cl_resources(ctx, h_A, h_B, h_C, NULL, NULL, NULL, kernel, prog);
        return false;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 100) * 0.05f;
        h_B[i] = (float)((i + 3) % 100) * 0.02f - 0.5f;
        h_C[i] = 0.0f;
    }

    // 3. Allocate OpenCL memory buffers
    cl_mem d_A = ctx->clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A, &err);
    cl_mem d_B = ctx->clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B, &err);
    cl_mem d_C = ctx->clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    if (!d_A || !d_B || !d_C) {
        fprintf(stderr, "[ERROR] Failed to allocate cl_mem buffers\n");
        cleanup_host_buffers(h_A, h_B, h_C);
        cleanup_cl_resources(ctx, d_A, d_B, d_C, kernel, prog);
        return false;
    }

    // 4. Configure Kernel arguments
    ctx->clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    ctx->clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    ctx->clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    ctx->clSetKernelArg(kernel, 3, sizeof(int), &N);

    // 5. Execute Kernel and measure compute time
    size_t global_size = ((size_t)N + 255) / 256 * 256;
    size_t local_size = 256;
    if (local_size > dev->max_workgroup_size) local_size = dev->max_workgroup_size;

    ctx->clFinish(ctx->queue);
    double t_exec_start = get_time_sec();

    ctx->clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    ctx->clFinish(ctx->queue);

    double t_exec = get_time_sec() - t_exec_start;

    // 6. Read result back to Host
    ctx->clEnqueueReadBuffer(ctx->queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL);

    // 7. Rigorous numerical validation
    int errors = validate_results(h_A, h_B, h_C, N);

    if (errors == 0) {
        double mflops = ((double)N * 3.0) / (t_exec * 1e6);
        printf("    [SUCCESS] %d elements computed with 100%% numerical accuracy!\n", N);
        printf("    -> Kernel execution time: %.3f ms (%.1f MFLOPS)\n", t_exec * 1000.0, mflops);
    } else {
        printf("    [ERROR] Found %d numerical errors!\n", errors);
    }

    cleanup_host_buffers(h_A, h_B, h_C);
    cleanup_cl_resources(ctx, d_A, d_B, d_C, kernel, prog);
    return (errors == 0);
}

int main(void) {
    printf("====================================================================\n");
    printf("  NATIVA - PRACTICAL TEST AND VALIDATION OF THE OPENCL BACKEND\n");
    printf("====================================================================\n");

    NativaCLContext ctx;
    if (!nativa_cl_init(&ctx)) {
        printf("[INFO] OpenCL is not available in this environment or OpenCL.dll was not found.\n");
        return 0;
    }

    nativa_cl_print_devices(&ctx);

    const int N = 1000000; // 1 Million elements
    printf("\nRunning batch tensor computation (%d float32 tensors)...\n", N);

    bool all_ok = true;
    for (int i = 0; i < ctx.device_count; i++) {
        if (!run_tensor_test_on_device(&ctx, i, N)) {
            all_ok = false;
        }
    }

    printf("\n====================================================================\n");
    if (all_ok) {
        printf("  FINAL STATUS: ALL OPENCL DEVICES HAVE BEEN VALIDATED SUCCESSFULLY!\n");
    } else {
        printf("  FINAL STATUS: SOME TESTS FAILED.\n");
    }
    printf("====================================================================\n");

    nativa_cl_release(&ctx);
    return all_ok ? 0 : 1;
}
