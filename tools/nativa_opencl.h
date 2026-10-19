// ============================================================================
// NATIVA OPENCL RUNTIME BRIDGE (ZERO-DEPENDENCY DYNAMIC LOADER)
// ============================================================================
// Provides universal support for OpenCL accelerators (GPU and CPU) without
// requiring an external OpenCL SDK installation. Dynamically loads OpenCL.dll
// (Windows) or libOpenCL.so (Linux) at runtime.
// ============================================================================

#ifndef NATIVA_OPENCL_H
#define NATIVA_OPENCL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE nat_cl_lib_t;
#define NAT_CL_LOAD_LIB(name) LoadLibraryA(name)
#define NAT_CL_GET_SYM(lib, sym) GetProcAddress((HMODULE)(lib), sym)
#define NAT_CL_FREE_LIB(lib) FreeLibrary((HMODULE)(lib))
#define NAT_CL_DEFAULT_LIB "OpenCL.dll"
#else
#include <dlfcn.h>
typedef void* nat_cl_lib_t;
#define NAT_CL_LOAD_LIB(name) dlopen(name, RTLD_LAZY)
#define NAT_CL_GET_SYM(lib, sym) dlsym(lib, sym)
#define NAT_CL_FREE_LIB(lib) dlclose(lib)
#define NAT_CL_DEFAULT_LIB "libOpenCL.so"
#endif

// ----------------------------------------------------------------------------
// OPENCL 1.2 TYPES AND CONSTANTS
// ----------------------------------------------------------------------------
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef void*    cl_platform_id;
typedef void*    cl_device_id;
typedef void*    cl_context;
typedef void*    cl_command_queue;
typedef void*    cl_mem;
typedef void*    cl_program;
typedef void*    cl_kernel;
typedef void*    cl_event;

#define CL_SUCCESS                     0
#define CL_DEVICE_TYPE_CPU             (1ULL << 1)
#define CL_DEVICE_TYPE_GPU             (1ULL << 2)
#define CL_DEVICE_TYPE_ALL             0xFFFFFFFFULL

#define CL_PLATFORM_NAME               0x0902
#define CL_DEVICE_TYPE                 0x1000
#define CL_DEVICE_NAME                 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS    0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE  0x1004
#define CL_DEVICE_GLOBAL_MEM_SIZE      0x101F

#define CL_MEM_READ_WRITE              (1ULL << 0)
#define CL_MEM_WRITE_ONLY              (1ULL << 1)
#define CL_MEM_READ_ONLY               (1ULL << 2)
#define CL_MEM_USE_HOST_PTR            (1ULL << 3)
#define CL_MEM_ALLOC_HOST_PTR          (1ULL << 4)
#define CL_MEM_COPY_HOST_PTR           (1ULL << 5)

#define CL_TRUE                        1
#define CL_FALSE                       0
#define CL_PROGRAM_BUILD_LOG           0x1183

// ----------------------------------------------------------------------------
// OPENCL FUNCTION SIGNATURES
// ----------------------------------------------------------------------------
typedef cl_int (*fn_clGetPlatformIDs)(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
typedef cl_int (*fn_clGetPlatformInfo)(cl_platform_id platform, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id platform, cl_ulong device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id device, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_context (*fn_clCreateContext)(const intptr_t* properties, cl_uint num_devices, const cl_device_id* devices, void* pfn_notify, void* user_data, cl_int* errcode_ret);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context context, cl_device_id device, cl_ulong properties, cl_int* errcode_ret);
typedef cl_program (*fn_clCreateProgramWithSource)(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret);
typedef cl_int (*fn_clBuildProgram)(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void* pfn_notify, void* user_data);
typedef cl_int (*fn_clGetProgramBuildInfo)(cl_program program, cl_device_id device, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_kernel (*fn_clCreateKernel)(cl_program program, const char* kernel_name, cl_int* errcode_ret);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
typedef cl_mem (*fn_clCreateBuffer)(cl_context context, cl_ulong flags, size_t size, void* host_ptr, cl_int* errcode_ret);
typedef cl_int (*fn_clEnqueueWriteBuffer)(cl_command_queue command_queue, cl_mem buffer, cl_uint blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (*fn_clEnqueueReadBuffer)(cl_command_queue command_queue, cl_mem buffer, cl_uint blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (*fn_clFinish)(cl_command_queue command_queue);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem memobj);
typedef cl_int (*fn_clReleaseKernel)(cl_kernel kernel);
typedef cl_int (*fn_clReleaseProgram)(cl_program program);
typedef cl_int (*fn_clReleaseCommandQueue)(cl_command_queue command_queue);
typedef cl_int (*fn_clReleaseContext)(cl_context context);

// ----------------------------------------------------------------------------
// NATIVA OPENCL DEVICE AND CONTEXT STRUCTURE
// ----------------------------------------------------------------------------
typedef struct {
    cl_platform_id platform;
    cl_device_id   device;
    char           platform_name[128];
    char           name[128];
    bool           is_gpu;
    cl_uint        compute_units;
    size_t         max_workgroup_size;
    cl_ulong       global_mem_bytes;
} NativaCLDevice;

typedef struct {
    nat_cl_lib_t lib;
    bool available;
    int  device_count;
    NativaCLDevice devices[16];

    // Active Context
    int              active_device_idx;
    cl_context       context;
    cl_command_queue queue;

    // Dynamic Function Table
    fn_clGetPlatformIDs clGetPlatformIDs;
    fn_clGetPlatformInfo clGetPlatformInfo;
    fn_clGetDeviceIDs clGetDeviceIDs;
    fn_clGetDeviceInfo clGetDeviceInfo;
    fn_clCreateContext clCreateContext;
    fn_clCreateCommandQueue clCreateCommandQueue;
    fn_clCreateProgramWithSource clCreateProgramWithSource;
    fn_clBuildProgram clBuildProgram;
    fn_clGetProgramBuildInfo clGetProgramBuildInfo;
    fn_clCreateKernel clCreateKernel;
    fn_clSetKernelArg clSetKernelArg;
    fn_clCreateBuffer clCreateBuffer;
    fn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
    fn_clEnqueueReadBuffer clEnqueueReadBuffer;
    fn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
    fn_clFinish clFinish;
    fn_clReleaseMemObject clReleaseMemObject;
    fn_clReleaseKernel clReleaseKernel;
    fn_clReleaseProgram clReleaseProgram;
    fn_clReleaseCommandQueue clReleaseCommandQueue;
    fn_clReleaseContext clReleaseContext;
} NativaCLContext;

// ----------------------------------------------------------------------------
// DYNAMIC INITIALIZATION
// ----------------------------------------------------------------------------
static inline bool nativa_cl_init(NativaCLContext* ctx) {
    if (!ctx) return false;
    memset(ctx, 0, sizeof(NativaCLContext));
    ctx->active_device_idx = -1;

    ctx->lib = NAT_CL_LOAD_LIB(NAT_CL_DEFAULT_LIB);
    if (!ctx->lib) {
        return false;
    }

    #define BIND_CL(name) do { \
        ctx->name = (fn_##name)NAT_CL_GET_SYM(ctx->lib, #name); \
        if (!ctx->name) { NAT_CL_FREE_LIB(ctx->lib); ctx->lib = NULL; return false; } \
    } while(0)

    BIND_CL(clGetPlatformIDs);
    BIND_CL(clGetPlatformInfo);
    BIND_CL(clGetDeviceIDs);
    BIND_CL(clGetDeviceInfo);
    BIND_CL(clCreateContext);
    BIND_CL(clCreateCommandQueue);
    BIND_CL(clCreateProgramWithSource);
    BIND_CL(clBuildProgram);
    BIND_CL(clGetProgramBuildInfo);
    BIND_CL(clCreateKernel);
    BIND_CL(clSetKernelArg);
    BIND_CL(clCreateBuffer);
    BIND_CL(clEnqueueWriteBuffer);
    BIND_CL(clEnqueueReadBuffer);
    BIND_CL(clEnqueueNDRangeKernel);
    BIND_CL(clFinish);
    BIND_CL(clReleaseMemObject);
    BIND_CL(clReleaseKernel);
    BIND_CL(clReleaseProgram);
    BIND_CL(clReleaseCommandQueue);
    BIND_CL(clReleaseContext);

    #undef BIND_CL

    // Discover Platforms and Devices
    cl_uint num_plat = 0;
    if (ctx->clGetPlatformIDs(0, NULL, &num_plat) != CL_SUCCESS || num_plat == 0) {
        return false;
    }

    cl_platform_id* plats = (cl_platform_id*)malloc(num_plat * sizeof(cl_platform_id));
    if (!plats) return false;
    ctx->clGetPlatformIDs(num_plat, plats, NULL);

    ctx->device_count = 0;
    for (cl_uint p = 0; p < num_plat && ctx->device_count < 16; p++) {
        char p_name[128] = {0};
        ctx->clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(p_name), p_name, NULL);

        cl_uint num_dev = 0;
        if (ctx->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_dev) == CL_SUCCESS && num_dev > 0) {
            cl_device_id* devs = (cl_device_id*)malloc(num_dev * sizeof(cl_device_id));
            if (devs) {
                if (ctx->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, num_dev, devs, NULL) == CL_SUCCESS) {
                    for (cl_uint d = 0; d < num_dev && ctx->device_count < 16; d++) {
                        NativaCLDevice* nd = &ctx->devices[ctx->device_count];
                        nd->platform = plats[p];
                        nd->device = devs[d];
                        strncpy(nd->platform_name, p_name, sizeof(nd->platform_name) - 1);

                        ctx->clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(nd->name), nd->name, NULL);
                        
                        cl_ulong dev_type = 0;
                        ctx->clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
                        nd->is_gpu = (dev_type & CL_DEVICE_TYPE_GPU) != 0;

                        ctx->clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(nd->compute_units), &nd->compute_units, NULL);
                        ctx->clGetDeviceInfo(devs[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(nd->max_workgroup_size), &nd->max_workgroup_size, NULL);
                        ctx->clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(nd->global_mem_bytes), &nd->global_mem_bytes, NULL);

                        ctx->device_count++;
                    }
                }
                free(devs);
            }
        }
    }
    free(plats);

    ctx->available = (ctx->device_count > 0);
    return ctx->available;
}

static inline void nativa_cl_print_devices(const NativaCLContext* ctx) {
    printf("====================================================================\n");
    printf("  NATIVA OPENCL - DETECTED DEVICES (%d available)\n", ctx ? ctx->device_count : 0);
    printf("====================================================================\n");
    if (!ctx || ctx->device_count == 0) {
        printf("  No compatible OpenCL device found.\n");
        printf("====================================================================\n");
        return;
    }
    for (int i = 0; i < ctx->device_count; i++) {
        const NativaCLDevice* d = &ctx->devices[i];
        double mem_mb = (double)d->global_mem_bytes / (1024.0 * 1024.0);
        printf("  [%d] [%s] %s\n", i, d->is_gpu ? "GPU" : "CPU", d->name);
        printf("      Platform: %s | Compute Units: %u | VRAM/RAM: %.1f MB | Max WorkGroup: %zu\n",
               d->platform_name, d->compute_units, mem_mb, d->max_workgroup_size);
    }
    printf("====================================================================\n");
}

static inline bool nativa_cl_activate_device(NativaCLContext* ctx, int device_idx) {
    if (!ctx || !ctx->available || device_idx < 0 || device_idx >= ctx->device_count) {
        return false;
    }

    // Release previous context if present
    if (ctx->queue) { ctx->clReleaseCommandQueue(ctx->queue); ctx->queue = NULL; }
    if (ctx->context) { ctx->clReleaseContext(ctx->context); ctx->context = NULL; }

    cl_device_id dev = ctx->devices[device_idx].device;
    cl_int err = 0;
    ctx->context = ctx->clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || !ctx->context) {
        return false;
    }

    ctx->queue = ctx->clCreateCommandQueue(ctx->context, dev, 0, &err);
    if (err != CL_SUCCESS || !ctx->queue) {
        ctx->clReleaseContext(ctx->context);
        ctx->context = NULL;
        return false;
    }

    ctx->active_device_idx = device_idx;
    return true;
}

static inline cl_program nativa_cl_build_program_from_source(NativaCLContext* ctx, const char* source, char* err_log_buf, size_t err_log_size) {
    if (!ctx || !ctx->context || !source) return NULL;

    cl_int err = 0;
    cl_program prog = ctx->clCreateProgramWithSource(ctx->context, 1, &source, NULL, &err);
    if (err != CL_SUCCESS || !prog) return NULL;

    cl_device_id dev = ctx->devices[ctx->active_device_idx].device;
    err = ctx->clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        if (err_log_buf && err_log_size > 0) {
            ctx->clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, err_log_size, err_log_buf, NULL);
        }
        ctx->clReleaseProgram(prog);
        return NULL;
    }
    return prog;
}

static inline void nativa_cl_release(NativaCLContext* ctx) {
    if (!ctx) return;
    if (ctx->queue) { ctx->clReleaseCommandQueue(ctx->queue); ctx->queue = NULL; }
    if (ctx->context) { ctx->clReleaseContext(ctx->context); ctx->context = NULL; }
    if (ctx->lib) { NAT_CL_FREE_LIB(ctx->lib); ctx->lib = NULL; }
    ctx->available = false;
    ctx->device_count = 0;
    ctx->active_device_idx = -1;
}

#endif // NATIVA_OPENCL_H
