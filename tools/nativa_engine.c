// ============================================================================
// NATIVA QWEN2/QWEN3 INT4 C99 ENGINE (ZERO-DEPENDENCY RUNTIME)
// ============================================================================
// High-performance native inference engine:
//   - Reads Qwen2 and Qwen3 models in self-describing .natq4 format
//   - Dynamically allocates memory and tensors according to model header
//   - Automatic CPU detection via CPUID and OpenMP auto-tuning
//   - SIMD kernel SSSE3/SSE4.1/AVX with fused dot products
//   - Supports compatible vocabularies and dimensions within format limits
// ============================================================================

#define _CRT_SECURE_NO_WARNINGS 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include <immintrin.h>
#include <omp.h>

#include "natq4_format.h"
#include "cpu_dispatch.h"
#include "nativa_sampling.h"

#pragma pack(push, 1)
typedef struct {
    uint16_t scale;
    uint8_t  qs[16];
} BlockQ4;
#pragma pack(pop)

typedef struct {
    float scale;
    int8_t qs[32];
    int32_t sum_adj;
} BlockQ8;

#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4]; // 8 bytes: BF16 scales for 4 contiguous rows
    uint8_t  qs[64];   // 64 bytes: nibbles for the 4 rows
} BlockQ4x4;
#pragma pack(pop)

// ----------------------------------------------------------------------------
// DYNAMIC NATIVE VOCABULARY
// ----------------------------------------------------------------------------
typedef struct {
    uint16_t len;
    char* str;
} VocabEntry;

static VocabEntry* g_vocab = NULL;
static uint32_t g_vocab_size = 0;

static void free_vocab(void) {
    if (g_vocab) {
        for (uint32_t i = 0; i < g_vocab_size; i++) {
            if (g_vocab[i].str) {
                free(g_vocab[i].str);
                g_vocab[i].str = NULL;
            }
        }
        free(g_vocab);
        g_vocab = NULL;
    }
    g_vocab_size = 0;
}

static bool load_vocab_stream(FILE* f) {
    free_vocab();
    if (fread(&g_vocab_size, sizeof(uint32_t), 1, f) != 1) {
        return false;
    }
    if (g_vocab_size == 0 || g_vocab_size > 1000000) {
        fprintf(stderr, "[ERROR] Vocabulary size %u out of valid bounds (1..1000000).\n", g_vocab_size);
        g_vocab_size = 0;
        return false;
    }
    g_vocab = (VocabEntry*)calloc(g_vocab_size, sizeof(VocabEntry));
    if (!g_vocab) {
        g_vocab_size = 0;
        return false;
    }

    uint32_t i;
    for (i = 0; i < g_vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) {
            fprintf(stderr, "[ERROR] Truncated vocabulary file at token index %u.\n", i);
            free_vocab();
            return false;
        }
        g_vocab[i].len = len;
        g_vocab[i].str = (char*)malloc((size_t)len + 1);
        if (!g_vocab[i].str) {
            fprintf(stderr, "[ERROR] Memory allocation failed for token %u (len=%u).\n", i, len);
            free_vocab();
            return false;
        }
        if (len > 0) {
            size_t bytes_read = fread(g_vocab[i].str, 1, len, f);
            if (bytes_read != len) {
                fprintf(stderr, "[ERROR] Incomplete token read at index %u (expected %u bytes, got %zu).\n",
                        i, len, bytes_read);
                free_vocab();
                return false;
            }
        }
        g_vocab[i].str[len] = '\0';
    }
    return true;
}

static bool load_vocab(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "rb");
    if (!f) return false;
    bool ok = load_vocab_stream(f);
    fclose(f);
    return ok;
}

static bool load_vocab_from_offset(const char* path, uint64_t offset) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
#if defined(_WIN32)
    if (_fseeki64(f, (__int64)offset, SEEK_SET) != 0) {
#else
    if (fseeko(f, (off_t)offset, SEEK_SET) != 0) {
#endif
        fclose(f);
        return false;
    }
    bool ok = load_vocab_stream(f);
    fclose(f);
    return ok;
}

static void get_executable_path(char* out_path, size_t max_len, const char* argv0) {
#ifdef _WIN32
    DWORD len = GetModuleFileNameA(NULL, out_path, (DWORD)max_len);
    if (len == 0 || len >= max_len) {
        strncpy(out_path, argv0 ? argv0 : "", max_len - 1);
        out_path[max_len - 1] = '\0';
    }
#else
    ssize_t len = readlink("/proc/self/exe", out_path, max_len - 1);
    if (len > 0) {
        out_path[len] = '\0';
    } else {
        strncpy(out_path, argv0 ? argv0 : "", max_len - 1);
        out_path[max_len - 1] = '\0';
    }
#endif
}

static bool detect_bundle(const char* exe_path, NativaBundleTrailer* trailer) {
    if (!exe_path || !exe_path[0]) return false;
    FILE* f = fopen(exe_path, "rb");
    if (!f) return false;
#if defined(_WIN32)
    if (_fseeki64(f, -((__int64)sizeof(NativaBundleTrailer)), SEEK_END) != 0) {
#else
    if (fseeko(f, -((off_t)sizeof(NativaBundleTrailer)), SEEK_END) != 0) {
#endif
        fclose(f);
        return false;
    }
    if (fread(trailer, sizeof(NativaBundleTrailer), 1, f) != 1) {
        fclose(f);
        return false;
    }
    fclose(f);
    return (trailer->magic == NATIVA_BUNDLE_MAGIC && trailer->version == 1);
}

static inline void print_token(int token_id) {
    if (token_id >= 0 && (uint32_t)token_id < g_vocab_size && g_vocab[token_id].len > 0) {
        fwrite(g_vocab[token_id].str, 1, g_vocab[token_id].len, stdout);
        fflush(stdout);
    }
}

// ----------------------------------------------------------------------------
// TIMER
// ----------------------------------------------------------------------------
#ifdef _WIN32
static LARGE_INTEGER g_timer_freq;

static inline void timer_init(void) {
    QueryPerformanceFrequency(&g_timer_freq);
}

static inline double timer_now(void) {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)g_timer_freq.QuadPart;
}
#else
static inline void timer_init(void) {}

static inline double timer_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

// ----------------------------------------------------------------------------
// DYNAMIC MEMORY ARENA
// ----------------------------------------------------------------------------
typedef struct {
    float* k_cache; // [layers * max_seq_len * kv_dim]
    float* v_cache; // [layers * max_seq_len * kv_dim]
    float* x;
    float* xn;
    float* q;
    float* k;
    float* v;
    float* attn_out;
    float* h_mlp;
    float* x_final;
    float* logits;
    float* scores; // [num_heads * max_seq_len]
    BlockQ8* xn_q8;
    BlockQ8* attn_q8;
    BlockQ8* h_mlp_q8;
    BlockQ8* x_final_q8;
    float* rope_freqs; // [head_dim / 2]
} ModelArena;

static ModelArena g_arena;

static void arena_free(void) {
    if (g_arena.k_cache)    { free(g_arena.k_cache);    g_arena.k_cache = NULL; }
    if (g_arena.v_cache)    { free(g_arena.v_cache);    g_arena.v_cache = NULL; }
    if (g_arena.x)          { free(g_arena.x);          g_arena.x = NULL; }
    if (g_arena.xn)         { free(g_arena.xn);         g_arena.xn = NULL; }
    if (g_arena.q)          { free(g_arena.q);          g_arena.q = NULL; }
    if (g_arena.k)          { free(g_arena.k);          g_arena.k = NULL; }
    if (g_arena.v)          { free(g_arena.v);          g_arena.v = NULL; }
    if (g_arena.attn_out)   { free(g_arena.attn_out);   g_arena.attn_out = NULL; }
    if (g_arena.h_mlp)      { free(g_arena.h_mlp);      g_arena.h_mlp = NULL; }
    if (g_arena.x_final)    { free(g_arena.x_final);    g_arena.x_final = NULL; }
    if (g_arena.logits)     { free(g_arena.logits);     g_arena.logits = NULL; }
    if (g_arena.scores)     { free(g_arena.scores);     g_arena.scores = NULL; }
    if (g_arena.xn_q8)      { free(g_arena.xn_q8);      g_arena.xn_q8 = NULL; }
    if (g_arena.attn_q8)    { free(g_arena.attn_q8);    g_arena.attn_q8 = NULL; }
    if (g_arena.h_mlp_q8)   { free(g_arena.h_mlp_q8);   g_arena.h_mlp_q8 = NULL; }
    if (g_arena.x_final_q8) { free(g_arena.x_final_q8); g_arena.x_final_q8 = NULL; }
    if (g_arena.rope_freqs) { free(g_arena.rope_freqs); g_arena.rope_freqs = NULL; }
}

static inline bool check_mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static bool arena_allocate(const NativaModelHeader* hdr) {
    arena_free();

    uint32_t q_dim = hdr->num_heads * hdr->head_dim;
    uint32_t kv_dim = hdr->num_kv_heads * hdr->head_dim;

    size_t tmp = 0;
    size_t kv_cache_elements = 0;
    if (check_mul_overflow_size(hdr->num_layers, hdr->max_seq_len, &tmp) ||
        check_mul_overflow_size(tmp, kv_dim, &kv_cache_elements) ||
        check_mul_overflow_size(kv_cache_elements, sizeof(float), &tmp)) {
        fprintf(stderr, "[ERROR] KV-cache allocation size calculation overflow.\n");
        return false;
    }

    size_t scores_elements = 0;
    if (check_mul_overflow_size(hdr->num_heads, hdr->max_seq_len, &scores_elements) ||
        check_mul_overflow_size(scores_elements, sizeof(float), &tmp)) {
        fprintf(stderr, "[ERROR] Attention scores allocation size calculation overflow.\n");
        return false;
    }

    g_arena.k_cache    = (float*)malloc(kv_cache_elements * sizeof(float));
    g_arena.v_cache    = (float*)malloc(kv_cache_elements * sizeof(float));

    g_arena.x          = (float*)malloc((size_t)hdr->dim * sizeof(float));
    g_arena.xn         = (float*)malloc((size_t)hdr->dim * sizeof(float));
    g_arena.q          = (float*)malloc((size_t)q_dim * sizeof(float));
    g_arena.k          = (float*)malloc((size_t)kv_dim * sizeof(float));
    g_arena.v          = (float*)malloc((size_t)kv_dim * sizeof(float));
    g_arena.attn_out   = (float*)malloc((size_t)q_dim * sizeof(float));
    g_arena.h_mlp      = (float*)malloc((size_t)hdr->intermediate_dim * sizeof(float));
    g_arena.x_final    = (float*)malloc((size_t)hdr->dim * sizeof(float));
    g_arena.logits     = (float*)malloc((size_t)hdr->vocab_size * sizeof(float));
    g_arena.scores     = (float*)malloc(scores_elements * sizeof(float));

    g_arena.xn_q8      = (BlockQ8*)malloc(((size_t)hdr->dim / 32) * sizeof(BlockQ8));
    g_arena.attn_q8    = (BlockQ8*)malloc(((size_t)q_dim / 32) * sizeof(BlockQ8));
    g_arena.h_mlp_q8   = (BlockQ8*)malloc(((size_t)hdr->intermediate_dim / 32) * sizeof(BlockQ8));
    g_arena.x_final_q8 = (BlockQ8*)malloc(((size_t)hdr->dim / 32) * sizeof(BlockQ8));

    g_arena.rope_freqs = (float*)malloc(((size_t)hdr->head_dim / 2) * sizeof(float));

    if (!g_arena.k_cache || !g_arena.v_cache || !g_arena.x || !g_arena.xn ||
        !g_arena.q || !g_arena.k || !g_arena.v || !g_arena.attn_out ||
        !g_arena.h_mlp || !g_arena.x_final || !g_arena.logits || !g_arena.scores ||
        !g_arena.xn_q8 || !g_arena.attn_q8 || !g_arena.h_mlp_q8 || !g_arena.x_final_q8 ||
        !g_arena.rope_freqs) {
        fprintf(stderr, "[ERROR] Memory allocation failed during arena setup.\n");
        arena_free();
        return false;
    }

    // Initialize RoPE
    int half_head = hdr->head_dim / 2;
    for (int i = 0; i < half_head; i++) {
        g_arena.rope_freqs[i] = 1.0f / powf(hdr->rope_theta, (float)i / (float)half_head);
    }
    return true;
}

static bool range_is_valid(uint64_t offset, uint64_t bytes, uint64_t file_size, uint64_t data_start) {
    return offset >= data_start && offset <= file_size && bytes <= file_size - offset;
}

static bool matrix_range_is_valid(uint64_t offset, uint32_t rows, uint32_t cols,
                                 uint64_t file_size, uint64_t data_start) {
    if (rows == 0 || rows % 4 != 0 || cols == 0 || cols % 32 != 0) return false;
    uint64_t bytes = (uint64_t)rows * (uint64_t)(cols / 32) * 18ULL;
    return range_is_valid(offset, bytes, file_size, data_start);
}

static bool validate_model_header(const NativaModelHeader* hdr, uint64_t file_size) {
    if (memcmp(hdr->magic, NATQ4_MAGIC, sizeof(hdr->magic)) != 0) {
        fprintf(stderr, "[ERROR] Invalid NATQ4X4 signature.\n");
        return false;
    }
    if (hdr->version < 1 || hdr->version > NATQ4_VERSION) {
        fprintf(stderr, "[ERROR] Unsupported NATQ4 version: %u.\n", hdr->version);
        return false;
    }
    if ((hdr->arch != NAT_ARCH_QWEN2 && hdr->arch != NAT_ARCH_QWEN3) || hdr->act_type != NAT_ACT_SWIGLU) {
        fprintf(stderr, "[ERROR] Current engine only supports Qwen2/Qwen3 with SwiGLU.\n");
        return false;
    }
    if (hdr->arch == NAT_ARCH_QWEN3 && hdr->version < 3) {
        fprintf(stderr, "[ERROR] Qwen3 requires NATQ4 format version 3 (Q/K norms).\n");
        return false;
    }
    if (hdr->data_start_off != NATQ4_HEADER_SIZE || hdr->num_layers == 0 ||
        hdr->num_layers > NATQ4_MAX_LAYERS || hdr->vocab_size < 4 || hdr->vocab_size > 1000000 ||
        hdr->dim < 32 || hdr->dim > 16384 || hdr->dim % 32 != 0 ||
        hdr->intermediate_dim < 32 || hdr->intermediate_dim > 32768 || hdr->intermediate_dim % 32 != 0 ||
        hdr->vocab_size % 4 != 0 || hdr->num_heads == 0 || hdr->num_heads > 256 ||
        hdr->num_kv_heads == 0 || hdr->num_kv_heads > hdr->num_heads ||
        hdr->num_heads % hdr->num_kv_heads != 0 || hdr->head_dim == 0 ||
        (hdr->arch != NAT_ARCH_QWEN3 && hdr->head_dim != hdr->dim / hdr->num_heads) || hdr->head_dim % 16 != 0 ||
        ((uint64_t)hdr->num_kv_heads * hdr->head_dim) % 4 != 0 ||
        ((uint64_t)hdr->num_heads * hdr->head_dim) % 32 != 0 ||
        hdr->has_attn_bias > 1 || hdr->tie_embeddings > 1 ||
        (hdr->version >= 3 && (hdr->tokenizer_vocab_size < 4 || hdr->tokenizer_vocab_size > hdr->vocab_size)) ||
        hdr->max_seq_len < 2 || hdr->max_seq_len > 32768 ||
        !isfinite(hdr->rope_theta) || hdr->rope_theta <= 0.0f ||
        !isfinite(hdr->rms_norm_eps) || hdr->rms_norm_eps <= 0.0f) {
        fprintf(stderr, "[ERROR] Invalid or incompatible Q4x4 hyperparameters.\n");
        return false;
    }

    const uint64_t norm_bytes = (uint64_t)hdr->dim * sizeof(uint16_t);
    const uint32_t q_dim = hdr->num_heads * hdr->head_dim;
    const uint32_t kv_dim = hdr->num_kv_heads * hdr->head_dim;
    if (!matrix_range_is_valid(hdr->embed_tokens_off, hdr->vocab_size, hdr->dim, file_size, hdr->data_start_off) ||
        !range_is_valid(hdr->norm_off, norm_bytes, file_size, hdr->data_start_off)) {
        fprintf(stderr, "[ERROR] Embedding or final norm point outside the file.\n");
        return false;
    }
    if (!hdr->tie_embeddings &&
        !matrix_range_is_valid(hdr->lm_head_off, hdr->vocab_size, hdr->dim, file_size, hdr->data_start_off)) {
        fprintf(stderr, "[ERROR] LM head points outside the file.\n");
        return false;
    }

    for (uint32_t l = 0; l < hdr->num_layers; l++) {
        const NativaLayerOffsets* lo = &hdr->layers[l];
        if (!range_is_valid(lo->norm1, norm_bytes, file_size, hdr->data_start_off) ||
            !range_is_valid(lo->norm2, norm_bytes, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wq, q_dim, hdr->dim, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wk, kv_dim, hdr->dim, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wv, kv_dim, hdr->dim, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wo, hdr->dim, q_dim, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wgate, hdr->intermediate_dim, hdr->dim, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wup, hdr->intermediate_dim, hdr->dim, file_size, hdr->data_start_off) ||
            !matrix_range_is_valid(lo->wdown, hdr->dim, hdr->intermediate_dim, file_size, hdr->data_start_off)) {
            fprintf(stderr, "[ERROR] Invalid weight offsets in layer %u.\n", l);
            return false;
        }
        if (hdr->has_attn_bias &&
            (!range_is_valid(lo->bq, (uint64_t)q_dim * sizeof(uint16_t), file_size, hdr->data_start_off) ||
             !range_is_valid(lo->bk, (uint64_t)kv_dim * sizeof(uint16_t), file_size, hdr->data_start_off) ||
             !range_is_valid(lo->bv, (uint64_t)kv_dim * sizeof(uint16_t), file_size, hdr->data_start_off))) {
            fprintf(stderr, "[ERROR] Invalid bias offsets in layer %u.\n", l);
            return false;
        }
        if (hdr->arch == NAT_ARCH_QWEN3 &&
            (!range_is_valid(hdr->qk_norms[l].q_norm, (uint64_t)hdr->head_dim * sizeof(uint16_t), file_size, hdr->data_start_off) ||
             !range_is_valid(hdr->qk_norms[l].k_norm, (uint64_t)hdr->head_dim * sizeof(uint16_t), file_size, hdr->data_start_off))) {
            fprintf(stderr, "[ERROR] Invalid Q/K RMSNorm offsets in layer %u.\n", l);
            return false;
        }
    }
    return true;
}

static inline bool is_stop_token(const NativaModelHeader* hdr, int token_id) {
    if (hdr->version == 1) return token_id == 151645 || token_id == 151643;
    return (uint32_t)token_id == hdr->eos_token_id || (uint32_t)token_id == hdr->eos_token_id_2;
}

// ----------------------------------------------------------------------------
// OPERAÇÕES MATEMÁTICAS SIMD
// ----------------------------------------------------------------------------
static inline float bf16_to_fp32(uint16_t v) {
    union { uint32_t u; float f; } conv;
    conv.u = ((uint32_t)v) << 16;
    return conv.f;
}

static inline __m128 bf16x4_to_ps(const uint16_t* p) {
    return _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), _mm_loadl_epi64((const __m128i*)p)));
}

static inline void quantize_x_to_q8(const float* restrict x, BlockQ8* restrict x_q8, int cols) {
    int num_blocks = cols / 32;
    int b, i;
    for (b = 0; b < num_blocks; b++) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (i = 0; i < 32; i++) {
            float af = fabsf(xb[i]);
            if (af > amax) amax = af;
        }

        if (amax < 1e-12f) {
            x_q8[b].scale = 0.0f;
            memset(x_q8[b].qs, 0, 32);
            x_q8[b].sum_adj = 0;
            continue;
        }

        float scale = amax / 127.0f;
        float inv_scale = 1.0f / scale;
        x_q8[b].scale = scale;

        int32_t s = 0;
        for (i = 0; i < 32; i++) {
            int q = (int)roundf(xb[i] * inv_scale);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            x_q8[b].qs[i] = (int8_t)q;
            s += q;
        }
        x_q8[b].sum_adj = s * 8;
    }
}

static inline void rmsnorm(float* restrict out, const float* restrict x, const uint16_t* restrict weight, int dim, float eps) {
    float sum_sq = 0.0f;
    int i;
    for (i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (i = 0; i < dim; i++) {
        union { uint32_t u; float f; } wconv;
        wconv.u = ((uint32_t)weight[i]) << 16;
        out[i] = x[i] * inv_rms * wconv.f;
    }
}

static inline void rmsnorm_heads_inplace(float* vectors, int heads, int head_dim,
                                         const uint16_t* weight, float eps) {
    for (int h = 0; h < heads; h++) {
        float* x = vectors + (size_t)h * head_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < head_dim; i++) sum_sq += x[i] * x[i];
        float inv_rms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
        for (int i = 0; i < head_dim; i++) {
            union { uint32_t u; float f; } wconv;
            wconv.u = ((uint32_t)weight[i]) << 16;
            x[i] = x[i] * inv_rms * wconv.f;
        }
    }
}

static inline void apply_rope(float* vec, int pos, int num_heads, int head_dim, const float* freqs) {
    int half = head_dim / 2;
    int h;
    for (h = 0; h < num_heads; h++) {
        float* head_vec = vec + h * head_dim;
        int i;
        for (i = 0; i < half; i++) {
            float freq = (float)pos * freqs[i];
            float cos_th = cosf(freq);
            float sin_th = sinf(freq);
            float v0 = head_vec[i];
            float v1 = head_vec[i + half];
            head_vec[i]        = v0 * cos_th - v1 * sin_th;
            head_vec[i + half] = v0 * sin_th + v1 * cos_th;
        }
    }
}

// ----------------------------------------------------------------------------
// KERNEL Q4x4 UNIVERSAL (SIMD SSSE3/SSE4.1 & Full C99 Scalar Backend)
// ----------------------------------------------------------------------------
static bool g_cpu_has_simd = true;

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

static inline __m128 q4x4_dot4(const BlockQ4x4* restrict W4, const BlockQ8* restrict x_q8, int nb) {
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

static inline void matvec_q4x4_scalar_group(float* y, const BlockQ4x4* W, const BlockQ8* x_q8,
                                            const uint16_t* bias, int g, int nb, int add) {
    float out4[4];
    q4x4_dot4_scalar(W + (size_t)g * nb, x_q8, nb, out4);
    for (int r = 0; r < 4; r++) {
        float val = out4[r];
        if (bias) val += bf16_to_fp32(bias[g * 4 + r]);
        if (add) val += y[g * 4 + r];
        y[g * 4 + r] = val;
    }
}

static inline void matvec_q4x4(float* y, const BlockQ4x4* W, const BlockQ8* x_q8,
                               const uint16_t* bias, int rows, int cols, int add) {
    int nb = cols / 32;
    int groups = rows / 4;
    int g;
    if (g_cpu_has_simd) {
        #pragma omp parallel for schedule(static)
        for (g = 0; g < groups; g++) {
            __m128 acc = q4x4_dot4(W + (size_t)g * nb, x_q8, nb);
            if (bias) acc = _mm_add_ps(acc, bf16x4_to_ps(bias + g * 4));
            if (add) acc = _mm_add_ps(acc, _mm_loadu_ps(&y[g * 4]));
            _mm_storeu_ps(&y[g * 4], acc);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (g = 0; g < groups; g++) {
            matvec_q4x4_scalar_group(y, W, x_q8, bias, g, nb, add);
        }
    }
}

// ----------------------------------------------------------------------------
// VECTORIZED SWIGLU & TILED MLP KERNELS (CUDA Tile IR inspiration)
// ----------------------------------------------------------------------------
static inline __m128 fast_exp_ps(__m128 x) {
    __m128 max_x = _mm_set1_ps(88.3762626647949f);
    __m128 min_x = _mm_set1_ps(-88.3762626647949f);
    x = _mm_max_ps(min_x, _mm_min_ps(max_x, x));

    __m128 log2ef = _mm_set1_ps(1.44269504088896341f);
    __m128 fx = _mm_mul_ps(x, log2ef);
    fx = _mm_floor_ps(_mm_add_ps(fx, _mm_set1_ps(0.5f)));

    __m128 c_c1 = _mm_set1_ps(0.693359375f);
    __m128 c_c2 = _mm_set1_ps(-2.12194440e-4f);
    __m128 y = _mm_sub_ps(x, _mm_mul_ps(fx, c_c1));
    y = _mm_sub_ps(y, _mm_mul_ps(fx, c_c2));

    __m128 y2 = _mm_mul_ps(y, y);
    __m128 p = _mm_set1_ps(1.9875691500E-4f);
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(1.3981999507E-3f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(8.3334519073E-3f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(4.1665795894E-2f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(1.6666665459E-1f));
    p = _mm_add_ps(_mm_mul_ps(p, y), _mm_set1_ps(5.0000001201E-1f));
    p = _mm_add_ps(_mm_mul_ps(p, y2), y);
    p = _mm_add_ps(p, _mm_set1_ps(1.0f));

    __m128i emm0 = _mm_cvttps_epi32(fx);
    emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(127));
    emm0 = _mm_slli_epi32(emm0, 23);
    return _mm_mul_ps(p, _mm_castsi128_ps(emm0));
}

static inline __m128 fast_swiglu_ps(__m128 g, __m128 u) {
    __m128 neg_g = _mm_sub_ps(_mm_setzero_ps(), g);
    __m128 exp_neg = fast_exp_ps(neg_g);
    __m128 one_plus_exp = _mm_add_ps(_mm_set1_ps(1.0f), exp_neg);
    __m128 silu_g = _mm_div_ps(g, one_plus_exp);
    return _mm_mul_ps(silu_g, u);
}

static inline void mlp_gate_up_swiglu_q4x4_tiled(
    float* restrict h_mlp,
    const BlockQ4x4* restrict Wgate,
    const BlockQ4x4* restrict Wup,
    const BlockQ8* restrict x_q8,
    int rows,
    int cols
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

            // Gate
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

            // Up (reutiliza x_lo e x_hi já em registradores)
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

        __m128 out = fast_swiglu_ps(acc_g, acc_u);
        _mm_storeu_ps(&h_mlp[g * 4], out);
    }

    // Cauda de linhas (quando rows % 4 != 0)
    int rem_rows = rows % 4;
    if (rem_rows > 0) {
        int g = groups;
        const BlockQ4x4* g_blk_row = Wgate + (size_t)g * nb;
        const BlockQ4x4* u_blk_row = Wup   + (size_t)g * nb;
        for (int r = 0; r < rem_rows; r++) {
            float g_acc = 0.0f, u_acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                const BlockQ8* xb = &x_q8[b];
                const BlockQ4x4* gb = &g_blk_row[b];
                const BlockQ4x4* ub = &u_blk_row[b];
                float gw_sc = bf16_to_fp32(gb->scale[r]);
                float uw_sc = bf16_to_fp32(ub->scale[r]);
                const uint8_t* g_qs = gb->qs + r * 16;
                const uint8_t* u_qs = ub->qs + r * 16;
                int32_t g_dot = 0, u_dot = 0;
                for (int i = 0; i < 16; i++) {
                    int8_t glo = (int8_t)(g_qs[i] & 0x0F) - 8;
                    int8_t ghi = (int8_t)(g_qs[i] >> 4) - 8;
                    int8_t ulo = (int8_t)(u_qs[i] & 0x0F) - 8;
                    int8_t uhi = (int8_t)(u_qs[i] >> 4) - 8;
                    g_dot += (int32_t)glo * (int32_t)xb->qs[i] + (int32_t)ghi * (int32_t)xb->qs[i + 16];
                    u_dot += (int32_t)ulo * (int32_t)xb->qs[i] + (int32_t)uhi * (int32_t)xb->qs[i + 16];
                }
                g_acc += (float)g_dot * (gw_sc * xb->scale);
                u_acc += (float)u_dot * (uw_sc * xb->scale);
            }
            float silu_g = g_acc / (1.0f + expf(-g_acc));
            h_mlp[g * 4 + r] = silu_g * u_acc;
        }
    }
}

// ----------------------------------------------------------------------------
// FORWARD PASS UNIVERSAL
// ----------------------------------------------------------------------------
static double g_prof_attn = 0.0;
static double g_prof_mlp = 0.0;
static double g_prof_lm = 0.0;
static int g_prof_tokens = 0;

static bool forward_token_universal(const NativaModelHeader* hdr, const uint8_t* model_bytes, int token_id, int pos) {
    if (token_id < 0 || (uint32_t)token_id >= hdr->vocab_size) {
        fprintf(stderr, "[WARNING] Invalid token_id %d (vocab_size=%u, context_pos=%d). Skipping forward.\n",
                token_id, hdr->vocab_size, pos);
        return false;
    }
    if (pos < 0 || (uint32_t)pos >= hdr->max_seq_len) {
        fprintf(stderr, "[WARNING] Invalid context position %d (max_seq_len=%u). Skipping forward.\n",
                pos, hdr->max_seq_len);
        return false;
    }

    const int dim = hdr->dim;
    const int inter_dim = hdr->intermediate_dim;
    const int num_heads = hdr->num_heads;
    const int num_kv_heads = hdr->num_kv_heads;
    const int head_dim = hdr->head_dim;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int heads_per_kv = num_heads / num_kv_heads;

    // 1. Embedding lookup Q4x4
    const BlockQ4x4* emb_grp = (const BlockQ4x4*)(model_bytes + hdr->embed_tokens_off) + (size_t)(token_id / 4) * (dim / 32);
    const int lane = token_id % 4;

    if (g_cpu_has_simd) {
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

            _mm_storeu_ps(&g_arena.x[b * 32],      _mm_mul_ps(f0, svec));
            _mm_storeu_ps(&g_arena.x[b * 32 + 4],  _mm_mul_ps(f1, svec));
            _mm_storeu_ps(&g_arena.x[b * 32 + 8],  _mm_mul_ps(f2, svec));
            _mm_storeu_ps(&g_arena.x[b * 32 + 12], _mm_mul_ps(f3, svec));

            __m128 f4 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(hi));
            __m128 f5 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
            __m128 f6 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
            __m128 f7 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 12)));

            _mm_storeu_ps(&g_arena.x[b * 32 + 16], _mm_mul_ps(f4, svec));
            _mm_storeu_ps(&g_arena.x[b * 32 + 20], _mm_mul_ps(f5, svec));
            _mm_storeu_ps(&g_arena.x[b * 32 + 24], _mm_mul_ps(f6, svec));
            _mm_storeu_ps(&g_arena.x[b * 32 + 28], _mm_mul_ps(f7, svec));
        }
    } else {
        embedding_lookup_scalar(g_arena.x, emb_grp, lane, dim);
    }

    // 2. Loop through layers
    for (uint32_t l = 0; l < hdr->num_layers; l++) {
        const NativaLayerOffsets* lo = &hdr->layers[l];
        const uint16_t* norm1_w = (const uint16_t*)(model_bytes + lo->norm1);
        const uint16_t* norm2_w = (const uint16_t*)(model_bytes + lo->norm2);
        const BlockQ4x4* wq_w   = (const BlockQ4x4*)(model_bytes + lo->wq);
        const uint16_t* bq_w   = lo->bq ? (const uint16_t*)(model_bytes + lo->bq) : NULL;
        const BlockQ4x4* wk_w   = (const BlockQ4x4*)(model_bytes + lo->wk);
        const uint16_t* bk_w   = lo->bk ? (const uint16_t*)(model_bytes + lo->bk) : NULL;
        const BlockQ4x4* wv_w   = (const BlockQ4x4*)(model_bytes + lo->wv);
        const uint16_t* bv_w   = lo->bv ? (const uint16_t*)(model_bytes + lo->bv) : NULL;
        const BlockQ4x4* wo_w   = (const BlockQ4x4*)(model_bytes + lo->wo);
        const BlockQ4x4* wgate_w = (const BlockQ4x4*)(model_bytes + lo->wgate);
        const BlockQ4x4* wup_w   = (const BlockQ4x4*)(model_bytes + lo->wup);
        const BlockQ4x4* wdown_w = (const BlockQ4x4*)(model_bytes + lo->wdown);

        double t_a0 = timer_now();
        rmsnorm(g_arena.xn, g_arena.x, norm1_w, dim, hdr->rms_norm_eps);
        quantize_x_to_q8(g_arena.xn, g_arena.xn_q8, dim);

        // Fused QKV
        const int nb = dim / 32;
        const int gq = q_dim / 4;
        const int gk = kv_dim / 4;
        int g;
        if (g_cpu_has_simd) {
            #pragma omp parallel for schedule(static)
            for (g = 0; g < gq + 2 * gk; g++) {
                float* y; const BlockQ4x4* W; const uint16_t* bias; int r;
                if (g < gq)            { y = g_arena.q; W = wq_w; bias = bq_w; r = g; }
                else if (g < gq + gk)  { y = g_arena.k; W = wk_w; bias = bk_w; r = g - gq; }
                else                   { y = g_arena.v; W = wv_w; bias = bv_w; r = g - gq - gk; }
                __m128 acc = q4x4_dot4(W + (size_t)r * nb, g_arena.xn_q8, nb);
                if (bias) acc = _mm_add_ps(acc, bf16x4_to_ps(bias + r * 4));
                _mm_storeu_ps(&y[r * 4], acc);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (g = 0; g < gq + 2 * gk; g++) {
                float* y; const BlockQ4x4* W; const uint16_t* bias; int r;
                if (g < gq)            { y = g_arena.q; W = wq_w; bias = bq_w; r = g; }
                else if (g < gq + gk)  { y = g_arena.k; W = wk_w; bias = bk_w; r = g - gq; }
                else                   { y = g_arena.v; W = wv_w; bias = bv_w; r = g - gq - gk; }
                matvec_q4x4_scalar_group(y, W, g_arena.xn_q8, bias, r, nb, 0);
            }
        }

        if (hdr->arch == NAT_ARCH_QWEN3) {
            const uint16_t* q_norm_w = (const uint16_t*)(model_bytes + hdr->qk_norms[l].q_norm);
            const uint16_t* k_norm_w = (const uint16_t*)(model_bytes + hdr->qk_norms[l].k_norm);
            rmsnorm_heads_inplace(g_arena.q, num_heads, head_dim, q_norm_w, hdr->rms_norm_eps);
            rmsnorm_heads_inplace(g_arena.k, num_kv_heads, head_dim, k_norm_w, hdr->rms_norm_eps);
        }

        apply_rope(g_arena.q, pos, num_heads, head_dim, g_arena.rope_freqs);
        apply_rope(g_arena.k, pos, num_kv_heads, head_dim, g_arena.rope_freqs);

        // Save to KV-Cache
        size_t layer_kv_offset = ((size_t)l * hdr->max_seq_len + pos) * kv_dim;
        memcpy(g_arena.k_cache + layer_kv_offset, g_arena.k, kv_dim * sizeof(float));
        memcpy(g_arena.v_cache + layer_kv_offset, g_arena.v, kv_dim * sizeof(float));

        // Grouped Query Attention (GQA)
        const float inv_sqrt_head = 1.0f / sqrtf((float)head_dim);
        int h;
        for (h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_kv;
            const float* q_h = g_arena.q + h * head_dim;
            float* out_h = g_arena.attn_out + h * head_dim;
            float* head_scores = g_arena.scores + h * hdr->max_seq_len;
            const float* k_base = g_arena.k_cache + ((size_t)l * hdr->max_seq_len) * kv_dim + kv_h * head_dim;
            const float* v_base = g_arena.v_cache + ((size_t)l * hdr->max_seq_len) * kv_dim + kv_h * head_dim;

            float max_score = -1e30f;
            int s;
            if (g_cpu_has_simd) {
                for (s = 0; s <= pos; s++) {
                    const float* k_vec = k_base + (size_t)s * kv_dim;
                    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
                    __m128 a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
                    int d;
                    for (d = 0; d < head_dim; d += 16) {
                        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(q_h + d),      _mm_loadu_ps(k_vec + d)));
                        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(q_h + d + 4),  _mm_loadu_ps(k_vec + d + 4)));
                        a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_loadu_ps(q_h + d + 8),  _mm_loadu_ps(k_vec + d + 8)));
                        a3 = _mm_add_ps(a3, _mm_mul_ps(_mm_loadu_ps(q_h + d + 12), _mm_loadu_ps(k_vec + d + 12)));
                    }
                    __m128 sum128 = _mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3));
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    float dot = _mm_cvtss_f32(sum128) * inv_sqrt_head;
                    head_scores[s] = dot;
                    if (dot > max_score) max_score = dot;
                }
            } else {
                for (s = 0; s <= pos; s++) {
                    const float* k_vec = k_base + (size_t)s * kv_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q_h[d] * k_vec[d];
                    }
                    dot *= inv_sqrt_head;
                    head_scores[s] = dot;
                    if (dot > max_score) max_score = dot;
                }
            }

            float sum_exp = 0.0f;
            for (s = 0; s <= pos; s++) {
                float exp_v = expf(head_scores[s] - max_score);
                head_scores[s] = exp_v;
                sum_exp += exp_v;
            }
            float inv_sum = 1.0f / sum_exp;

            memset(out_h, 0, head_dim * sizeof(float));
            if (g_cpu_has_simd) {
                for (s = 0; s <= pos; s++) {
                    float p = head_scores[s] * inv_sum;
                    __m128 pvec = _mm_set1_ps(p);
                    const float* v_vec = v_base + (size_t)s * kv_dim;
                    int d;
                    for (d = 0; d < head_dim; d += 16) {
                        _mm_storeu_ps(out_h + d,      _mm_add_ps(_mm_loadu_ps(out_h + d),      _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d))));
                        _mm_storeu_ps(out_h + d + 4,  _mm_add_ps(_mm_loadu_ps(out_h + d + 4),  _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d + 4))));
                        _mm_storeu_ps(out_h + d + 8,  _mm_add_ps(_mm_loadu_ps(out_h + d + 8),  _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d + 8))));
                        _mm_storeu_ps(out_h + d + 12, _mm_add_ps(_mm_loadu_ps(out_h + d + 12), _mm_mul_ps(pvec, _mm_loadu_ps(v_vec + d + 12))));
                    }
                }
            } else {
                for (s = 0; s <= pos; s++) {
                    float p = head_scores[s] * inv_sum;
                    const float* v_vec = v_base + (size_t)s * kv_dim;
                    for (int d = 0; d < head_dim; d++) {
                        out_h[d] += p * v_vec[d];
                    }
                }
            }
        }

        quantize_x_to_q8(g_arena.attn_out, g_arena.attn_q8, q_dim);
        matvec_q4x4(g_arena.x, wo_w, g_arena.attn_q8, NULL, dim, q_dim, 1);
        g_prof_attn += (timer_now() - t_a0);

        // MLP
        double t_m0 = timer_now();
        rmsnorm(g_arena.xn, g_arena.x, norm2_w, dim, hdr->rms_norm_eps);
        quantize_x_to_q8(g_arena.xn, g_arena.xn_q8, dim);

        const int mlp_groups = inter_dim / 4;
        if (g_cpu_has_simd) {
            mlp_gate_up_swiglu_q4x4_tiled(g_arena.h_mlp, wgate_w, wup_w, g_arena.xn_q8, inter_dim, dim);
        } else {
            #pragma omp parallel for schedule(static)
            for (g = 0; g < mlp_groups; g++) {
                float gv[4], uv[4];
                q4x4_dot4_scalar(wgate_w + (size_t)g * nb, g_arena.xn_q8, nb, gv);
                q4x4_dot4_scalar(wup_w   + (size_t)g * nb, g_arena.xn_q8, nb, uv);
                for (int j = 0; j < 4; j++) {
                    g_arena.h_mlp[g * 4 + j] = (gv[j] / (1.0f + expf(-gv[j]))) * uv[j];
                }
            }
        }

        quantize_x_to_q8(g_arena.h_mlp, g_arena.h_mlp_q8, inter_dim);
        matvec_q4x4(g_arena.x, wdown_w, g_arena.h_mlp_q8, NULL, dim, inter_dim, 1);
        g_prof_mlp += (timer_now() - t_m0);
    }

    // Final Norm
    const uint16_t* final_norm_w = (const uint16_t*)(model_bytes + hdr->norm_off);
    rmsnorm(g_arena.x_final, g_arena.x, final_norm_w, dim, hdr->rms_norm_eps);
    quantize_x_to_q8(g_arena.x_final, g_arena.x_final_q8, dim);

    // LM Head
    double t_l0 = timer_now();
    uint64_t lm_off = hdr->tie_embeddings ? hdr->embed_tokens_off : hdr->lm_head_off;
    const BlockQ4x4* lm_w = (const BlockQ4x4*)(model_bytes + lm_off);
    matvec_q4x4(g_arena.logits, lm_w, g_arena.x_final_q8, NULL, hdr->vocab_size, dim, 0);
    g_prof_lm += (timer_now() - t_l0);
    g_prof_tokens++;
    return true;
}

// ----------------------------------------------------------------------------
// (Sampling functions TokenScore, sample_argmax, sample_topk in nativa_sampling.h)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// MAIN ENGINE
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    timer_init();

    char self_exe[1024];
    get_executable_path(self_exe, sizeof(self_exe), argv[0]);

    NativaBundleTrailer bundle_trailer;
    bool is_bundle = detect_bundle(self_exe, &bundle_trailer);

    const char* model_path = NULL;
    uint64_t model_offset = 0;
    uint64_t model_file_size = 0;
    const char* vocab_path = "data/qwen_vocab.bin";
    const char* eval_tokens_path = NULL;
    int custom_threads = 0;
    bool chat_mode = false;
    bool bench_mode = false;
    int bench_tokens = 25;
    bool force_scalar = false;

    if (is_bundle) {
        printf("[*] [NATIVA BUNDLE] Self-contained executable detected!\n");
        model_path = self_exe;
        model_offset = bundle_trailer.model_offset;
        model_file_size = bundle_trailer.model_size;
        chat_mode = (bundle_trailer.flags == 0);

        int a = 1;
        while (a < argc) {
            if (strcmp(argv[a], "--chat") == 0 || strcmp(argv[a], "--interactive") == 0) {
                chat_mode = true;
                bench_mode = false;
                a++;
            } else if (strcmp(argv[a], "--bench") == 0) {
                bench_mode = true;
                chat_mode = false;
                a++;
                if (a < argc && argv[a][0] != '-') {
                    bench_tokens = atoi(argv[a++]);
                }
            } else if (strcmp(argv[a], "--threads") == 0 && a + 1 < argc) {
                custom_threads = atoi(argv[a + 1]);
                a += 2;
            } else if (strcmp(argv[a], "--eval-tokens") == 0 && a + 1 < argc) {
                eval_tokens_path = argv[a + 1];
                a += 2;
            } else if (strcmp(argv[a], "--vocab") == 0 && a + 1 < argc) {
                vocab_path = argv[a + 1];
                a += 2;
            } else if (strcmp(argv[a], "--scalar") == 0) {
                force_scalar = true;
                a++;
            } else {
                a++;
            }
        }
    } else {
        int arg_idx = 1;
        if (argc > 1 && strcmp(argv[1], "--") == 0) {
            arg_idx = 2;
        }
        if (argc <= arg_idx) {
            printf("Usage: nativa_engine [--] <model.natq4> [--chat] [--bench <N>] [--threads <N>] [--eval-tokens <file>] [--vocab <vocab.bin>] [--scalar]\n");
            return 1;
        }

        model_path = argv[arg_idx];
        int a = arg_idx + 1;
        while (a < argc) {
            if (strcmp(argv[a], "--chat") == 0 || strcmp(argv[a], "--interactive") == 0) {
                chat_mode = true;
                a++;
            } else if (strcmp(argv[a], "--bench") == 0) {
                bench_mode = true;
                a++;
                if (a < argc && argv[a][0] != '-') {
                    bench_tokens = atoi(argv[a++]);
                }
            } else if (strcmp(argv[a], "--threads") == 0 && a + 1 < argc) {
                custom_threads = atoi(argv[a + 1]);
                a += 2;
            } else if (strcmp(argv[a], "--eval-tokens") == 0 && a + 1 < argc) {
                eval_tokens_path = argv[a + 1];
                a += 2;
            } else if (strcmp(argv[a], "--vocab") == 0 && a + 1 < argc) {
                vocab_path = argv[a + 1];
                a += 2;
            } else if (strcmp(argv[a], "--scalar") == 0) {
                force_scalar = true;
                a++;
            } else {
                a++;
            }
        }
    }

    // 1. Detect CPU and apply auto-tuning
    CPUInfo cpu;
    cpu_detect(&cpu);
    cpu_print_info(&cpu);
    cpu_apply_auto_tune(&cpu);
    if (custom_threads > 0) {
        omp_set_num_threads(custom_threads);
        printf("  [OPENMP] Threads manually set to: %d\n", custom_threads);
    }
    g_cpu_has_simd = cpu.ssse3 && cpu.sse41;
    if (force_scalar) {
        g_cpu_has_simd = false;
        printf("  [NATIVA UNIVERSAL] Flag --scalar specified: C99 scalar backend forced!\n");
    } else if (!g_cpu_has_simd) {
        printf("  [NATIVA UNIVERSAL] CPU without SSSE3/SSE4.1 detected. Full C99 scalar mode active!\n");
    } else {
        printf("  [NATIVA SIMD] SSSE3 + SSE4.1 acceleration active.\n");
    }

    // 2. Load vocabulary
    if (is_bundle) {
        if (!load_vocab_from_offset(self_exe, bundle_trailer.vocab_offset)) {
            fprintf(stderr, "[WARNING] Embedded vocabulary not loaded (only token_ids will be used)\n");
        }
    } else {
        if (!load_vocab(vocab_path)) {
            fprintf(stderr, "[WARNING] Vocabulary %s not loaded (only token_ids will be used)\n", vocab_path);
        }
    }

    // 3. Map .natq4 model file
    const uint8_t* model_bytes = NULL;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ERROR] Could not open %s (Code: %lu)\n", model_path, GetLastError());
        return 1;
    }

    if (!is_bundle) {
        LARGE_INTEGER model_size;
        if (!GetFileSizeEx(hFile, &model_size) || model_size.QuadPart < NATQ4_HEADER_SIZE) {
            fprintf(stderr, "[ERROR] NATQ4 file smaller than required header of %u bytes.\n", NATQ4_HEADER_SIZE);
            CloseHandle(hFile);
            return 1;
        }
        model_file_size = (uint64_t)model_size.QuadPart;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "[ERROR] Failed to create file mapping (Code: %lu)\n", GetLastError());
        CloseHandle(hFile);
        return 1;
    }

    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    uint64_t gran = sys_info.dwAllocationGranularity ? (uint64_t)sys_info.dwAllocationGranularity : 65536ULL;
    uint64_t aligned_offset = (model_offset / gran) * gran;
    uint64_t offset_delta = model_offset - aligned_offset;
    SIZE_T map_size = (SIZE_T)(model_file_size + offset_delta);

    DWORD off_high = (DWORD)(aligned_offset >> 32);
    DWORD off_low  = (DWORD)(aligned_offset & 0xFFFFFFFF);
    const uint8_t* raw_mapped = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, off_high, off_low, map_size);
    if (!raw_mapped) {
        fprintf(stderr, "[ERROR] Failed to map file to memory (Code: %lu)\n", GetLastError());
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }
    model_bytes = raw_mapped + offset_delta;

    #define UNMAP_AND_CLOSE() do { \
        if (raw_mapped) UnmapViewOfFile(raw_mapped); \
        if (hMapping) CloseHandle(hMapping); \
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile); \
        arena_free(); \
        free_vocab(); \
    } while (0)
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[ERROR] Could not open %s\n", model_path);
        return 1;
    }

    if (!is_bundle) {
        struct stat st;
        if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < NATQ4_HEADER_SIZE) {
            fprintf(stderr, "[ERROR] NATQ4 file smaller than required header of %u bytes.\n", NATQ4_HEADER_SIZE);
            close(fd);
            return 1;
        }
        model_file_size = (uint64_t)st.st_size;
    }

    long page_size = sysconf(_SC_PAGE_SIZE);
    uint64_t gran = page_size > 0 ? (uint64_t)page_size : 4096ULL;
    uint64_t aligned_offset = (model_offset / gran) * gran;
    uint64_t offset_delta = model_offset - aligned_offset;
    size_t map_size = (size_t)(model_file_size + offset_delta);

    const uint8_t* raw_mapped = (const uint8_t*)mmap(NULL, map_size, PROT_READ, MAP_SHARED, fd, (off_t)aligned_offset);
    if (raw_mapped == MAP_FAILED || !raw_mapped) {
        fprintf(stderr, "[ERROR] Failed to map file to memory\n");
        close(fd);
        return 1;
    }
    model_bytes = raw_mapped + offset_delta;

    #define UNMAP_AND_CLOSE() do { \
        if (raw_mapped && raw_mapped != MAP_FAILED) munmap((void*)raw_mapped, map_size); \
        if (fd >= 0) close(fd); \
        arena_free(); \
        free_vocab(); \
    } while (0)
#endif

    // 4. Validate model header
    const NativaModelHeader* hdr = (const NativaModelHeader*)model_bytes;
    if (!validate_model_header(hdr, model_file_size)) {
        UNMAP_AND_CLOSE();
        return 1;
    }

    if (g_vocab && (g_vocab_size > hdr->vocab_size ||
        (hdr->version >= 3 && g_vocab_size != hdr->tokenizer_vocab_size))) {
        fprintf(stderr, "[ERROR] Loaded vocabulary covers %u IDs; model requires %u active IDs and has %u rows.\n",
                g_vocab_size, hdr->version >= 3 ? hdr->tokenizer_vocab_size : g_vocab_size, hdr->vocab_size);
        UNMAP_AND_CLOSE();
        return 1;
    }

    printf("[*] Model .natq4 loaded successfully:\n");
    printf("    -> Dimension     : %u | Intermediate: %u\n", hdr->dim, hdr->intermediate_dim);
    printf("    -> Layers        : %u | Heads: %u Q / %u KV (Head Dim: %u)\n",
           hdr->num_layers, hdr->num_heads, hdr->num_kv_heads, hdr->head_dim);
    const int sampling_vocab_size = (int)(hdr->version >= 3 ? hdr->tokenizer_vocab_size :
        (g_vocab ? g_vocab_size : hdr->vocab_size));
    printf("    -> Vocabulary    : %d active IDs / %u rows | Max Context: %u\n",
           sampling_vocab_size, hdr->vocab_size, hdr->max_seq_len);

    // 5. Allocate dynamic arena
    if (!arena_allocate(hdr)) {
        fprintf(stderr, "[ERROR] Failed to allocate dynamic memory arena!\n");
        UNMAP_AND_CLOSE();
        return 1;
    }

    if (chat_mode) {
        printf("[NATIVA_CHAT_READY]\n");
        fflush(stdout);

        int cur_seq_pos = 0;
        char* line_buf = (char*)malloc(131072);
        if (!line_buf) {
            fprintf(stderr, "[ERROR] Failed to allocate line buffer!\n");
            UNMAP_AND_CLOSE();
            return 1;
        }

        while (fgets(line_buf, 131072, stdin)) {
            if (strncmp(line_buf, "QUIT", 4) == 0) break;
            if (strncmp(line_buf, "RESET", 5) == 0 || strncmp(line_buf, "CLEAR", 5) == 0) {
                cur_seq_pos = 0;
                continue;
            }
            if (strncmp(line_buf, "EVAL_PPL_FILE", 13) == 0) {
                char* fpath = line_buf + 13;
                while (*fpath && (*fpath == ' ' || *fpath == '\t')) fpath++;
                char* end_p = fpath + strlen(fpath) - 1;
                while (end_p >= fpath && (*end_p == '\r' || *end_p == '\n' || *end_p == ' ' || *end_p == '\t')) {
                    *end_p = '\0';
                    end_p--;
                }
                FILE* tf = fopen(fpath, "r");
                if (!tf) {
                    printf("\n[PPL_ERROR failed to open file: %s]\n", fpath);
                    fflush(stdout);
                    continue;
                }
                cur_seq_pos = 0;
                double t_ppl0 = timer_now();
                int evaluated = 0;
                int correct = 0;
                double total_nll = 0.0;
                int prev_tok = -1;
                int cur_tok = 0;
                while (fscanf(tf, "%d", &cur_tok) == 1 && cur_seq_pos < (int)hdr->max_seq_len - 1) {
                    if (cur_tok < 0 || (uint32_t)cur_tok >= hdr->vocab_size) {
                        fprintf(stderr, "[WARNING] Rejecting out-of-bounds token %d in PPL file (vocab_size=%u).\n",
                                cur_tok, hdr->vocab_size);
                        break;
                    }
                    if (prev_tok != -1) {
                        float max_l = -1e30f;
                        for (int v = 0; v < sampling_vocab_size; v++) {
                            if (g_arena.logits[v] > max_l) max_l = g_arena.logits[v];
                        }
                        float sum_exp = 0.0f;
                        int best_v = 0;
                        for (int v = 0; v < sampling_vocab_size; v++) {
                            float exp_v = expf(g_arena.logits[v] - max_l);
                            sum_exp += exp_v;
                            if (g_arena.logits[v] > g_arena.logits[best_v]) best_v = v;
                        }
                        float target_l = (cur_tok >= 0 && cur_tok < sampling_vocab_size) ? g_arena.logits[cur_tok] : -1e30f;
                        float log_prob = target_l - max_l - logf(sum_exp);
                        total_nll -= (double)log_prob;
                        if (best_v == cur_tok) correct++;
                        if (evaluated < 64) {
                            printf("[NLL_STEP %2d] id=%7d nll=%.4f best=%7d acc=%d\n",
                                    evaluated, cur_tok, -log_prob, best_v, best_v == cur_tok);
                            fflush(stdout);
                        }
                        evaluated++;
                    }
                    forward_token_universal(hdr, model_bytes, cur_tok, cur_seq_pos);
                    cur_seq_pos++;
                    prev_tok = cur_tok;
                }
                fclose(tf);
                double ppl_sec = timer_now() - t_ppl0;
                double mean_nll = (evaluated > 0) ? (total_nll / evaluated) : 0.0;
                double ppl_val = exp(mean_nll);
                double acc_val = (evaluated > 0) ? (100.0 * correct / evaluated) : 0.0;
                double ppl_speed = (ppl_sec > 0.0) ? (evaluated / ppl_sec) : 0.0;
                printf("\n[PPL_DONE tokens=%d nll=%.4f ppl=%.4f acc=%.2f time=%.3f speed=%.2f]\n",
                        evaluated, mean_nll, ppl_val, acc_val, ppl_sec, ppl_speed);
                fflush(stdout);
                continue;
            }
            if (strncmp(line_buf, "INPUT", 5) != 0) continue;

            char* ptr = line_buf + 5;
            int num_in = 0, max_gen = 1000;
            if (sscanf(ptr, "%d %d", &num_in, &max_gen) < 1 || num_in <= 0) continue;
            if (max_gen <= 0) max_gen = 1000;

            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            while (*ptr && (*ptr >= '0' && *ptr <= '9')) ptr++;
            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            while (*ptr && (*ptr >= '0' && *ptr <= '9')) ptr++;
            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;

            int t_idx;
            for (t_idx = 0; t_idx < num_in && cur_seq_pos < (int)hdr->max_seq_len; t_idx++) {
                int tok_val = 0;
                if (sscanf(ptr, "%d", &tok_val) != 1) break;
                if (tok_val < 0 || (uint32_t)tok_val >= hdr->vocab_size) {
                    fprintf(stderr, "[WARNING] Rejecting out-of-bounds token %d (vocab_size=%u).\n", tok_val, hdr->vocab_size);
                    while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
                    while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
                    continue;
                }
                forward_token_universal(hdr, model_bytes, tok_val, cur_seq_pos);
                cur_seq_pos++;
                while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
                while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            }

            int gen_step = 0;
            double t_turn0 = timer_now();
            while (cur_seq_pos < (int)hdr->max_seq_len - 1 && gen_step < max_gen) {
                int next_tok = sample_topk(g_arena.logits, sampling_vocab_size, 0.6f, 30);
                if (next_tok < 0 || (uint32_t)next_tok >= hdr->vocab_size) break;
                if (is_stop_token(hdr, next_tok)) break;

                print_token(next_tok);
                forward_token_universal(hdr, model_bytes, next_tok, cur_seq_pos);
                cur_seq_pos++;
                gen_step++;
            }
            double turn_sec = timer_now() - t_turn0;
            double speed = (turn_sec > 0.0) ? (gen_step / turn_sec) : 0.0;
            printf("\n[TURN_DONE tokens=%d time=%.2f speed=%.2f]\n", gen_step, turn_sec, speed);
            fflush(stdout);
        }
        free(line_buf);
    } else if (eval_tokens_path) {
        FILE* tf = fopen(eval_tokens_path, "r");
        if (!tf) {
            fprintf(stderr, "[ERROR] Could not open tokens file: %s\n", eval_tokens_path);
            UNMAP_AND_CLOSE();
            return 1;
        }

        int max_cap = 65536;
        int* token_list = (int*)malloc(max_cap * sizeof(int));
        if (!token_list) {
            fprintf(stderr, "[ERROR] Failed to allocate memory for token list.\n");
            fclose(tf);
            UNMAP_AND_CLOSE();
            return 1;
        }
        int total_tokens = 0;
        int val = 0;
        while (fscanf(tf, "%d", &val) == 1) {
            if (val < 0 || (uint32_t)val >= hdr->vocab_size) {
                fprintf(stderr, "[WARNING] Rejecting out-of-bounds token %d from evaluation file.\n", val);
                continue;
            }
            if (total_tokens >= max_cap) {
                max_cap *= 2;
                int* new_list = (int*)realloc(token_list, max_cap * sizeof(int));
                if (!new_list) break;
                token_list = new_list;
            }
            token_list[total_tokens++] = val;
        }
        fclose(tf);

        if (total_tokens < 2) {
            fprintf(stderr, "[ERROR] Tokens file requires at least 2 tokens to compute PPL.\n");
            free(token_list);
            UNMAP_AND_CLOSE();
            return 1;
        }

        int max_seq = (int)hdr->max_seq_len;
        if (total_tokens > max_seq) {
            printf("[WARNING] Truncating tokens to model context limit (%d tokens).\n", max_seq);
            total_tokens = max_seq;
        }

        printf("\n====================================================================\n");
        printf("  [NATIVA BENCHMARK] PERPLEXITY EVALUATION (WIKITEXT / NLL)\n");
        printf("====================================================================\n");
        printf("  Tokens in split   : %d\n", total_tokens);
        printf("  Evaluated pairs   : %d\n", total_tokens - 1);
        printf("  Active vocabulary : %d\n", sampling_vocab_size);
        printf("--------------------------------------------------------------------\n");

        double t_ppl0 = timer_now();
        int evaluated = 0;
        int correct = 0;
        double total_nll = 0.0;

        for (int p = 0; p < total_tokens - 1; p++) {
            int input_tok = token_list[p];
            int target_tok = token_list[p + 1];

            forward_token_universal(hdr, model_bytes, input_tok, p);

            float max_l = -1e30f;
            for (int v = 0; v < sampling_vocab_size; v++) {
                if (g_arena.logits[v] > max_l) max_l = g_arena.logits[v];
            }
            float sum_exp = 0.0f;
            int best_v = 0;
            for (int v = 0; v < sampling_vocab_size; v++) {
                float exp_v = expf(g_arena.logits[v] - max_l);
                sum_exp += exp_v;
                if (g_arena.logits[v] > g_arena.logits[best_v]) best_v = v;
            }
            float target_l = (target_tok >= 0 && target_tok < sampling_vocab_size) ? g_arena.logits[target_tok] : -1e30f;
            float log_prob = target_l - max_l - logf(sum_exp);
            total_nll -= (double)log_prob;
            if (best_v == target_tok) correct++;
            evaluated++;

            if (p < 512) {
                printf("[NLL_NATIVA] pos=%3d input=%7d target=%7d nll=%.4f best=%7d\n",
                        p, input_tok, target_tok, -log_prob, best_v);
                fflush(stdout);
            }

            if ((p + 1) % 50 == 0 || p == total_tokens - 2) {
                double cur_sec = timer_now() - t_ppl0;
                double cur_speed = (cur_sec > 0.0) ? (evaluated / cur_sec) : 0.0;
                double cur_nll = total_nll / evaluated;
                double cur_ppl = exp(cur_nll);
                double cur_acc = 100.0 * correct / evaluated;
                printf("  [%4d/%4d] NLL: %.4f | PPL: %.2f | Top-1: %.2f%% | Speed: %.1f tok/s\r",
                        evaluated, total_tokens - 1, cur_nll, cur_ppl, cur_acc, cur_speed);
                fflush(stdout);
            }
        }
        printf("\n--------------------------------------------------------------------\n");

        double ppl_sec = timer_now() - t_ppl0;
        double mean_nll = (evaluated > 0) ? (total_nll / evaluated) : 0.0;
        double ppl_val = exp(mean_nll);
        double acc_val = (evaluated > 0) ? (100.0 * correct / evaluated) : 0.0;
        double ppl_speed = (ppl_sec > 0.0) ? (evaluated / ppl_sec) : 0.0;

        printf("  [OFFICIAL PERPLEXITY RESULT]:\n");
        printf("    Evaluated tokens : %d\n", evaluated);
        printf("    Mean NLL         : %.4f nats/token\n", mean_nll);
        printf("    Perplexity PPL   : \x1b[1;32m%.4f\x1b[0m\n", ppl_val);
        printf("    Top-1 Accuracy   : %.2f%%\n", acc_val);
        printf("    Total time       : %.3f s\n", ppl_sec);
        printf("    Throughput       : %.2f tokens/s\n", ppl_speed);
        printf("====================================================================\n");
        printf("[PPL_DONE tokens=%d nll=%.4f ppl=%.4f acc=%.2f time=%.3f speed=%.2f]\n",
                evaluated, mean_nll, ppl_val, acc_val, ppl_sec, ppl_speed);
        fflush(stdout);

        free(token_list);
    } else {
        // Benchmark mode
        int default_prompt[] = { 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 77091, 198 };
        int prompt_len = 14;

        printf("\n[*] Running prompt prefill (%d tokens)...\n", prompt_len);
        double t0 = timer_now();
        for (int p = 0; p < prompt_len - 1; p++) {
            if (default_prompt[p] >= 0 && (uint32_t)default_prompt[p] < hdr->vocab_size) {
                forward_token_universal(hdr, model_bytes, default_prompt[p], p);
            }
        }
        printf("    -> Prefill completed in %.3f s!\n\n", timer_now() - t0);

        printf("[*] Generating %d tokens in universal engine:\n----------------------------------------------------\n", bench_tokens);
        int cur = default_prompt[prompt_len - 1];
        int pos = prompt_len - 1;
        int gen = 0;
        uint32_t seq_hash = 2166136261u;

        g_prof_attn = g_prof_mlp = g_prof_lm = 0.0;
        g_prof_tokens = 0;
        double t_gen0 = timer_now();

        while (gen < bench_tokens && pos < (int)hdr->max_seq_len - 1) {
            forward_token_universal(hdr, model_bytes, cur, pos);
            int next = bench_mode ? sample_argmax(g_arena.logits, sampling_vocab_size)
                                  : sample_topk(g_arena.logits, sampling_vocab_size, 0.6f, 30);
            if (next < 0 || (uint32_t)next >= hdr->vocab_size) break;
            if (!bench_mode && is_stop_token(hdr, next)) break;

            seq_hash = (seq_hash ^ (uint32_t)next) * 16777619u;
            if (!bench_mode) print_token(next);
            cur = next;
            pos++;
            gen++;
        }
        double gen_sec = timer_now() - t_gen0;
        double tok_s = (gen_sec > 0.0) ? (gen / gen_sec) : 0.0;

        printf("\n----------------------------------------------------\n");
        printf("[UNIVERSAL .NATQ4 ENGINE RESULT]:\n");
        printf("  Tokens generated : %d\n", gen);
        printf("  Total time     : %.3f s\n", gen_sec);
        printf("  Speed          : \x1b[1;32m%.2f tokens/second\x1b[0m\n", tok_s);
        if (bench_mode) printf("  BENCH_HASH=%08x\n", seq_hash);
        printf("====================================================================\n");
    }

    UNMAP_AND_CLOSE();
    return 0;
}
