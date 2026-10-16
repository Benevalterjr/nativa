// ============================================================================
// NATIVA QWEN2/QWEN3 INT4 C99 ENGINE (ZERO-DEPENDENCY RUNTIME)
// ============================================================================
// Motor de inferência nativo de alto desempenho:
//   - Lê modelos Qwen2 e Qwen3 no formato autodescritivo .natq4
//   - Aloca memória e tensores dinamicamente conforme o cabeçalho do modelo
//   - Detecção automática de CPU via CPUID e auto-ajuste de OpenMP
//   - Kernel SIMD SSSE3/SSE4.1/AVX com produtos escalares fundidos
//   - Suporta vocabulários compatíveis e dimensões dentro dos limites do formato
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
    uint16_t scale[4]; // 8 bytes: escalas BF16 para as 4 linhas contiguas
    uint8_t  qs[64];   // 64 bytes: nibbles das 4 linhas
} BlockQ4x4;
#pragma pack(pop)

// ----------------------------------------------------------------------------
// VOCABULÁRIO NATIVO DINÂMICO
// ----------------------------------------------------------------------------
typedef struct {
    uint16_t len;
    char* str;
} VocabEntry;

static VocabEntry* g_vocab = NULL;
static uint32_t g_vocab_size = 0;

static bool load_vocab(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "rb");
    if (!f) return false;
    if (fread(&g_vocab_size, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    g_vocab = (VocabEntry*)malloc(g_vocab_size * sizeof(VocabEntry));
    if (!g_vocab) { fclose(f); return false; }

    uint32_t i;
    for (i = 0; i < g_vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) {
            g_vocab[i].len = 0;
            g_vocab[i].str = "";
            continue;
        }
        g_vocab[i].len = len;
        g_vocab[i].str = (char*)malloc(len + 1);
        if (len > 0) {
            fread(g_vocab[i].str, 1, len, f);
        }
        g_vocab[i].str[len] = '\0';
    }
    fclose(f);
    return true;
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
// MEMORY ARENA DINÂMICA
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

static bool arena_allocate(const NativaModelHeader* hdr) {
    uint32_t q_dim = hdr->num_heads * hdr->head_dim;
    uint32_t kv_dim = hdr->num_kv_heads * hdr->head_dim;
    size_t kv_cache_elements = (size_t)hdr->num_layers * hdr->max_seq_len * kv_dim;

    g_arena.k_cache = (float*)malloc(kv_cache_elements * sizeof(float));
    g_arena.v_cache = (float*)malloc(kv_cache_elements * sizeof(float));

    g_arena.x        = (float*)malloc(hdr->dim * sizeof(float));
    g_arena.xn       = (float*)malloc(hdr->dim * sizeof(float));
    g_arena.q        = (float*)malloc(q_dim * sizeof(float));
    g_arena.k        = (float*)malloc(kv_dim * sizeof(float));
    g_arena.v        = (float*)malloc(kv_dim * sizeof(float));
    g_arena.attn_out = (float*)malloc(q_dim * sizeof(float));
    g_arena.h_mlp    = (float*)malloc(hdr->intermediate_dim * sizeof(float));
    g_arena.x_final  = (float*)malloc(hdr->dim * sizeof(float));
    g_arena.logits   = (float*)malloc(hdr->vocab_size * sizeof(float));
    g_arena.scores   = (float*)malloc((size_t)hdr->num_heads * hdr->max_seq_len * sizeof(float));

    g_arena.xn_q8      = (BlockQ8*)malloc((hdr->dim / 32) * sizeof(BlockQ8));
    g_arena.attn_q8    = (BlockQ8*)malloc((q_dim / 32) * sizeof(BlockQ8));
    g_arena.h_mlp_q8   = (BlockQ8*)malloc((hdr->intermediate_dim / 32) * sizeof(BlockQ8));
    g_arena.x_final_q8 = (BlockQ8*)malloc((hdr->dim / 32) * sizeof(BlockQ8));

    g_arena.rope_freqs = (float*)malloc((hdr->head_dim / 2) * sizeof(float));

    if (!g_arena.k_cache || !g_arena.v_cache || !g_arena.x || !g_arena.logits) {
        return false;
    }

    // Inicializa RoPE
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
        fprintf(stderr, "[ERRO] Assinatura NATQ4X4 inválida.\n");
        return false;
    }
    if (hdr->version < 1 || hdr->version > NATQ4_VERSION) {
        fprintf(stderr, "[ERRO] Versão NATQ4 não suportada: %u.\n", hdr->version);
        return false;
    }
    if ((hdr->arch != NAT_ARCH_QWEN2 && hdr->arch != NAT_ARCH_QWEN3) || hdr->act_type != NAT_ACT_SWIGLU) {
        fprintf(stderr, "[ERRO] O motor atual aceita apenas Qwen2/Qwen3 com SwiGLU.\n");
        return false;
    }
    if (hdr->arch == NAT_ARCH_QWEN3 && hdr->version < 3) {
        fprintf(stderr, "[ERRO] Qwen3 requer o formato NATQ4 versão 3 (normas Q/K).\n");
        return false;
    }
    if (hdr->data_start_off != NATQ4_HEADER_SIZE || hdr->num_layers == 0 ||
        hdr->num_layers > NATQ4_MAX_LAYERS || hdr->vocab_size < 4 || hdr->vocab_size > 1000000 ||
        hdr->dim < 32 || hdr->dim > 16384 || hdr->dim % 32 != 0 ||
        hdr->intermediate_dim < 32 || hdr->intermediate_dim > 32768 || hdr->intermediate_dim % 32 != 0 ||
        hdr->vocab_size % 4 != 0 || hdr->num_heads == 0 || hdr->num_heads > 256 ||
        hdr->num_kv_heads == 0 || hdr->num_kv_heads > hdr->num_heads ||
        hdr->num_heads % hdr->num_kv_heads != 0 || hdr->head_dim == 0 ||
        hdr->num_heads % hdr->num_kv_heads != 0 || hdr->head_dim == 0 ||
        (hdr->arch != NAT_ARCH_QWEN3 && hdr->head_dim != hdr->dim / hdr->num_heads) || hdr->head_dim % 16 != 0 ||
        ((uint64_t)hdr->num_kv_heads * hdr->head_dim) % 4 != 0 ||
        ((uint64_t)hdr->num_heads * hdr->head_dim) % 32 != 0 ||
        hdr->has_attn_bias > 1 || hdr->tie_embeddings > 1 ||
        (hdr->version >= 3 && (hdr->tokenizer_vocab_size < 4 || hdr->tokenizer_vocab_size > hdr->vocab_size)) ||
        hdr->max_seq_len < 2 || hdr->max_seq_len > 32768 ||
        !isfinite(hdr->rope_theta) || hdr->rope_theta <= 0.0f ||
        !isfinite(hdr->rms_norm_eps) || hdr->rms_norm_eps <= 0.0f) {
        fprintf(stderr, "[ERRO] Hiperparâmetros inválidos ou incompatíveis com Q4x4.\n");
        return false;
    }

    const uint64_t norm_bytes = (uint64_t)hdr->dim * sizeof(uint16_t);
    const uint32_t q_dim = hdr->num_heads * hdr->head_dim;
    const uint32_t kv_dim = hdr->num_kv_heads * hdr->head_dim;
    if (!matrix_range_is_valid(hdr->embed_tokens_off, hdr->vocab_size, hdr->dim, file_size, hdr->data_start_off) ||
        !range_is_valid(hdr->norm_off, norm_bytes, file_size, hdr->data_start_off)) {
        fprintf(stderr, "[ERRO] Embedding ou norma final apontam para fora do arquivo.\n");
        return false;
    }
    if (!hdr->tie_embeddings &&
        !matrix_range_is_valid(hdr->lm_head_off, hdr->vocab_size, hdr->dim, file_size, hdr->data_start_off)) {
        fprintf(stderr, "[ERRO] LM head aponta para fora do arquivo.\n");
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
            fprintf(stderr, "[ERRO] Offsets de pesos inválidos na camada %u.\n", l);
            return false;
        }
        if (hdr->has_attn_bias &&
            (!range_is_valid(lo->bq, norm_bytes, file_size, hdr->data_start_off) ||
             !range_is_valid(lo->bk, (uint64_t)kv_dim * sizeof(uint16_t), file_size, hdr->data_start_off) ||
             !range_is_valid(lo->bv, (uint64_t)kv_dim * sizeof(uint16_t), file_size, hdr->data_start_off))) {
            fprintf(stderr, "[ERRO] Offsets de bias inválidos na camada %u.\n", l);
            return false;
        }
        if (hdr->arch == NAT_ARCH_QWEN3 &&
            (!range_is_valid(hdr->qk_norms[l].q_norm, (uint64_t)hdr->head_dim * sizeof(uint16_t), file_size, hdr->data_start_off) ||
             !range_is_valid(hdr->qk_norms[l].k_norm, (uint64_t)hdr->head_dim * sizeof(uint16_t), file_size, hdr->data_start_off))) {
            fprintf(stderr, "[ERRO] Offsets de RMSNorm Q/K inválidos na camada %u.\n", l);
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
// KERNEL Q4x4 UNIVERSAL (SSSE3 Fused 4-Row Vector Dot Product)
// ----------------------------------------------------------------------------
static bool g_cpu_has_ssse3 = true;

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

static inline void q4x4_dot4_scalar(const BlockQ4x4* restrict W4, const BlockQ8* restrict x_q8, int nb, float* out4) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    int b;
    for (b = 0; b < nb; b++) {
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

            int8_t x_lo = xb->qs[i * 2];
            int8_t x_hi = xb->qs[i * 2 + 1];

            dot0 += ((int)(b0 & 0x0F) - 8) * (int)x_lo + ((int)(b0 >> 4) - 8) * (int)x_hi;
            dot1 += ((int)(b1 & 0x0F) - 8) * (int)x_lo + ((int)(b1 >> 4) - 8) * (int)x_hi;
            dot2 += ((int)(b2 & 0x0F) - 8) * (int)x_lo + ((int)(b2 >> 4) - 8) * (int)x_hi;
            dot3 += ((int)(b3 & 0x0F) - 8) * (int)x_lo + ((int)(b3 >> 4) - 8) * (int)x_hi;
        }

        float sc = xb->scale;
        acc0 += (float)dot0 * s0 * sc;
        acc1 += (float)dot1 * s1 * sc;
        acc2 += (float)dot2 * s2 * sc;
        acc3 += (float)dot3 * s3 * sc;
    }
    out4[0] = acc0;
    out4[1] = acc1;
    out4[2] = acc2;
    out4[3] = acc3;
}

static inline void matvec_q4x4(float* restrict y, const BlockQ4x4* restrict W, const BlockQ8* restrict x_q8,
                               const uint16_t* restrict bias, int rows, int cols, int add) {
    int nb = cols / 32;
    int groups = rows / 4;
    int g;
    if (g_cpu_has_ssse3) {
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
            float out4[4];
            q4x4_dot4_scalar(W + (size_t)g * nb, x_q8, nb, out4);
            for (int r = 0; r < 4; r++) {
                float val = out4[r];
                if (bias) val += bf16_to_fp32(bias[g * 4 + r]);
                if (add) val += y[g * 4 + r];
                y[g * 4 + r] = val;
            }
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

static void forward_token_universal(const NativaModelHeader* hdr, const uint8_t* model_bytes, int token_id, int pos) {
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

    // 2. Loop através das camadas
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

        if (hdr->arch == NAT_ARCH_QWEN3) {
            const uint16_t* q_norm_w = (const uint16_t*)(model_bytes + hdr->qk_norms[l].q_norm);
            const uint16_t* k_norm_w = (const uint16_t*)(model_bytes + hdr->qk_norms[l].k_norm);
            rmsnorm_heads_inplace(g_arena.q, num_heads, head_dim, q_norm_w, hdr->rms_norm_eps);
            rmsnorm_heads_inplace(g_arena.k, num_kv_heads, head_dim, k_norm_w, hdr->rms_norm_eps);
        }

        apply_rope(g_arena.q, pos, num_heads, head_dim, g_arena.rope_freqs);
        apply_rope(g_arena.k, pos, num_kv_heads, head_dim, g_arena.rope_freqs);

        // Grava no KV-Cache
        size_t layer_kv_offset = ((size_t)l * hdr->max_seq_len + pos) * kv_dim;
        memcpy(g_arena.k_cache + layer_kv_offset, g_arena.k, kv_dim * sizeof(float));
        memcpy(g_arena.v_cache + layer_kv_offset, g_arena.v, kv_dim * sizeof(float));

        // Grouped Query Attention (GQA) SIMD
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

            float sum_exp = 0.0f;
            for (s = 0; s <= pos; s++) {
                float exp_v = expf(head_scores[s] - max_score);
                head_scores[s] = exp_v;
                sum_exp += exp_v;
            }
            float inv_sum = 1.0f / sum_exp;

            memset(out_h, 0, head_dim * sizeof(float));
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
        }

        quantize_x_to_q8(g_arena.attn_out, g_arena.attn_q8, q_dim);
        matvec_q4x4(g_arena.x, wo_w, g_arena.attn_q8, NULL, dim, q_dim, 1);
        g_prof_attn += (timer_now() - t_a0);

        // MLP
        double t_m0 = timer_now();
        rmsnorm(g_arena.xn, g_arena.x, norm2_w, dim, hdr->rms_norm_eps);
        quantize_x_to_q8(g_arena.xn, g_arena.xn_q8, dim);

        const int mlp_groups = inter_dim / 4;
        #pragma omp parallel for schedule(static)
        for (g = 0; g < mlp_groups; g++) {
            float gv[4], uv[4];
            _mm_storeu_ps(gv, q4x4_dot4(wgate_w + (size_t)g * nb, g_arena.xn_q8, nb));
            _mm_storeu_ps(uv, q4x4_dot4(wup_w   + (size_t)g * nb, g_arena.xn_q8, nb));
            int j;
            for (j = 0; j < 4; j++) {
                g_arena.h_mlp[g * 4 + j] = (gv[j] / (1.0f + expf(-gv[j]))) * uv[j];
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
}

// ----------------------------------------------------------------------------
// (Funções de amostragem TokenScore, sample_argmax, sample_topk em nativa_sampling.h)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// MAIN ENGINE
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    timer_init();

    if (argc < 2) {
        printf("Uso: nativa_engine.exe <modelo.natq4> [--chat] [--bench <N>] [--vocab <vocab.bin>]\n");
        return 1;
    }

    const char* model_path = argv[1];
    const char* vocab_path = "data/qwen_vocab.bin";
    bool chat_mode = false;
    bool bench_mode = false;
    int bench_tokens = 25;

    for (int a = 2; a < argc; a++) {
        if (strcmp(argv[a], "--chat") == 0 || strcmp(argv[a], "--interactive") == 0) {
            chat_mode = true;
        } else if (strcmp(argv[a], "--bench") == 0 && a + 1 < argc) {
            bench_mode = true;
            bench_tokens = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--vocab") == 0 && a + 1 < argc) {
            vocab_path = argv[++a];
        }
    }

    // 1. Detectar CPU e aplicar auto-ajuste
    CPUInfo cpu;
    cpu_detect(&cpu);
    cpu_print_info(&cpu);
    cpu_apply_auto_tune(&cpu);
    g_cpu_has_ssse3 = cpu.ssse3;
    if (!g_cpu_has_ssse3) {
        printf("  [NATIVA UNIVERSAL] Processador sem SSSE3 detectado. Modo escalar C99 ativo!\n");
    }

    // 2. Carregar vocabulario
    if (!load_vocab(vocab_path)) {
        fprintf(stderr, "[AVISO] Vocabulario %s nao carregado (apenas token_ids serao usados)\n", vocab_path);
    }

    // 3. Mapear arquivo .natq4
    uint64_t model_file_size = 0;
    const uint8_t* model_bytes = NULL;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ERRO] Nao foi possivel abrir %s (Codigo: %lu)\n", model_path, GetLastError());
        return 1;
    }

    LARGE_INTEGER model_size;
    if (!GetFileSizeEx(hFile, &model_size) || model_size.QuadPart < NATQ4_HEADER_SIZE) {
        fprintf(stderr, "[ERRO] Arquivo NATQ4 menor que o cabeçalho obrigatório de %u bytes.\n", NATQ4_HEADER_SIZE);
        CloseHandle(hFile);
        return 1;
    }
    model_file_size = (uint64_t)model_size.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "[ERRO] Falha ao criar mapeamento de arquivo\n");
        CloseHandle(hFile);
        return 1;
    }

    model_bytes = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!model_bytes) {
        fprintf(stderr, "[ERRO] Falha ao mapear arquivo na memoria\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }

    #define UNMAP_AND_CLOSE() do { \
        if (model_bytes) UnmapViewOfFile(model_bytes); \
        if (hMapping) CloseHandle(hMapping); \
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile); \
    } while (0)
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[ERRO] Nao foi possivel abrir %s\n", model_path);
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < NATQ4_HEADER_SIZE) {
        fprintf(stderr, "[ERRO] Arquivo NATQ4 menor que o cabeçalho obrigatório de %u bytes.\n", NATQ4_HEADER_SIZE);
        close(fd);
        return 1;
    }
    model_file_size = (uint64_t)st.st_size;

    model_bytes = (const uint8_t*)mmap(NULL, (size_t)model_file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (model_bytes == MAP_FAILED || !model_bytes) {
        fprintf(stderr, "[ERRO] Falha ao mapear arquivo na memoria\n");
        close(fd);
        return 1;
    }

    #define UNMAP_AND_CLOSE() do { \
        if (model_bytes && model_bytes != MAP_FAILED) munmap((void*)model_bytes, (size_t)model_file_size); \
        if (fd >= 0) close(fd); \
    } while (0)
#endif

    // 4. Validar cabeçalho do modelo
    const NativaModelHeader* hdr = (const NativaModelHeader*)model_bytes;
    if (!validate_model_header(hdr, model_file_size)) {
        UNMAP_AND_CLOSE();
        return 1;
    }

    if (g_vocab && (g_vocab_size > hdr->vocab_size ||
        (hdr->version >= 3 && g_vocab_size != hdr->tokenizer_vocab_size))) {
        fprintf(stderr, "[ERRO] O vocabulário carregado cobre %u IDs; o modelo requer %u IDs ativos e tem %u linhas.\n",
                g_vocab_size, hdr->version >= 3 ? hdr->tokenizer_vocab_size : g_vocab_size, hdr->vocab_size);
        UNMAP_AND_CLOSE();
        return 1;
    }

    printf("[*] Modelo .natq4 carregado com sucesso:\n");
    printf("    -> Dimensao      : %u | Intermediaria: %u\n", hdr->dim, hdr->intermediate_dim);
    printf("    -> Camadas       : %u | Cabecas: %u Q / %u KV (Head Dim: %u)\n",
           hdr->num_layers, hdr->num_heads, hdr->num_kv_heads, hdr->head_dim);
    const int sampling_vocab_size = (int)(hdr->version >= 3 ? hdr->tokenizer_vocab_size :
        (g_vocab ? g_vocab_size : hdr->vocab_size));
    printf("    -> Vocabulário   : %d IDs ativos / %u linhas | Contexto Max: %u\n",
           sampling_vocab_size, hdr->vocab_size, hdr->max_seq_len);

    // 5. Alocar arena dinamica
    if (!arena_allocate(hdr)) {
        fprintf(stderr, "[ERRO] Falha ao alocar arena dinamica de memoria!\n");
        UNMAP_AND_CLOSE();
        return 1;
    }

    if (chat_mode) {
        printf("[NATIVA_CHAT_READY]\n");
        fflush(stdout);

        int cur_seq_pos = 0;
        char* line_buf = (char*)malloc(131072);
        if (!line_buf) {
            fprintf(stderr, "[ERRO] Falha ao alocar buffer de linha!\n");
            UNMAP_AND_CLOSE();
            return 1;
        }

        while (fgets(line_buf, 131072, stdin)) {
            if (strncmp(line_buf, "QUIT", 4) == 0) break;
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
                forward_token_universal(hdr, model_bytes, tok_val, cur_seq_pos);
                cur_seq_pos++;
                while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
                while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            }

            int gen_step = 0;
            double t_turn0 = timer_now();
            while (cur_seq_pos < (int)hdr->max_seq_len - 1 && gen_step < max_gen) {
                int next_tok = sample_topk(g_arena.logits, sampling_vocab_size, 0.6f, 30);
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
    } else {
        // Benchmark mode
        int default_prompt[] = { 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 77091, 198 };
        int prompt_len = 14;

        printf("\n[*] Executando prefill do prompt (%d tokens)...\n", prompt_len);
        double t0 = timer_now();
        for (int p = 0; p < prompt_len - 1; p++) {
            forward_token_universal(hdr, model_bytes, default_prompt[p], p);
        }
        printf("    -> Prefill concluido em %.3f s!\n\n", timer_now() - t0);

        printf("[*] Gerando %d tokens no motor universal:\n----------------------------------------------------\n", bench_tokens);
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
        printf("[RESULTADO MOTOR UNIVERSAL .NATQ4]:\n");
        printf("  Tokens gerados : %d\n", gen);
        printf("  Tempo total    : %.3f s\n", gen_sec);
        printf("  Velocidade     : \x1b[1;32m%.2f tokens/segundo\x1b[0m\n", tok_s);
        if (bench_mode) printf("  BENCH_HASH=%08x\n", seq_hash);
        printf("====================================================================\n");
    }

    UNMAP_AND_CLOSE();
    return 0;
}
