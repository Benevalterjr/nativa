// ============================================================================
// NATIVA NATIVE C RUNNER - QWEN2.5-1.5B-INSTRUCT IN PURE C99
// ============================================================================
// Motor de inferência nativo de alta performance para Qwen2.5-1.5B (28 Camadas):
//   - Zero-Copy mmap do modelo oficial safetensors (3.08 GB)
//   - SIMD AVX/SSE para cálculo de BFloat16 sob demanda
//   - OpenMP multi-core em todas as 28 camadas
//   - Suporte a modo benchmark e modo interativo (--interactive)
// ============================================================================

#define _CRT_SECURE_NO_WARNINGS 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include <omp.h>

#include "qwen_offsets_1.5b.h"

#define MAX_SEQ_LEN 1024

// ----------------------------------------------------------------------------
// VOCABULÁRIO NATIVO
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
    g_vocab = (VocabEntry*)calloc(g_vocab_size, sizeof(VocabEntry));
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
        if (!g_vocab[i].str) {
            g_vocab[i].len = 0;
            g_vocab[i].str = "";
            continue;
        }
        if (len > 0) {
            fread(g_vocab[i].str, 1, len, f);
        }
        g_vocab[i].str[len] = '\0';
    }
    fclose(f);
    return true;
}

static inline void print_token(int token_id) {
    if (g_vocab && token_id >= 0 && (uint32_t)token_id < g_vocab_size && g_vocab[token_id].len > 0) {
        fwrite(g_vocab[token_id].str, 1, g_vocab[token_id].len, stdout);
        fflush(stdout);
    }
}

// ----------------------------------------------------------------------------
// TIMER WINDOWS
// ----------------------------------------------------------------------------
static LARGE_INTEGER g_timer_freq;

static inline void timer_init(void) {
    QueryPerformanceFrequency(&g_timer_freq);
}

static inline double timer_now(void) {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)g_timer_freq.QuadPart;
}

// ----------------------------------------------------------------------------
// MEMORY ARENA GLOBAL (KV-CACHE E ATIVAÇÕES 1.5B)
// ----------------------------------------------------------------------------
static float g_k_cache[QWEN15_NUM_LAYERS][MAX_SEQ_LEN][QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];
static float g_v_cache[QWEN15_NUM_LAYERS][MAX_SEQ_LEN][QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];

static float g_x[QWEN15_DIM];
static float g_xn[QWEN15_DIM];
static float g_q[QWEN15_DIM];
static float g_k[QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];
static float g_v[QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM];
static float g_attn_out[QWEN15_DIM];
static float g_gate[QWEN15_INTERMEDIATE_DIM];
static float g_up[QWEN15_INTERMEDIATE_DIM];
static float g_h_mlp[QWEN15_INTERMEDIATE_DIM];
static float g_x_final[QWEN15_DIM];
static float g_logits[QWEN15_VOCAB_SIZE];
static float g_scores[QWEN15_NUM_HEADS][MAX_SEQ_LEN];

static float g_rope_freqs[64];

static void init_rope_freqs(void) {
    int i;
    for (i = 0; i < 64; i++) {
        g_rope_freqs[i] = 1.0f / powf(QWEN15_ROPE_THETA, (float)i / 64.0f);
    }
}

// ----------------------------------------------------------------------------
// OPERAÇÕES MATEMÁTICAS SIMD
// ----------------------------------------------------------------------------
static inline float bf16_to_fp32(uint16_t v) {
    union { uint32_t u; float f; } conv;
    conv.u = ((uint32_t)v) << 16;
    return conv.f;
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

static inline void apply_rope(float* vec, int pos, int heads) {
    int h, i;
    for (h = 0; h < heads; h++) {
        float* head_ptr = vec + h * 128;
        for (i = 0; i < 64; i++) {
            float angle = (float)pos * g_rope_freqs[i];
            float cos_v = cosf(angle);
            float sin_v = sinf(angle);
            float x1 = head_ptr[i];
            float x2 = head_ptr[i + 64];
            head_ptr[i] = x1 * cos_v - x2 * sin_v;
            head_ptr[i + 64] = x1 * sin_v + x2 * cos_v;
        }
    }
}

static inline void matvec_bf16(float* restrict y, const uint16_t* restrict W, const float* restrict x, const uint16_t* restrict bias, int rows, int cols) {
    __m128i zero = _mm_setzero_si128();
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i++) {
        const uint16_t* row_w = W + (size_t)i * cols;
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();
        int j;
        
        for (j = 0; j < cols; j += 16) {
            __m128i bf0 = _mm_loadu_si128((const __m128i*)(row_w + j));
            __m128i bf1 = _mm_loadu_si128((const __m128i*)(row_w + j + 8));
            
            __m128 f0 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, bf0));
            __m128 f1 = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, bf0));
            __m128 f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, bf1));
            __m128 f3 = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, bf1));
            
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(f0, _mm_loadu_ps(x + j)));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(f1, _mm_loadu_ps(x + j + 4)));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(f2, _mm_loadu_ps(x + j + 8)));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(f3, _mm_loadu_ps(x + j + 12)));
        }
        __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
        __m128 shuf = _mm_movehdup_ps(acc);
        __m128 sums = _mm_add_ps(acc, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);
        
        if (bias) {
            union { uint32_t u; float f; } bconv;
            bconv.u = ((uint32_t)bias[i]) << 16;
            sum += bconv.f;
        }
        y[i] = sum;
    }
}

static inline void matvec_bf16_add(float* restrict y, const uint16_t* restrict W, const float* restrict x, int rows, int cols) {
    __m128i zero = _mm_setzero_si128();
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i++) {
        const uint16_t* row_w = W + (size_t)i * cols;
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();
        int j;
        
        for (j = 0; j < cols; j += 16) {
            __m128i bf0 = _mm_loadu_si128((const __m128i*)(row_w + j));
            __m128i bf1 = _mm_loadu_si128((const __m128i*)(row_w + j + 8));
            
            __m128 f0 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, bf0));
            __m128 f1 = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, bf0));
            __m128 f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, bf1));
            __m128 f3 = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, bf1));
            
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(f0, _mm_loadu_ps(x + j)));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(f1, _mm_loadu_ps(x + j + 4)));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(f2, _mm_loadu_ps(x + j + 8)));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(f3, _mm_loadu_ps(x + j + 12)));
        }
        __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
        __m128 shuf = _mm_movehdup_ps(acc);
        __m128 sums = _mm_add_ps(acc, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        y[i] += _mm_cvtss_f32(sums);
    }
}

// ----------------------------------------------------------------------------
// FORWARD PASS 1.5B
// ----------------------------------------------------------------------------
static void forward_token(const uint8_t* model_bytes, int token_id, int pos) {
    // 1. Embedding lookup
    const uint16_t* embed_row = (const uint16_t*)(model_bytes + g_qwen15_offsets.embed_tokens) + (size_t)token_id * QWEN15_DIM;
    int i;
    for (i = 0; i < QWEN15_DIM; i++) {
        g_x[i] = bf16_to_fp32(embed_row[i]);
    }

    // 2. Loop através das 28 camadas
    for (int l = 0; l < QWEN15_NUM_LAYERS; l++) {
        const Qwen15LayerOffsets* lo = &g_qwen15_offsets.layers[l];
        const uint16_t* norm1_w = (const uint16_t*)(model_bytes + lo->norm1);
        const uint16_t* norm2_w = (const uint16_t*)(model_bytes + lo->norm2);
        const uint16_t* wq_w = (const uint16_t*)(model_bytes + lo->wq);
        const uint16_t* bq_w = (const uint16_t*)(model_bytes + lo->bq);
        const uint16_t* wk_w = (const uint16_t*)(model_bytes + lo->wk);
        const uint16_t* bk_w = (const uint16_t*)(model_bytes + lo->bk);
        const uint16_t* wv_w = (const uint16_t*)(model_bytes + lo->wv);
        const uint16_t* bv_w = (const uint16_t*)(model_bytes + lo->bv);
        const uint16_t* wo_w = (const uint16_t*)(model_bytes + lo->wo);
        const uint16_t* wgate_w = (const uint16_t*)(model_bytes + lo->wgate);
        const uint16_t* wup_w = (const uint16_t*)(model_bytes + lo->wup);
        const uint16_t* wdown_w = (const uint16_t*)(model_bytes + lo->wdown);

        // Input RMSNorm
        rmsnorm(g_xn, g_x, norm1_w, QWEN15_DIM, QWEN15_RMS_EPS);

        // Q, K, V Projections
        matvec_bf16(g_q, wq_w, g_xn, bq_w, QWEN15_DIM, QWEN15_DIM);
        matvec_bf16(g_k, wk_w, g_xn, bk_w, QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM, QWEN15_DIM);
        matvec_bf16(g_v, wv_w, g_xn, bv_w, QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM, QWEN15_DIM);

        // RoPE Positional Embeddings
        apply_rope(g_q, pos, QWEN15_NUM_HEADS);
        apply_rope(g_k, pos, QWEN15_NUM_KV_HEADS);

        // Grava no KV-Cache
        memcpy(&g_k_cache[l][pos][0], g_k, (QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM) * sizeof(float));
        memcpy(&g_v_cache[l][pos][0], g_v, (QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM) * sizeof(float));

        // Grouped Query Attention (GQA) paralelo sobre os 12 heads (12/2 = 6)
        int h;
        #pragma omp parallel for schedule(static)
        for (h = 0; h < QWEN15_NUM_HEADS; h++) {
            int kv_h = h / 6;
            const float* q_h = g_q + h * QWEN15_HEAD_DIM;
            float* out_h = g_attn_out + h * QWEN15_HEAD_DIM;
            float* head_scores = g_scores[h];

            float max_score = -1e30f;
            int s, d;
            for (s = 0; s <= pos; s++) {
                const float* k_vec = &g_k_cache[l][s][kv_h * QWEN15_HEAD_DIM];
                float dot = 0.0f;
                for (d = 0; d < QWEN15_HEAD_DIM; d++) {
                    dot += q_h[d] * k_vec[d];
                }
                dot *= 0.0883883476f; // 1 / sqrt(128)
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

            for (d = 0; d < QWEN15_HEAD_DIM; d++) {
                out_h[d] = 0.0f;
            }
            for (s = 0; s <= pos; s++) {
                float p = head_scores[s] * inv_sum;
                const float* v_vec = &g_v_cache[l][s][kv_h * QWEN15_HEAD_DIM];
                for (d = 0; d < QWEN15_HEAD_DIM; d++) {
                    out_h[d] += p * v_vec[d];
                }
            }
        }

        // Residual + Wo
        matvec_bf16_add(g_x, wo_w, g_attn_out, QWEN15_DIM, QWEN15_DIM);

        // Post-Attention RMSNorm
        rmsnorm(g_xn, g_x, norm2_w, QWEN15_DIM, QWEN15_RMS_EPS);

        // SwiGLU MLP
        matvec_bf16(g_gate, wgate_w, g_xn, NULL, QWEN15_INTERMEDIATE_DIM, QWEN15_DIM);
        matvec_bf16(g_up, wup_w, g_xn, NULL, QWEN15_INTERMEDIATE_DIM, QWEN15_DIM);

        int mi;
        #pragma omp parallel for schedule(static)
        for (mi = 0; mi < QWEN15_INTERMEDIATE_DIM; mi++) {
            float g = g_gate[mi];
            float silu_g = g / (1.0f + expf(-g));
            g_h_mlp[mi] = silu_g * g_up[mi];
        }

        // Residual + Wdown
        matvec_bf16_add(g_x, wdown_w, g_h_mlp, QWEN15_DIM, QWEN15_INTERMEDIATE_DIM);
    }

    // Final RMSNorm
    const uint16_t* norm_w = (const uint16_t*)(model_bytes + g_qwen15_offsets.norm);
    rmsnorm(g_x_final, g_x, norm_w, QWEN15_DIM, QWEN15_RMS_EPS);

    // LM Head
    const uint16_t* embed_w = (const uint16_t*)(model_bytes + g_qwen15_offsets.embed_tokens);
    matvec_bf16(g_logits, embed_w, g_x_final, NULL, QWEN15_VOCAB_SIZE, QWEN15_DIM);
}

static inline int sample_argmax(const float* logits, int size) {
    int best = 0;
    float max_val = logits[0];
    int i;
    for (i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best = i;
        }
    }
    return best;
}

// ----------------------------------------------------------------------------
// PROGRAMA PRINCIPAL
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("======================================================================\n");
    printf("  NATIVA NATIVE C RUNNER - QWEN2.5-1.5B-INSTRUCT (C99 ENGINE)\n");
    printf("======================================================================\n");

    timer_init();
    init_rope_freqs();

    const char* model_path = "data/qwen2.5_1.5b_instruct.safetensors";
    const char* vocab_path = "data/qwen_vocab.bin";
    if (GetFileAttributesA(model_path) == INVALID_FILE_ATTRIBUTES) {
        if (GetFileAttributesA("../data/qwen2.5_1.5b_instruct.safetensors") != INVALID_FILE_ATTRIBUTES) {
            model_path = "../data/qwen2.5_1.5b_instruct.safetensors";
            vocab_path = "../data/qwen_vocab.bin";
        }
    }

    // 1. Carrega Vocabulário Binário
    printf("[*] Carregando vocabulario (%s)...\n", vocab_path);
    if (!load_vocab(vocab_path)) {
        fprintf(stderr, "[ERRO] Nao foi possivel carregar o vocabulario: %s\n", vocab_path);
        return 1;
    }
    printf("    -> %u tokens carregados no vocabulario.\n", g_vocab_size);

    // 2. Win32 Zero-Copy mmap do modelo safetensors
    printf("[*] Mapeando pesos do modelo 1.5B via Win32 mmap (%s)...\n", model_path);
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ERRO] Falha ao abrir o arquivo do modelo: %s\n", model_path);
        return 1;
    }
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "[ERRO] CreateFileMapping falhou.\n");
        CloseHandle(hFile);
        return 1;
    }
    const uint8_t* model_bytes = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!model_bytes) {
        fprintf(stderr, "[ERRO] MapViewOfFile falhou.\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }
    printf("    -> Modelo 1.5B mapeado em memoria virtual com sucesso (Zero-Copy).\n");

    // -------------------------------------------------------------
    // MODO INTERATIVO (CHAT STREAMING MULTI-TURN EM C)
    // -------------------------------------------------------------
    if (argc > 1 && strcmp(argv[1], "--interactive") == 0) {
        printf("[NATIVA_CHAT_READY]\n");
        fflush(stdout);

        int cur_seq_pos = 0;
        char* line_buf = (char*)malloc(131072);

        while (fgets(line_buf, 131072, stdin)) {
            if (strncmp(line_buf, "QUIT", 4) == 0) break;
            if (strncmp(line_buf, "INPUT", 5) != 0) continue;

            char* ptr = line_buf + 5;
            int num_in = 0, max_gen = 60;
            if (sscanf(ptr, "%d %d", &num_in, &max_gen) < 1 || num_in <= 0) continue;
            if (max_gen <= 0) max_gen = 60;

            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            while (*ptr && (*ptr >= '0' && *ptr <= '9')) ptr++;
            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            while (*ptr && (*ptr >= '0' && *ptr <= '9')) ptr++;
            while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;

            int t_idx;
            for (t_idx = 0; t_idx < num_in && cur_seq_pos < MAX_SEQ_LEN; t_idx++) {
                int tok_val = 0;
                if (sscanf(ptr, "%d", &tok_val) != 1) break;
                forward_token(model_bytes, tok_val, cur_seq_pos);
                cur_seq_pos++;
                while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
                while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
            }

            int gen_step = 0;
            while (cur_seq_pos < MAX_SEQ_LEN - 1 && gen_step < max_gen) {
                int next_tok = sample_argmax(g_logits, QWEN15_VOCAB_SIZE);
                if (next_tok == 151645 || next_tok == 151643) {
                    break;
                }
                print_token(next_tok);
                forward_token(model_bytes, next_tok, cur_seq_pos);
                cur_seq_pos++;
                gen_step++;
            }

            printf("\n[TURN_DONE]\n");
            fflush(stdout);
        }

        free(line_buf);
        UnmapViewOfFile(model_bytes);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 0;
    }

    // Modo Benchmark
    int prompt_tokens_buf[1024];
    int prompt_len = 5;
    int num_tokens_generate = 20;
    bool custom_prompt = false;

    prompt_tokens_buf[0] = 9286;
    prompt_tokens_buf[1] = 16488;
    prompt_tokens_buf[2] = 21392;
    prompt_tokens_buf[3] = 374;
    prompt_tokens_buf[4] = 45790;

    int a;
    for (a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--tokens") == 0 && a + 1 < argc) {
            int n_toks = atoi(argv[a + 1]);
            if (n_toks > 0 && n_toks < 1024 && a + 1 + n_toks < argc) {
                prompt_len = n_toks;
                custom_prompt = true;
                int t;
                for (t = 0; t < n_toks; t++) {
                    prompt_tokens_buf[t] = atoi(argv[a + 2 + t]);
                }
                a += 1 + n_toks;
                if (a + 1 < argc) {
                    num_tokens_generate = atoi(argv[a + 1]);
                }
            }
        }
    }

    if (custom_prompt) {
        printf("\n[*] Prompt customizado (%d tokens):\n\"", prompt_len);
        int p_i;
        for (p_i = 0; p_i < prompt_len; p_i++) {
            print_token(prompt_tokens_buf[p_i]);
        }
        printf("\"\n");
    } else {
        printf("\n[*] Prompt oficial: 'Artificial Intelligence is transforming'\n");
    }
    printf("[*] Tokens no prompt: %d\n", prompt_len);

    printf("[*] Executando Prefill nativo em C (1.5B)...\n");
    double t_prefill_start = timer_now();

    int pos;
    for (pos = 0; pos < prompt_len; pos++) {
        forward_token(model_bytes, prompt_tokens_buf[pos], pos);
    }

    int next_token = sample_argmax(g_logits, QWEN15_VOCAB_SIZE);
    double t_prefill = timer_now() - t_prefill_start;

    printf("    -> Prefill concluido em %.2f ms (%.1f tokens/s).\n", t_prefill * 1000.0, (double)prompt_len / t_prefill);
    printf("    -> Primeiro token gerado: [ID %d] ", next_token);
    print_token(next_token);
    printf("\n");

    printf("\n[*] Gerando %d tokens no Qwen2.5-1.5B nativo C99...\n", num_tokens_generate);
    if (custom_prompt) {
        printf("--- Resposta Gerada ---\n");
    } else {
        printf("--- Texto Gerado ---\nArtificial Intelligence is transforming");
    }
    print_token(next_token);

    int cur_token = next_token;
    double t_decode_start = timer_now();
    int generated_count = 1;

    int step;
    for (step = 1; step < num_tokens_generate; step++) {
        if (cur_token == 151645 || cur_token == 151643) {
            break;
        }
        int cur_pos = prompt_len + step - 1;
        forward_token(model_bytes, cur_token, cur_pos);
        cur_token = sample_argmax(g_logits, QWEN15_VOCAB_SIZE);
        if (cur_token == 151645 || cur_token == 151643) {
            break;
        }
        print_token(cur_token);
        generated_count++;
    }

    double t_decode_total = timer_now() - t_decode_start;
    double avg_step_time = (generated_count > 1) ? (t_decode_total / (double)(generated_count - 1)) : 0.0;
    double tokens_per_sec = (avg_step_time > 0.0) ? (1.0 / avg_step_time) : 0.0;

    printf("\n--------------------\n\n");
    printf("======================================================================\n");
    printf("  RESULTADO DO MOTOR NATIVO C99 - QWEN2.5-1.5B-INSTRUCT\n");
    printf("======================================================================\n");
    printf("• Tokens gerados no decode: %d\n", generated_count);
    printf("• Tempo total de Decode:    %.3f s\n", t_decode_total);
    printf("• Tempo medio por token:    %.2f ms\n", avg_step_time * 1000.0);
    printf("• Velocidade NATIVA em C:   %.2f TOKENS/SEGUNDO!\n", tokens_per_sec);
    printf("======================================================================\n");

    UnmapViewOfFile(model_bytes);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    DWORD proc_list[2];
    if (GetConsoleProcessList(proc_list, 2) <= 1) {
        printf("\nPressione ENTER para fechar esta janela...\n");
        getchar();
    }

    return 0;
}
