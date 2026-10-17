#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    int32_t n_tokens;
    int32_t * token;
    float * embd;
    int32_t * pos;
    int32_t * n_seq_id;
    int32_t ** seq_id;
    int8_t * logits;
} llama_batch;

typedef void* (*pfn_llama_backend_init)(void);
typedef void* (*pfn_llama_model_load_from_file)(const char*, void*);
typedef void* (*pfn_llama_init_from_model)(void*, void*);
typedef void (*pfn_llama_model_default_params)(void*);
typedef void (*pfn_llama_context_default_params)(void*);
typedef llama_batch (*pfn_llama_batch_init)(int32_t, int32_t, int32_t);
typedef void (*pfn_llama_batch_free)(llama_batch);
typedef int32_t (*pfn_llama_decode)(void*, llama_batch);
typedef float* (*pfn_llama_get_logits_ith)(void*, int32_t);
typedef int32_t (*pfn_llama_n_vocab)(void*);

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "data/qwen3_0.6b_q4_k_m.gguf";
    const char* tokens_path = (argc > 2) ? argv[2] : "data/wiki_512_ids.txt";
    int max_tokens = (argc > 3) ? atoi(argv[3]) : 64;

    char abs_dir[MAX_PATH];
    GetFullPathNameA("tools/llama_cpp", MAX_PATH, abs_dir, NULL);
    SetDllDirectoryA(abs_dir);

    char abs_dll[MAX_PATH];
    GetFullPathNameA("tools/llama_cpp/llama.dll", MAX_PATH, abs_dll, NULL);
    HMODULE hDll = LoadLibraryA(abs_dll);
    if (!hDll) {
        fprintf(stderr, "Erro ao carregar tools/llama_cpp/llama.dll (err=%lu)\n", GetLastError());
        return 1;
    }

    pfn_llama_backend_init llama_backend_init = (pfn_llama_backend_init)GetProcAddress(hDll, "llama_backend_init");
    pfn_llama_model_default_params llama_model_default_params = (pfn_llama_model_default_params)GetProcAddress(hDll, "llama_model_default_params");
    pfn_llama_model_load_from_file llama_model_load_from_file = (pfn_llama_model_load_from_file)GetProcAddress(hDll, "llama_model_load_from_file");
    pfn_llama_context_default_params llama_context_default_params = (pfn_llama_context_default_params)GetProcAddress(hDll, "llama_context_default_params");
    pfn_llama_init_from_model llama_init_from_model = (pfn_llama_init_from_model)GetProcAddress(hDll, "llama_init_from_model");
    pfn_llama_batch_init llama_batch_init = (pfn_llama_batch_init)GetProcAddress(hDll, "llama_batch_init");
    pfn_llama_batch_free llama_batch_free = (pfn_llama_batch_free)GetProcAddress(hDll, "llama_batch_free");
    pfn_llama_decode llama_decode = (pfn_llama_decode)GetProcAddress(hDll, "llama_decode");
    pfn_llama_get_logits_ith llama_get_logits_ith = (pfn_llama_get_logits_ith)GetProcAddress(hDll, "llama_get_logits_ith");
    pfn_llama_n_vocab llama_n_vocab = (pfn_llama_n_vocab)GetProcAddress(hDll, "llama_n_vocab");

    char abs_ggml[MAX_PATH];
    GetFullPathNameA("tools/llama_cpp/ggml.dll", MAX_PATH, abs_ggml, NULL);
    HMODULE hGgml = LoadLibraryA(abs_ggml);
    if (hGgml) {
        typedef void (*pfn_ggml_backend_load_all)(void);
        typedef void (*pfn_ggml_backend_load_all_from_path)(const char*);
        pfn_ggml_backend_load_all ggml_load_all = (pfn_ggml_backend_load_all)GetProcAddress(hGgml, "ggml_backend_load_all");
        pfn_ggml_backend_load_all_from_path ggml_load_path = (pfn_ggml_backend_load_all_from_path)GetProcAddress(hGgml, "ggml_backend_load_all_from_path");
        if (ggml_load_path) {
            ggml_load_path(abs_dir);
        } else if (ggml_load_all) {
            ggml_load_all();
        }
    }

    llama_backend_init();

    // Model params: get default params buffer
    // On x64, large structs returned by value take a hidden pointer as first arg
    char mparams[512] = {0};
    typedef void (*fn_mparams)(char*);
    ((fn_mparams)llama_model_default_params)(mparams);

    typedef void* (*fn_load)(const char*, char*);
    void* model = ((fn_load)llama_model_load_from_file)(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Erro ao carregar modelo: %s\n", model_path);
        return 1;
    }

    char cparams[1024] = {0};
    typedef void (*fn_cparams)(char*);
    ((fn_cparams)llama_context_default_params)(cparams);

    // Set n_ctx in cparams:
    // In llama_context_params, uint32_t n_ctx is typically offset 0 or 8.
    // Let's set n_ctx to 2048 at first uint32 fields
    uint32_t* u32 = (uint32_t*)cparams;
    u32[0] = 2048; // n_ctx

    typedef void* (*fn_init_ctx)(void*, char*);
    void* ctx = ((fn_init_ctx)llama_init_from_model)(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Erro ao inicializar contexto llama\n");
        return 1;
    }

    int n_vocab = llama_n_vocab(model);
    printf("llama.cpp carregado com sucesso! n_vocab=%d\n", n_vocab);

    // Ler tokens
    FILE* f = fopen(tokens_path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir tokens: %s\n", tokens_path);
        return 1;
    }
    int tokens[4096] = {0};
    int count = 0;
    while (count < 4096 && fscanf(f, "%d", &tokens[count]) == 1) {
        count++;
    }
    fclose(f);

    if (count < 2) {
        fprintf(stderr, "Erro: Arquivo contem tokens insuficientes para avaliacao (count=%d)\n", count);
        return 1;
    }

    if (max_tokens > count) max_tokens = count;
    printf("Tokens carregados: %d | Avaliando primeiros: %d\n", count, max_tokens);

    // Batch de 1 token
    typedef void (*fn_batch_init)(llama_batch*, int32_t, int32_t, int32_t);
    llama_batch batch;
    memset(&batch, 0, sizeof(batch));
    ((fn_batch_init)llama_batch_init)(&batch, 1, 0, 1);

    for (int t = 0; t < max_tokens - 1 && t + 1 < count; t++) {
        batch.n_tokens = 1;
        batch.token[0] = tokens[t];
        batch.pos[0] = t;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        int res = llama_decode(ctx, batch);
        if (res != 0) {
            fprintf(stderr, "Erro no llama_decode na posicao %d: %d\n", t, res);
            break;
        }

        float* logits = llama_get_logits_ith(ctx, 0);
        int target = tokens[t + 1];

        // Softmax
        float max_l = -1e30f;
        for (int v = 0; v < n_vocab; v++) {
            if (logits[v] > max_l) max_l = logits[v];
        }
        double sum_exp = 0.0;
        int best_v = 0;
        for (int v = 0; v < n_vocab; v++) {
            double ev = exp((double)(logits[v] - max_l));
            sum_exp += ev;
            if (logits[v] > logits[best_v]) best_v = v;
        }
        double target_l = (target >= 0 && target < n_vocab) ? (double)logits[target] : -1e30;
        double log_prob = target_l - (double)max_l - log(sum_exp);
        double nll = -log_prob;

        printf("[NLL_LLAMA] pos=%3d input=%7d target=%7d nll=%.4f best=%7d\n",
               t, tokens[t], target, nll, best_v);
        fflush(stdout);
    }

    if (llama_batch_free) {
        llama_batch_free(batch);
    }

    return 0;
}
