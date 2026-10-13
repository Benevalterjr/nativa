#define _CRT_SECURE_NO_WARNINGS 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <omp.h>

#include "qwen_offsets_1.5b.h"

#pragma pack(push, 1)
typedef struct {
    uint16_t scale; // BF16 scale
    uint8_t  qs[16]; // 16 bytes for 32 nibbles
} BlockQ4;
#pragma pack(pop)

static inline float bf16_to_fp32(uint16_t v) {
    union { uint32_t u; float f; } conv;
    conv.u = ((uint32_t)v) << 16;
    return conv.f;
}

static inline uint16_t fp32_to_bf16(float f) {
    union { uint32_t u; float f; } conv;
    conv.f = f;
    return (uint16_t)(conv.u >> 16);
}

static inline void quantize_block_32(const uint16_t* in_bf16, BlockQ4* out_block) {
    float vals[32];
    float amax = 0.0f;
    int i;
    for (i = 0; i < 32; i++) {
        float f = bf16_to_fp32(in_bf16[i]);
        vals[i] = f;
        float af = fabsf(f);
        if (af > amax) amax = af;
    }

    if (amax < 1e-12f) {
        out_block->scale = 0;
        memset(out_block->qs, 0x88, 16);
        return;
    }

    float scale = amax / 7.0f;
    uint16_t scale_bf16 = fp32_to_bf16(scale);
    float recon_scale = bf16_to_fp32(scale_bf16);
    if (recon_scale < 1e-12f) recon_scale = scale;
    float inv_scale = 1.0f / recon_scale;

    out_block->scale = scale_bf16;

    for (i = 0; i < 16; i++) {
        int q0 = (int)roundf(vals[i] * inv_scale) + 8;
        int q1 = (int)roundf(vals[i + 16] * inv_scale) + 8;

        if (q0 < 0) q0 = 0;
        if (q0 > 15) q0 = 15;
        if (q1 < 0) q1 = 0;
        if (q1 > 15) q1 = 15;

        out_block->qs[i] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

static void quantize_matrix(const uint8_t* in_base, uint64_t in_offset, int rows, int cols, FILE* out_file, uint64_t* out_offset) {
    int num_blocks = cols / 32;
    size_t out_bytes = (size_t)rows * num_blocks * sizeof(BlockQ4);
    BlockQ4* out_buf = (BlockQ4*)malloc(out_bytes);
    if (!out_buf) {
        fprintf(stderr, "Erro ao alocar memoria para quantizar matriz (%d x %d)\n", rows, cols);
        exit(1);
    }

    const uint16_t* in_bf16 = (const uint16_t*)(in_base + in_offset);

    int r;
    #pragma omp parallel for schedule(static)
    for (r = 0; r < rows; r++) {
        const uint16_t* row_src = in_bf16 + (size_t)r * cols;
        BlockQ4* row_dst = out_buf + (size_t)r * num_blocks;
        for (int b = 0; b < num_blocks; b++) {
            quantize_block_32(row_src + b * 32, &row_dst[b]);
        }
    }

    *out_offset = (uint64_t)_ftelli64(out_file);
    fwrite(out_buf, 1, out_bytes, out_file);
    free(out_buf);
}

static void copy_raw_bf16(const uint8_t* in_base, uint64_t in_offset, size_t num_elements, FILE* out_file, uint64_t* out_offset) {
    size_t bytes = num_elements * sizeof(uint16_t);
    *out_offset = (uint64_t)_ftelli64(out_file);
    fwrite(in_base + in_offset, 1, bytes, out_file);
}

int main(void) {
    const char* in_model_path = "data/qwen2.5_1.5b_instruct.safetensors";
    const char* out_model_path = "data/qwen2.5_1.5b_q4.bin";
    const char* out_header_path = "tools/qwen_offsets_1.5b_q4.h";

    printf("====================================================================\n");
    printf("  NATIVA QUANTIZADOR INT4 (Q4_0) — QWEN2.5-1.5B-INSTRUCT\n");
    printf("  Entrada : %s (3.08 GB BF16)\n", in_model_path);
    printf("  Saida   : %s (~828 MB INT4)\n", out_model_path);
    printf("====================================================================\n");

    // 1. Abrir e mapear modelo de entrada com Win32
    HANDLE hFile = CreateFileA(in_model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("[ERRO] Nao foi possivel abrir %s (Codigo: %lu)\n", in_model_path, GetLastError());
        return 1;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        printf("[ERRO] Nao foi possivel criar mapeamento de arquivo\n");
        CloseHandle(hFile);
        return 1;
    }

    const uint8_t* model_bytes = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!model_bytes) {
        printf("[ERRO] Nao foi possivel mapear o arquivo na memoria\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }

    FILE* out_file = fopen(out_model_path, "wb");
    if (!out_file) {
        printf("[ERRO] Nao foi possivel criar arquivo de saida %s\n", out_model_path);
        UnmapViewOfFile(model_bytes);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 1;
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    Qwen15ModelOffsets q4_offsets;
    memset(&q4_offsets, 0, sizeof(q4_offsets));

    // 2. Quantizar embed_tokens (151936 x 1536)
    printf("[1/3] Quantizando embedding tokens & lm_head (151,936 x 1,536)...\n");
    quantize_matrix(model_bytes, g_qwen15_offsets.embed_tokens, QWEN15_VOCAB_SIZE, QWEN15_DIM, out_file, &q4_offsets.embed_tokens);
    printf("      -> Embed tokens gravados no offset: %llu bytes (~125 MB)\n", q4_offsets.embed_tokens);

    // 3. Copiar model.norm (1536 BF16)
    copy_raw_bf16(model_bytes, g_qwen15_offsets.norm, QWEN15_DIM, out_file, &q4_offsets.norm);

    // 4. Quantizar 28 camadas do Transformer
    printf("[2/3] Quantizando as 28 camadas do Transformer em INT4...\n");
    for (int l = 0; l < QWEN15_NUM_LAYERS; l++) {
        const Qwen15LayerOffsets* in_l = &g_qwen15_offsets.layers[l];
        Qwen15LayerOffsets* out_l = &q4_offsets.layers[l];

        // Layernorms em BF16 original
        copy_raw_bf16(model_bytes, in_l->norm1, QWEN15_DIM, out_file, &out_l->norm1);
        copy_raw_bf16(model_bytes, in_l->norm2, QWEN15_DIM, out_file, &out_l->norm2);

        // Biases da atencao em BF16 original
        copy_raw_bf16(model_bytes, in_l->bq, QWEN15_DIM, out_file, &out_l->bq);
        copy_raw_bf16(model_bytes, in_l->bk, QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM, out_file, &out_l->bk);
        copy_raw_bf16(model_bytes, in_l->bv, QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM, out_file, &out_l->bv);

        // Projeções da Atenção em INT4
        quantize_matrix(model_bytes, in_l->wq, QWEN15_DIM, QWEN15_DIM, out_file, &out_l->wq);
        quantize_matrix(model_bytes, in_l->wk, QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM, QWEN15_DIM, out_file, &out_l->wk);
        quantize_matrix(model_bytes, in_l->wv, QWEN15_NUM_KV_HEADS * QWEN15_HEAD_DIM, QWEN15_DIM, out_file, &out_l->wv);
        quantize_matrix(model_bytes, in_l->wo, QWEN15_DIM, QWEN15_DIM, out_file, &out_l->wo);

        // MLP SwiGLU em INT4
        quantize_matrix(model_bytes, in_l->wgate, QWEN15_INTERMEDIATE_DIM, QWEN15_DIM, out_file, &out_l->wgate);
        quantize_matrix(model_bytes, in_l->wup, QWEN15_INTERMEDIATE_DIM, QWEN15_DIM, out_file, &out_l->wup);
        quantize_matrix(model_bytes, in_l->wdown, QWEN15_DIM, QWEN15_INTERMEDIATE_DIM, out_file, &out_l->wdown);

        if ((l + 1) % 4 == 0 || l == QWEN15_NUM_LAYERS - 1) {
            printf("      -> Camadas %2d/28 quantizadas...\n", l + 1);
        }
    }

    uint64_t total_out_size = (uint64_t)_ftelli64(out_file);
    fclose(out_file);

    UnmapViewOfFile(model_bytes);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    QueryPerformanceCounter(&t1);
    double total_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;

    printf("[3/3] Exportando header C %s...\n", out_header_path);
    FILE* h_file = fopen(out_header_path, "w");
    if (!h_file) {
        printf("[ERRO] Nao foi possivel escrever header %s\n", out_header_path);
        return 1;
    }

    fprintf(h_file, "#ifndef QWEN_OFFSETS_1_5B_Q4_H\n#define QWEN_OFFSETS_1_5B_Q4_H\n\n#include <stdint.h>\n\n");
    fprintf(h_file, "#define QWEN15_VOCAB_SIZE %d\n", QWEN15_VOCAB_SIZE);
    fprintf(h_file, "#define QWEN15_DIM %d\n", QWEN15_DIM);
    fprintf(h_file, "#define QWEN15_INTERMEDIATE_DIM %d\n", QWEN15_INTERMEDIATE_DIM);
    fprintf(h_file, "#define QWEN15_NUM_LAYERS %d\n", QWEN15_NUM_LAYERS);
    fprintf(h_file, "#define QWEN15_NUM_HEADS %d\n", QWEN15_NUM_HEADS);
    fprintf(h_file, "#define QWEN15_NUM_KV_HEADS %d\n", QWEN15_NUM_KV_HEADS);
    fprintf(h_file, "#define QWEN15_HEAD_DIM %d\n", QWEN15_HEAD_DIM);
    fprintf(h_file, "#define QWEN15_ROPE_THETA %.1ff\n", QWEN15_ROPE_THETA);
    fprintf(h_file, "#define QWEN15_RMS_EPS 1e-6f\n\n");

    fprintf(h_file, "typedef struct {\n");
    fprintf(h_file, "    uint64_t norm1;\n    uint64_t norm2;\n");
    fprintf(h_file, "    uint64_t wq;\n    uint64_t bq;\n");
    fprintf(h_file, "    uint64_t wk;\n    uint64_t bk;\n");
    fprintf(h_file, "    uint64_t wv;\n    uint64_t bv;\n");
    fprintf(h_file, "    uint64_t wo;\n    uint64_t wgate;\n");
    fprintf(h_file, "    uint64_t wup;\n    uint64_t wdown;\n");
    fprintf(h_file, "} Qwen15LayerOffsets;\n\n");

    fprintf(h_file, "typedef struct {\n");
    fprintf(h_file, "    uint64_t embed_tokens;\n    uint64_t norm;\n");
    fprintf(h_file, "    Qwen15LayerOffsets layers[QWEN15_NUM_LAYERS];\n");
    fprintf(h_file, "} Qwen15ModelOffsets;\n\n");

    fprintf(h_file, "static const Qwen15ModelOffsets g_qwen15_offsets = {\n");
    fprintf(h_file, "    .embed_tokens = %lluULL,\n", q4_offsets.embed_tokens);
    fprintf(h_file, "    .norm = %lluULL,\n", q4_offsets.norm);
    fprintf(h_file, "    .layers = {\n");
    for (int l = 0; l < QWEN15_NUM_LAYERS; l++) {
        const Qwen15LayerOffsets* lo = &q4_offsets.layers[l];
        fprintf(h_file, "        // Camada %d\n", l);
        fprintf(h_file, "        {\n");
        fprintf(h_file, "            .norm1 = %lluULL,\n", lo->norm1);
        fprintf(h_file, "            .norm2 = %lluULL,\n", lo->norm2);
        fprintf(h_file, "            .wq = %lluULL,\n", lo->wq);
        fprintf(h_file, "            .bq = %lluULL,\n", lo->bq);
        fprintf(h_file, "            .wk = %lluULL,\n", lo->wk);
        fprintf(h_file, "            .bk = %lluULL,\n", lo->bk);
        fprintf(h_file, "            .wv = %lluULL,\n", lo->wv);
        fprintf(h_file, "            .bv = %lluULL,\n", lo->bv);
        fprintf(h_file, "            .wo = %lluULL,\n", lo->wo);
        fprintf(h_file, "            .wgate = %lluULL,\n", lo->wgate);
        fprintf(h_file, "            .wup = %lluULL,\n", lo->wup);
        fprintf(h_file, "            .wdown = %lluULL,\n", lo->wdown);
        fprintf(h_file, "        }%s\n", (l == QWEN15_NUM_LAYERS - 1) ? "" : ",");
    }
    fprintf(h_file, "    }\n};\n\n#endif // QWEN_OFFSETS_1_5B_Q4_H\n");
    fclose(h_file);

    printf("\n====================================================================\n");
    printf("[SUCESSO] QUANTIZACAO CONCLUIDA COM EXITO!\n");
    printf("  Tamanho final : %llu bytes (%.2f MB)\n", total_out_size, (double)total_out_size / (1024.0 * 1024.0));
    printf("  Tempo total   : %.2f segundos\n", total_sec);
    printf("====================================================================\n");

    return 0;
}
