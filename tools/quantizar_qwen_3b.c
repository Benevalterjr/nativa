#define _CRT_SECURE_NO_WARNINGS 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <omp.h>

#include "qwen_offsets_3b_raw.h"
#include "qwen_offsets_3b_q4.h"

#pragma pack(push, 1)
typedef struct {
    uint16_t scale[4]; // 8 bytes BF16 scales for the 4 rows
    uint8_t  qs[64];   // 64 bytes (16 bytes each for rows 0, 1, 2, 3)
} BlockQ4x4;
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

static inline void quantize_block_32(const uint16_t* in_bf16, uint16_t* out_scale, uint8_t* out_qs) {
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
        *out_scale = 0;
        memset(out_qs, 0x88, 16);
        return;
    }

    float scale = amax / 7.0f;
    uint16_t scale_bf16 = fp32_to_bf16(scale);
    float recon_scale = bf16_to_fp32(scale_bf16);
    if (recon_scale < 1e-12f) recon_scale = scale;
    float inv_scale = 1.0f / recon_scale;

    *out_scale = scale_bf16;

    for (i = 0; i < 16; i++) {
        int q0 = (int)roundf(vals[i] * inv_scale) + 8;
        int q1 = (int)roundf(vals[i + 16] * inv_scale) + 8;

        if (q0 < 0) q0 = 0;
        if (q0 > 15) q0 = 15;
        if (q1 < 0) q1 = 0;
        if (q1 > 15) q1 = 15;

        out_qs[i] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

static void quantize_matrix_q4x4(const uint8_t* in_ptr, int rows, int cols, FILE* out_file, uint64_t expected_offset) {
    uint64_t cur_off = (uint64_t)_ftelli64(out_file);
    if (cur_off != expected_offset) {
        fprintf(stderr, "[ERRO OFFSET] Esperado %llu, atual %llu ao quantizar matriz (%d x %d)\n",
                expected_offset, cur_off, rows, cols);
        exit(1);
    }

    int nb = cols / 32;
    int num_groups = rows / 4;
    size_t out_bytes = (size_t)num_groups * nb * sizeof(BlockQ4x4);

    BlockQ4x4* out_buf = (BlockQ4x4*)malloc(out_bytes);
    if (!out_buf) {
        fprintf(stderr, "Erro ao alocar memoria para quantizar matriz (%d x %d)\n", rows, cols);
        exit(1);
    }

    const uint16_t* in_bf16 = (const uint16_t*)in_ptr;

    int g;
    #pragma omp parallel for schedule(static)
    for (g = 0; g < num_groups; g++) {
        for (int b = 0; b < nb; b++) {
            BlockQ4x4* target = out_buf + (size_t)g * nb + b;
            for (int r = 0; r < 4; r++) {
                const uint16_t* row_src = in_bf16 + (size_t)(4 * g + r) * cols + b * 32;
                quantize_block_32(row_src, &target->scale[r], target->qs + r * 16);
            }
        }
    }

    fwrite(out_buf, 1, out_bytes, out_file);
    free(out_buf);
}

static void copy_raw_bf16(const uint8_t* in_ptr, size_t num_elements, FILE* out_file, uint64_t expected_offset) {
    uint64_t cur_off = (uint64_t)_ftelli64(out_file);
    if (cur_off != expected_offset) {
        fprintf(stderr, "[ERRO OFFSET] Esperado %llu, atual %llu ao copiar BF16 (%zu elementos)\n",
                expected_offset, cur_off, num_elements);
        exit(1);
    }
    size_t bytes = num_elements * sizeof(uint16_t);
    fwrite(in_ptr, 1, bytes, out_file);
}

typedef struct {
    HANDLE hFile;
    HANDLE hMapping;
    const uint8_t* base;
} MappedShard;

static bool map_shard(const char* path, MappedShard* shard) {
    shard->hFile = INVALID_HANDLE_VALUE;
    shard->hMapping = NULL;
    shard->base = NULL;
    shard->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (shard->hFile == INVALID_HANDLE_VALUE) {
        printf("[ERRO] Nao foi possivel abrir shard %s (Erro: %lu)\n", path, GetLastError());
        return false;
    }
    shard->hMapping = CreateFileMappingA(shard->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!shard->hMapping) {
        printf("[ERRO] Nao foi possivel criar mapeamento para %s\n", path);
        CloseHandle(shard->hFile);
        shard->hFile = INVALID_HANDLE_VALUE;
        return false;
    }
    shard->base = (const uint8_t*)MapViewOfFile(shard->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!shard->base) {
        printf("[ERRO] Nao foi possivel mapear visualizacao de %s\n", path);
        CloseHandle(shard->hMapping);
        shard->hMapping = NULL;
        CloseHandle(shard->hFile);
        shard->hFile = INVALID_HANDLE_VALUE;
        return false;
    }
    return true;
}

static void unmap_shard(MappedShard* shard) {
    if (shard->base) {
        UnmapViewOfFile(shard->base);
        shard->base = NULL;
    }
    if (shard->hMapping) {
        CloseHandle(shard->hMapping);
        shard->hMapping = NULL;
    }
    if (shard->hFile && shard->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(shard->hFile);
        shard->hFile = INVALID_HANDLE_VALUE;
    }
}

int main(void) {
    const char* shard1_path = "data/qwen3b/model-00001-of-00002.safetensors";
    const char* shard2_path = "data/qwen3b/model-00002-of-00002.safetensors";
    const char* out_model_path = "data/qwen2.5_3b_q4x4.bin";

    printf("====================================================================\n");
    printf("  NATIVA QUANTIZADOR DIRETO Q4x4 - QWEN2.5-3B-INSTRUCT\n");
    printf("  Shards Entrada : %s + %s\n", shard1_path, shard2_path);
    printf("  Saida INT4     : %s (~1.62 GB)\n", out_model_path);
    printf("====================================================================\n");

    MappedShard s1 = { INVALID_HANDLE_VALUE, NULL, NULL };
    MappedShard s2 = { INVALID_HANDLE_VALUE, NULL, NULL };
    printf("[*] Mapeando Shard 1 (3.97 GB)...\n");
    if (!map_shard(shard1_path, &s1)) {
        unmap_shard(&s1);
        return 1;
    }

    printf("[*] Mapeando Shard 2 (2.20 GB)...\n");
    if (!map_shard(shard2_path, &s2)) {
        unmap_shard(&s2);
        unmap_shard(&s1);
        return 1;
    }

    const uint8_t* shards[2] = { s1.base, s2.base };

    FILE* out_file = fopen(out_model_path, "wb");
    if (!out_file) {
        printf("[ERRO] Nao foi possivel criar %s\n", out_model_path);
        unmap_shard(&s1);
        unmap_shard(&s2);
        return 1;
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    // 1. Quantizar embed_tokens (151936 x 2048) diretamente para BlockQ4x4
    printf("[1/3] Quantizando embedding tokens & lm_head (151,936 x 2,048)...\n");
    const uint8_t* emb_src = shards[g_raw_model_locs.embed_tokens.shard] + g_raw_model_locs.embed_tokens.offset;
    quantize_matrix_q4x4(emb_src, QWEN3B_VOCAB_SIZE, QWEN3B_DIM, out_file, g_qwen3b_offsets.embed_tokens);
    printf("      -> Embed tokens gravados no offset: %llu bytes (~167 MB)\n", g_qwen3b_offsets.embed_tokens);

    // 2. Copiar model.norm (2048 BF16)
    const uint8_t* norm_src = shards[g_raw_model_locs.norm.shard] + g_raw_model_locs.norm.offset;
    copy_raw_bf16(norm_src, QWEN3B_DIM, out_file, g_qwen3b_offsets.norm);

    // 3. Processar as 36 camadas do Transformer
    printf("[2/3] Quantizando as 36 camadas do Transformer diretamente em Q4x4...\n");
    for (int l = 0; l < QWEN3B_NUM_LAYERS; l++) {
        const RawLayerLocs* raw_l = &g_raw_model_locs.layers[l];
        const Qwen3bLayerOffsets* q4_l = &g_qwen3b_offsets.layers[l];

        // Layernorms BF16
        copy_raw_bf16(shards[raw_l->norm1.shard] + raw_l->norm1.offset, QWEN3B_DIM, out_file, q4_l->norm1);
        copy_raw_bf16(shards[raw_l->norm2.shard] + raw_l->norm2.offset, QWEN3B_DIM, out_file, q4_l->norm2);

        // Biases BF16
        copy_raw_bf16(shards[raw_l->bq.shard] + raw_l->bq.offset, QWEN3B_DIM, out_file, q4_l->bq);
        copy_raw_bf16(shards[raw_l->bk.shard] + raw_l->bk.offset, QWEN3B_NUM_KV_HEADS * QWEN3B_HEAD_DIM, out_file, q4_l->bk);
        copy_raw_bf16(shards[raw_l->bv.shard] + raw_l->bv.offset, QWEN3B_NUM_KV_HEADS * QWEN3B_HEAD_DIM, out_file, q4_l->bv);

        // Atencao Q4x4
        quantize_matrix_q4x4(shards[raw_l->wq.shard] + raw_l->wq.offset, QWEN3B_DIM, QWEN3B_DIM, out_file, q4_l->wq);
        quantize_matrix_q4x4(shards[raw_l->wk.shard] + raw_l->wk.offset, QWEN3B_NUM_KV_HEADS * QWEN3B_HEAD_DIM, QWEN3B_DIM, out_file, q4_l->wk);
        quantize_matrix_q4x4(shards[raw_l->wv.shard] + raw_l->wv.offset, QWEN3B_NUM_KV_HEADS * QWEN3B_HEAD_DIM, QWEN3B_DIM, out_file, q4_l->wv);
        quantize_matrix_q4x4(shards[raw_l->wo.shard] + raw_l->wo.offset, QWEN3B_DIM, QWEN3B_DIM, out_file, q4_l->wo);

        // MLP SwiGLU Q4x4
        quantize_matrix_q4x4(shards[raw_l->wgate.shard] + raw_l->wgate.offset, QWEN3B_INTERMEDIATE_DIM, QWEN3B_DIM, out_file, q4_l->wgate);
        quantize_matrix_q4x4(shards[raw_l->wup.shard] + raw_l->wup.offset, QWEN3B_INTERMEDIATE_DIM, QWEN3B_DIM, out_file, q4_l->wup);
        quantize_matrix_q4x4(shards[raw_l->wdown.shard] + raw_l->wdown.offset, QWEN3B_DIM, QWEN3B_INTERMEDIATE_DIM, out_file, q4_l->wdown);

        if ((l + 1) % 6 == 0 || l == QWEN3B_NUM_LAYERS - 1) {
            printf("      -> Camadas %2d/36 quantizadas...\n", l + 1);
        }
    }

    uint64_t total_out_size = (uint64_t)_ftelli64(out_file);
    fclose(out_file);

    unmap_shard(&s1);
    unmap_shard(&s2);

    QueryPerformanceCounter(&t1);
    double total_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;

    printf("\n====================================================================\n");
    printf("[SUCESSO] QUANTIZACAO DIRETA Q4x4 CONCLUIDA COM EXITO!\n");
    printf("  Tamanho final : %llu bytes (%.2f MB)\n", total_out_size, (double)total_out_size / (1024.0 * 1024.0));
    printf("  Tempo total   : %.2f segundos\n", total_sec);
    printf("====================================================================\n");

    return 0;
}
