#ifndef NATQ4_FORMAT_H
#define NATQ4_FORMAT_H

#include <stdint.h>

#define NATQ4_MAGIC "NATQ4X4"
#define NATQ4_VERSION 3
#define NATQ4_MAX_LAYERS 128
#define NATQ4_HEADER_SIZE 16384 // 16 KB page-aligned header

typedef enum {
    NAT_ARCH_QWEN2   = 1,
    NAT_ARCH_LLAMA   = 2,
    NAT_ARCH_MISTRAL = 3,
    NAT_ARCH_GEMMA   = 4,
    NAT_ARCH_PHI     = 5,
    NAT_ARCH_QWEN3   = 6,
} NativaArchType;

typedef enum {
    NAT_ACT_SWIGLU = 1,
    NAT_ACT_GEGLU  = 2,
    NAT_ACT_GELU   = 3,
    NAT_ACT_RELU   = 4,
} NativaActType;

typedef struct {
    uint64_t norm1;
    uint64_t norm2;
    uint64_t wq;
    uint64_t bq;     // 0 se nao houver bias
    uint64_t wk;
    uint64_t bk;     // 0 se nao houver bias
    uint64_t wv;
    uint64_t bv;     // 0 se nao houver bias
    uint64_t wo;
    uint64_t wgate;
    uint64_t wup;
    uint64_t wdown;
} NativaLayerOffsets;

typedef struct {
    uint64_t q_norm;
    uint64_t k_norm;
} NativaQkNormOffsets;

#pragma pack(push, 1)
typedef struct {
    char     magic[8];            // "NATQ4X4\0" (8B)
    uint32_t version;             // 1 legado, 2 EOS, 3 Q/K norms
    uint32_t arch;                // NativaArchType
    uint32_t vocab_size;          // 151936, 32000, 128256, etc.
    uint32_t dim;                 // hidden_size (ex: 2048, 4096)
    uint32_t intermediate_dim;    // mlp intermediate size (ex: 11008, 14336)
    uint32_t num_layers;          // 28, 32, 36, etc.
    uint32_t num_heads;           // Q heads (ex: 12, 16, 32)
    uint32_t num_kv_heads;        // KV heads (ex: 2, 8, 32)
    uint32_t head_dim;            // 128, 64, etc.
    uint32_t has_attn_bias;       // 1 para Qwen, 0 para LLaMA
    uint32_t tie_embeddings;      // 1 se lm_head usa embed_tokens
    uint32_t act_type;            // NativaActType
    uint32_t max_seq_len;         // 2048, 4096, 8192
    float    rope_theta;          // 1000000.0f, 10000.0f
    float    rms_norm_eps;        // 1e-6f, 1e-5f
    uint64_t embed_tokens_off;
    uint64_t norm_off;
    uint64_t lm_head_off;         // se nao for tied
    uint64_t data_start_off;      // 16384 (16 KB)
    NativaLayerOffsets layers[NATQ4_MAX_LAYERS];
    uint32_t eos_token_id;        // UINT32_MAX se ausente
    uint32_t eos_token_id_2;      // segundo token de parada opcional
    NativaQkNormOffsets qk_norms[NATQ4_MAX_LAYERS];
    uint32_t tokenizer_vocab_size; // IDs válidos; pode ser menor que vocab_size por padding
    uint8_t  reserved[NATQ4_HEADER_SIZE - 14448];
} NativaModelHeader;
#pragma pack(pop)

#define NATIVA_BUNDLE_MAGIC 0x4E4154495641464CULL // "NATIVAF\0"
#define NATIVA_BUNDLE_ALIGN 65536                  // 64 KB page/allocation alignment

#pragma pack(push, 1)
typedef struct {
    uint64_t magic;         // NATIVA_BUNDLE_MAGIC (8B)
    uint64_t engine_size;   // Tamanho original do binário executável (8B)
    uint64_t vocab_offset;  // Offset onde o vocabulário começa (8B)
    uint64_t vocab_size;    // Tamanho do vocabulário em bytes (8B)
    uint64_t model_offset;  // Offset onde o .natq4 começa (8B)
    uint64_t model_size;    // Tamanho do .natq4 em bytes (8B)
    uint32_t version;       // Versão do formato do bundle (1) (4B)
    uint32_t flags;         // Flags: 0 = chat interativo, 1 = benchmark (4B)
    uint8_t  reserved[8];   // Zeros para alinhamento e expansão (8B) -> total = exatamente 64B
} NativaBundleTrailer;
#pragma pack(pop)

#endif // NATQ4_FORMAT_H
