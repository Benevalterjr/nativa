import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa

MODEL_FILE = os.path.join("data", "qwen2.5_1.5b_instruct.safetensors")
HEADER_FILE = os.path.join("tools", "qwen_offsets_1.5b.h")

def export_offsets():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERRO] Arquivo do modelo não encontrado: {MODEL_FILE}")
        return False

    print(f"[*] Lendo cabeçalho safetensors de {MODEL_FILE}...")
    data_start, hdr = nativa.safetensors_read_header(MODEL_FILE)
    print(f"    -> data_start: {data_start}")

    embed_off = data_start + hdr["model.embed_tokens.weight"]["data_offsets"][0]
    norm_off = data_start + hdr["model.norm.weight"]["data_offsets"][0]

    num_layers = 28
    layers_code = []
    for l in range(num_layers):
        prefix = f"model.layers.{l}."
        n1 = data_start + hdr[prefix + "input_layernorm.weight"]["data_offsets"][0]
        n2 = data_start + hdr[prefix + "post_attention_layernorm.weight"]["data_offsets"][0]
        wq = data_start + hdr[prefix + "self_attn.q_proj.weight"]["data_offsets"][0]
        bq = data_start + hdr[prefix + "self_attn.q_proj.bias"]["data_offsets"][0]
        wk = data_start + hdr[prefix + "self_attn.k_proj.weight"]["data_offsets"][0]
        bk = data_start + hdr[prefix + "self_attn.k_proj.bias"]["data_offsets"][0]
        wv = data_start + hdr[prefix + "self_attn.v_proj.weight"]["data_offsets"][0]
        bv = data_start + hdr[prefix + "self_attn.v_proj.bias"]["data_offsets"][0]
        wo = data_start + hdr[prefix + "self_attn.o_proj.weight"]["data_offsets"][0]
        wgate = data_start + hdr[prefix + "mlp.gate_proj.weight"]["data_offsets"][0]
        wup = data_start + hdr[prefix + "mlp.up_proj.weight"]["data_offsets"][0]
        wdown = data_start + hdr[prefix + "mlp.down_proj.weight"]["data_offsets"][0]
        layers_code.append(f"""        // Camada {l}
        {{
            .norm1 = {n1}ULL,
            .norm2 = {n2}ULL,
            .wq = {wq}ULL,
            .bq = {bq}ULL,
            .wk = {wk}ULL,
            .bk = {bk}ULL,
            .wv = {wv}ULL,
            .bv = {bv}ULL,
            .wo = {wo}ULL,
            .wgate = {wgate}ULL,
            .wup = {wup}ULL,
            .wdown = {wdown}ULL,
        }}""")

    layers_str = ",\n".join(layers_code)

    h_content = f"""#ifndef QWEN_OFFSETS_1_5B_H
#define QWEN_OFFSETS_1_5B_H

#include <stdint.h>

#define QWEN15_VOCAB_SIZE 151936
#define QWEN15_DIM 1536
#define QWEN15_INTERMEDIATE_DIM 8960
#define QWEN15_NUM_LAYERS 28
#define QWEN15_NUM_HEADS 12
#define QWEN15_NUM_KV_HEADS 2
#define QWEN15_HEAD_DIM 128
#define QWEN15_ROPE_THETA 1000000.0f
#define QWEN15_RMS_EPS 1e-6f

typedef struct {{
    uint64_t norm1;
    uint64_t norm2;
    uint64_t wq;
    uint64_t bq;
    uint64_t wk;
    uint64_t bk;
    uint64_t wv;
    uint64_t bv;
    uint64_t wo;
    uint64_t wgate;
    uint64_t wup;
    uint64_t wdown;
}} Qwen15LayerOffsets;

typedef struct {{
    uint64_t embed_tokens;
    uint64_t norm;
    Qwen15LayerOffsets layers[QWEN15_NUM_LAYERS];
}} Qwen15ModelOffsets;

static const Qwen15ModelOffsets g_qwen15_offsets = {{
    .embed_tokens = {embed_off}ULL,
    .norm = {norm_off}ULL,
    .layers = {{
{layers_str}
    }}
}};

#endif // QWEN_OFFSETS_1_5B_H
"""

    with open(HEADER_FILE, "w", encoding="utf-8") as f:
        f.write(h_content)

    print(f"[OK] Header de offsets para 1.5B gerado com sucesso em {HEADER_FILE}!")
    return True


if __name__ == "__main__":
    export_offsets()
