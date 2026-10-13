#!/usr/bin/env python3
"""tools/export_offsets_3b.py
Extrai offsets brutos dos shards safetensors de Qwen2.5-3B-Instruct
e gera as tabelas de offsets:
  1. tools/qwen_offsets_3b_raw.h (para o quantizador C mapear os 2 shards)
  2. tools/qwen_offsets_3b_q4.h  (para o motor de inferência C99 ler o modelo quantizado)
"""

import json
import os
import struct

SHARD1_PATH = os.path.join("data", "qwen3b", "model-00001-of-00002.safetensors")
SHARD2_PATH = os.path.join("data", "qwen3b", "model-00002-of-00002.safetensors")
INDEX_PATH = os.path.join("data", "qwen3b", "model.safetensors.index.json")

RAW_HEADER = os.path.join("tools", "qwen_offsets_3b_raw.h")
Q4_HEADER = os.path.join("tools", "qwen_offsets_3b_q4.h")

VOCAB_SIZE = 151936
DIM = 2048
INTERMEDIATE_DIM = 11008
NUM_LAYERS = 36
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6


def read_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        return data_start, header_json


def main():
    print("=" * 70)
    print("  EXPORTADOR DE OFFSETS QWEN2.5-3B-INSTRUCT")
    print("=" * 70)

    if not os.path.exists(SHARD1_PATH) or not os.path.exists(SHARD2_PATH):
        print("[ERRO] Arquivos de shard safetensors não encontrados em data/qwen3b/")
        return 1

    dstart1, hdr1 = read_safetensors_header(SHARD1_PATH)
    dstart2, hdr2 = read_safetensors_header(SHARD2_PATH)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    print(f"[*] Shard 1 data_start: {dstart1}")
    print(f"[*] Shard 2 data_start: {dstart2}")

    def get_raw_loc(tensor_name):
        shard_fname = weight_map[tensor_name]
        if "00001" in shard_fname:
            shard_idx = 0
            off = dstart1 + hdr1[tensor_name]["data_offsets"][0]
        else:
            shard_idx = 1
            off = dstart2 + hdr2[tensor_name]["data_offsets"][0]
        return shard_idx, off

    # 1. Gerar qwen_offsets_3b_raw.h
    print("[*] Gerando tools/qwen_offsets_3b_raw.h...")
    emb_shard, emb_off = get_raw_loc("model.embed_tokens.weight")
    norm_shard, norm_off = get_raw_loc("model.norm.weight")

    raw_layers_code = []
    for l in range(NUM_LAYERS):
        pfx = f"model.layers.{l}."
        tensors = [
            ("norm1", pfx + "input_layernorm.weight"),
            ("norm2", pfx + "post_attention_layernorm.weight"),
            ("wq", pfx + "self_attn.q_proj.weight"),
            ("bq", pfx + "self_attn.q_proj.bias"),
            ("wk", pfx + "self_attn.k_proj.weight"),
            ("bk", pfx + "self_attn.k_proj.bias"),
            ("wv", pfx + "self_attn.v_proj.weight"),
            ("bv", pfx + "self_attn.v_proj.bias"),
            ("wo", pfx + "self_attn.o_proj.weight"),
            ("wgate", pfx + "mlp.gate_proj.weight"),
            ("wup", pfx + "mlp.up_proj.weight"),
            ("wdown", pfx + "mlp.down_proj.weight"),
        ]
        layer_fields = []
        for field, tname in tensors:
            s_idx, s_off = get_raw_loc(tname)
            layer_fields.append(f"            .{field} = {{ .shard = {s_idx}, .offset = {s_off}ULL }}")
        layer_block = f"        // Camada {l}\n        {{\n" + ",\n".join(layer_fields) + "\n        }"
        raw_layers_code.append(layer_block)

    raw_layers_str = ",\n".join(raw_layers_code)

    raw_content = f"""#ifndef QWEN_OFFSETS_3B_RAW_H
#define QWEN_OFFSETS_3B_RAW_H

#include <stdint.h>

typedef struct {{
    uint8_t shard;
    uint64_t offset;
}} RawTensorLoc;

typedef struct {{
    RawTensorLoc norm1;
    RawTensorLoc norm2;
    RawTensorLoc wq;
    RawTensorLoc bq;
    RawTensorLoc wk;
    RawTensorLoc bk;
    RawTensorLoc wv;
    RawTensorLoc bv;
    RawTensorLoc wo;
    RawTensorLoc wgate;
    RawTensorLoc wup;
    RawTensorLoc wdown;
}} RawLayerLocs;

typedef struct {{
    RawTensorLoc embed_tokens;
    RawTensorLoc norm;
    RawLayerLocs layers[{NUM_LAYERS}];
}} RawModelLocs;

static const RawModelLocs g_raw_model_locs = {{
    .embed_tokens = {{ .shard = {emb_shard}, .offset = {emb_off}ULL }},
    .norm = {{ .shard = {norm_shard}, .offset = {norm_off}ULL }},
    .layers = {{
{raw_layers_str}
    }}
}};

#endif // QWEN_OFFSETS_3B_RAW_H
"""
    with open(RAW_HEADER, "w", encoding="utf-8") as f:
        f.write(raw_content)
    print(f"[OK] {RAW_HEADER} gerado!")

    # 2. Calcular offsets determinísticos para o formato q4x4 final
    # Layout Q4x4:
    #   BlockQ4x4 = 8 bytes escalas BF16 + 64 bytes nibbles = 72 bytes por bloco de 32 colunas x 4 linhas.
    #   Tamanho de matriz rows x cols: (rows / 4) * (cols / 32) * 72 = rows * (cols / 32) * 18 bytes.
    #   Tamanho de vetor BF16: dim * 2 bytes.
    print("[*] Calculando layout q4x4 determinístico...")

    def q4x4_bytes(rows, cols):
        return rows * (cols // 32) * 18

    def bf16_bytes(dim):
        return dim * 2

    cur_offset = 0

    # embed_tokens: 151936 x 2048
    q4_embed_off = cur_offset
    cur_offset += q4x4_bytes(VOCAB_SIZE, DIM)

    # norm: 2048 BF16
    q4_norm_off = cur_offset
    cur_offset += bf16_bytes(DIM)

    q4_layers_code = []
    for l in range(NUM_LAYERS):
        lo_norm1 = cur_offset
        cur_offset += bf16_bytes(DIM)

        lo_norm2 = cur_offset
        cur_offset += bf16_bytes(DIM)

        lo_bq = cur_offset
        cur_offset += bf16_bytes(DIM)

        lo_bk = cur_offset
        cur_offset += bf16_bytes(NUM_KV_HEADS * HEAD_DIM)

        lo_bv = cur_offset
        cur_offset += bf16_bytes(NUM_KV_HEADS * HEAD_DIM)

        lo_wq = cur_offset
        cur_offset += q4x4_bytes(DIM, DIM)

        lo_wk = cur_offset
        cur_offset += q4x4_bytes(NUM_KV_HEADS * HEAD_DIM, DIM)

        lo_wv = cur_offset
        cur_offset += q4x4_bytes(NUM_KV_HEADS * HEAD_DIM, DIM)

        lo_wo = cur_offset
        cur_offset += q4x4_bytes(DIM, DIM)

        lo_wgate = cur_offset
        cur_offset += q4x4_bytes(INTERMEDIATE_DIM, DIM)

        lo_wup = cur_offset
        cur_offset += q4x4_bytes(INTERMEDIATE_DIM, DIM)

        lo_wdown = cur_offset
        cur_offset += q4x4_bytes(DIM, INTERMEDIATE_DIM)

        layer_block = f"""        // Camada {l}
        {{
            .norm1 = {lo_norm1}ULL,
            .norm2 = {lo_norm2}ULL,
            .wq = {lo_wq}ULL,
            .bq = {lo_bq}ULL,
            .wk = {lo_wk}ULL,
            .bk = {lo_bk}ULL,
            .wv = {lo_wv}ULL,
            .bv = {lo_bv}ULL,
            .wo = {lo_wo}ULL,
            .wgate = {lo_wgate}ULL,
            .wup = {lo_wup}ULL,
            .wdown = {lo_wdown}ULL,
        }}"""
        q4_layers_code.append(layer_block)

    total_q4_size = cur_offset
    print(f"[*] Tamanho final calculado do binário Q4x4: {total_q4_size} bytes ({total_q4_size / (1024*1024):.2f} MB)")

    q4_layers_str = ",\n".join(q4_layers_code)

    q4_content = f"""#ifndef QWEN_OFFSETS_3B_Q4_H
#define QWEN_OFFSETS_3B_Q4_H

#include <stdint.h>

#define QWEN3B_VOCAB_SIZE {VOCAB_SIZE}
#define QWEN3B_DIM {DIM}
#define QWEN3B_INTERMEDIATE_DIM {INTERMEDIATE_DIM}
#define QWEN3B_NUM_LAYERS {NUM_LAYERS}
#define QWEN3B_NUM_HEADS {NUM_HEADS}
#define QWEN3B_NUM_KV_HEADS {NUM_KV_HEADS}
#define QWEN3B_HEAD_DIM {HEAD_DIM}
#define QWEN3B_ROPE_THETA {ROPE_THETA:.1f}f
#define QWEN3B_RMS_EPS {RMS_EPS}f

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
}} Qwen3bLayerOffsets;

typedef struct {{
    uint64_t embed_tokens;
    uint64_t norm;
    Qwen3bLayerOffsets layers[QWEN3B_NUM_LAYERS];
}} Qwen3bModelOffsets;

static const Qwen3bModelOffsets g_qwen3b_offsets = {{
    .embed_tokens = {q4_embed_off}ULL,
    .norm = {q4_norm_off}ULL,
    .layers = {{
{q4_layers_str}
    }}
}};

#endif // QWEN_OFFSETS_3B_Q4_H
"""
    with open(Q4_HEADER, "w", encoding="utf-8") as f:
        f.write(q4_content)
    print(f"[OK] {Q4_HEADER} gerado!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    main()
