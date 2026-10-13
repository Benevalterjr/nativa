#!/usr/bin/env python3
"""tools/pack_natq4.py
Empacota modelos quantizados Q4x4 no formato autodescritivo universal .natq4 do Nativa.
Escreve o cabeçalho de 16 KB no início do arquivo e anexa os pesos.
"""

import os
import struct
import sys

HEADER_SIZE = 16384  # 16 KB


def pack_qwen_3b_to_natq4(raw_bin_path: str, out_natq4_path: str):
    import re

    hdr_c_path = os.path.join("tools", "qwen_offsets_3b_q4.h")
    if not os.path.exists(hdr_c_path):
        print(f"[ERRO] Header {hdr_c_path} nao encontrado!")
        return False

    text = open(hdr_c_path, "r", encoding="utf-8").read()

    # Extrai offsets
    m_emb = re.search(r"\.embed_tokens\s*=\s*(\d+)ULL", text)
    m_norm = re.search(r"\.norm\s*=\s*(\d+)ULL", text)
    embed_off = int(m_emb.group(1)) if m_emb else 0
    norm_off = int(m_norm.group(1)) if m_norm else 0

    layers = []
    for block in re.findall(r"// Camada \d+\s*\{(.*?)\}", text, re.S):
        d = dict((k, int(v)) for k, v in re.findall(r"\.(\w+)\s*=\s*(\d+)ULL", block))
        layers.append(d)

    print(f"[*] Lido modelo 3B: {len(layers)} camadas, embed={embed_off}, norm={norm_off}")

    # Monta o header C struct NativaModelHeader (16384 bytes)
    # Magic: "NATQ4X4\0" (8B)
    magic = b"NATQ4X4\x00"
    version = 2
    arch = 1  # Qwen2
    vocab_size = 151936
    dim = 2048
    intermediate_dim = 11008
    num_layers = 36
    num_heads = 16
    num_kv_heads = 2
    head_dim = 128
    has_attn_bias = 1
    tie_embeddings = 1
    act_type = 1  # SwiGLU
    max_seq_len = 2048
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    data_start = HEADER_SIZE
    embed_tokens_off = embed_off + data_start
    norm_off_abs = norm_off + data_start
    lm_head_off = embed_tokens_off

    hdr_bytes = bytearray(HEADER_SIZE)
    # pack initial fields
    struct.pack_into(
        "<8s13I2f4Q",
        hdr_bytes,
        0,
        magic,
        version,
        arch,
        vocab_size,
        dim,
        intermediate_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        has_attn_bias,
        tie_embeddings,
        act_type,
        max_seq_len,
        rope_theta,
        rms_norm_eps,
        embed_tokens_off,
        norm_off_abs,
        lm_head_off,
        data_start,
    )

    # Offset onde comeca o array layers: 8 + 13*4 + 2*4 + 4*8 = 100 bytes
    layer_array_start = 100
    for l_idx, l in enumerate(layers):
        layer_offset = layer_array_start + l_idx * 96
        struct.pack_into(
            "<12Q",
            hdr_bytes,
            layer_offset,
            l["norm1"] + data_start,
            l["norm2"] + data_start,
            l["wq"] + data_start,
            l["bq"] + data_start,
            l["wk"] + data_start,
            l["bk"] + data_start,
            l["wv"] + data_start,
            l["bv"] + data_start,
            l["wo"] + data_start,
            l["wgate"] + data_start,
            l["wup"] + data_start,
            l["wdown"] + data_start,
        )

    # Campos EOS da versão 2 ficam depois do array fixo de 128 camadas.
    # Qwen2.5 usa im_end como EOS e endoftext como parada adicional.
    struct.pack_into("<2I", hdr_bytes, 100 + 128 * 96, 151645, 151643)

    print(f"[*] Criando {out_natq4_path} com header autodescritivo de 16 KB...")
    chunk_size = 8 * 1024 * 1024
    with open(out_natq4_path, "wb") as out_f:
        out_f.write(hdr_bytes)
        with open(raw_bin_path, "rb") as in_f:
            while True:
                buf = in_f.read(chunk_size)
                if not buf:
                    break
                out_f.write(buf)

    total_size = os.path.getsize(out_natq4_path)
    print(f"[OK] Modelo universal criado com sucesso: {out_natq4_path} ({total_size / (1024*1024):.2f} MB)")
    return True


if __name__ == "__main__":
    raw_path = "data/qwen2.5_3b_q4x4.bin"
    natq4_path = "data/qwen2.5_3b.natq4"
    if os.path.exists(raw_path):
        pack_qwen_3b_to_natq4(raw_path, natq4_path)
    else:
        print(f"[ERRO] {raw_path} nao existe!")
