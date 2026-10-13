#!/usr/bin/env python3
"""
tools/teste_geracao_qwen.py
Demonstracao de geracao de texto comparando Qwen2.5-0.5B:
1. Original (Pesos BF16/FP32 oficiais da Alibaba)
2. BitNet b1.58 (Pesos quantizados para {-1, 0, 1} com AbsMean Scaling)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import struct
import numpy as np
from nativa import Tokenizer

def load_qwen_header(path):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len).decode("utf-8"))
    return header, hdr_len

def load_tensor(f, header, hdr_len, name):
    info = header[name]
    offset_start, offset_end = info["data_offsets"]
    f.seek(8 + hdr_len + offset_start)
    raw = f.read(offset_end - offset_start)
    dtype_str = info["dtype"]
    if dtype_str == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype_str == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype_str == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(f"Dtype {dtype_str} not supported")
    return arr.reshape(info["shape"])

def rmsnorm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * weight

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def quantize_bitnet(W):
    gamma = float(np.mean(np.abs(W)))
    scale = 1.0 / (gamma + 1e-7)
    W_trits = np.clip(np.round(W * scale), -1.0, 1.0).astype(np.int8)
    return W_trits, gamma

def apply_rope(x, pos, base=1000000.0): # Qwen2.5 uses base=1000000.0 (1M)
    dim = len(x)
    half = dim // 2
    x1 = x[:half]
    x2 = x[half:]
    freqs = 1.0 / (base ** (np.arange(0, half, dtype=float) / half))
    angles = float(pos) * freqs
    out1 = x1 * np.cos(angles) - x2 * np.sin(angles)
    out2 = x1 * np.sin(angles) + x2 * np.cos(angles)
    return np.concatenate([out1, out2])

def run_test():
    model_path = os.path.join("data", "qwen2.5_0.5b.safetensors")
    tok_path = os.path.join("data", "tokenizer_qwen.json")

    if not os.path.exists(model_path):
        print(f"Erro: {model_path} nao encontrado.")
        return

    print("==========================================================================")
    print("  TESTE DE GERACAO DE TEXTO: QWEN2.5-0.5B (FP32 vs BITNET b1.58)")
    print("==========================================================================")

    # 1. Carregar Tokenizer
    tok = Tokenizer(tok_path)
    prompt = "A capital do Brasil é"
    prompt_tokens = tok.encode(prompt).tolist()
    print(f"[*] Prompt: \"{prompt}\"")
    print(f"[*] Tokens: {prompt_tokens}")

    # 2. Carregar Cabecalho SafeTensors
    f = open(model_path, "rb")
    header, hdr_len = load_qwen_header(model_path)

    # Carregar Embeddings e Final Norm
    print("[*] Carregando Embeddings e Final Norm...")
    t0 = time.time()
    embed_table = load_tensor(f, header, hdr_len, "model.embed_tokens.weight")
    final_norm = load_tensor(f, header, hdr_len, "model.norm.weight")
    print(f"    Carregado em {time.time() - t0:.2f}s! Embeddings shape: {embed_table.shape}")

    # Para demonstrar rapido e sem estourar memoria, vamos usar 4 camadas ou 24 camadas
    # Vamos verificar quantas camadas queremos testar:
    num_layers = 4  # Teste rapido com 4 camadas reais
    print(f"[*] Carregando e quantizando {num_layers} camadas para BitNet b1.58...")

    layers_fp32 = []
    layers_bitnet = []

    for l in range(num_layers):
        l_dict = {
            "norm1": load_tensor(f, header, hdr_len, f"model.layers.{l}.input_layernorm.weight"),
            "norm2": load_tensor(f, header, hdr_len, f"model.layers.{l}.post_attention_layernorm.weight"),
            "W_q": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.q_proj.weight"),
            "b_q": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.q_proj.bias"),
            "W_k": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.k_proj.weight"),
            "b_k": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.k_proj.bias"),
            "W_v": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.v_proj.weight"),
            "b_v": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.v_proj.bias"),
            "W_o": load_tensor(f, header, hdr_len, f"model.layers.{l}.self_attn.o_proj.weight"),
            "W_gate": load_tensor(f, header, hdr_len, f"model.layers.{l}.mlp.gate_proj.weight"),
            "W_up": load_tensor(f, header, hdr_len, f"model.layers.{l}.mlp.up_proj.weight"),
            "W_down": load_tensor(f, header, hdr_len, f"model.layers.{l}.mlp.down_proj.weight"),
        }
        layers_fp32.append(l_dict)

        # Quantizacao BitNet b1.58
        W_q_t, g_q = quantize_bitnet(l_dict["W_q"])
        W_k_t, g_k = quantize_bitnet(l_dict["W_k"])
        W_v_t, g_v = quantize_bitnet(l_dict["W_v"])
        W_o_t, g_o = quantize_bitnet(l_dict["W_o"])
        W_gate_t, g_gate = quantize_bitnet(l_dict["W_gate"])
        W_up_t, g_up = quantize_bitnet(l_dict["W_up"])
        W_down_t, g_down = quantize_bitnet(l_dict["W_down"])

        l_bit = {
            "norm1": l_dict["norm1"],
            "norm2": l_dict["norm2"],
            "W_q_t": W_q_t, "g_q": g_q, "b_q": l_dict["b_q"],
            "W_k_t": W_k_t, "g_k": g_k, "b_k": l_dict["b_k"],
            "W_v_t": W_v_t, "g_v": g_v, "b_v": l_dict["b_v"],
            "W_o_t": W_o_t, "g_o": g_o,
            "W_gate_t": W_gate_t, "g_gate": g_gate,
            "W_up_t": W_up_t, "g_up": g_up,
            "W_down_t": W_down_t, "g_down": g_down,
        }
        layers_bitnet.append(l_bit)

    f.close()
    print(f"[OK] {num_layers} camadas carregadas e quantizadas!")

    # Funcao de Forward Pass para 1 token
    def forward_token(token_id, layers, is_bitnet, pos, kv_caches):
        x = embed_table[token_id].copy()

        for l_idx, l in enumerate(layers):
            xn = rmsnorm(x, l["norm1"])

            if is_bitnet:
                q = (l["W_q_t"] @ xn) * l["g_q"] + l["b_q"]
                k = (l["W_k_t"] @ xn) * l["g_k"] + l["b_k"]
                v = (l["W_v_t"] @ xn) * l["g_v"] + l["b_v"]
            else:
                q = (l["W_q"] @ xn) + l["b_q"]
                k = (l["W_k"] @ xn) + l["b_k"]
                v = (l["W_v"] @ xn) + l["b_v"]

            # Aplicar RoPE em cada cabeca (dim_head = 64)
            q_heads = []
            for h in range(14):
                qh = q[h * 64 : (h + 1) * 64]
                q_heads.append(apply_rope(qh, pos))
            q_rot = np.concatenate(q_heads)

            k_heads = []
            for kh in range(2):
                kh_vec = k[kh * 64 : (kh + 1) * 64]
                k_heads.append(apply_rope(kh_vec, pos))
            k_rot = np.concatenate(k_heads)

            # Atualizar KV Cache da camada
            k_cache, v_cache = kv_caches[l_idx]
            k_cache.append(k_rot)
            v_cache.append(v)

            # Atencao GQA (14 Query heads, 2 KV heads -> 7 Q heads por KV head)
            k_mat = np.array(k_cache) # [seq_len, 128]
            v_mat = np.array(v_cache) # [seq_len, 128]

            attn_outs = []
            for h in range(14):
                kv_h = h // 7
                qh = q_rot[h * 64 : (h + 1) * 64]
                kh = k_mat[:, kv_h * 64 : (kv_h + 1) * 64]
                vh = v_mat[:, kv_h * 64 : (kv_h + 1) * 64]

                scores = np.dot(kh, qh) / 8.0 # sqrt(64) = 8.0
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                out_h = np.dot(weights, vh)
                attn_outs.append(out_h)

            attn_cat = np.concatenate(attn_outs)

            if is_bitnet:
                attn_proj = (l["W_o_t"] @ attn_cat) * l["g_o"]
            else:
                attn_proj = l["W_o"] @ attn_cat

            x = x + attn_proj

            # SwiGLU MLP
            xn2 = rmsnorm(x, l["norm2"])
            if is_bitnet:
                gate = (l["W_gate_t"] @ xn2) * l["g_gate"]
                up = (l["W_up_t"] @ xn2) * l["g_up"]
                h = silu(gate) * up
                mlp_out = (l["W_down_t"] @ h) * l["g_down"]
            else:
                gate = l["W_gate"] @ xn2
                up = l["W_up"] @ xn2
                h = silu(gate) * up
                mlp_out = l["W_down"] @ h

            x = x + mlp_out

        x_final = rmsnorm(x, final_norm)
        logits = np.dot(embed_table, x_final)
        return logits

    def generate(layers, is_bitnet, name, max_tokens=10):
        print(f"\n--- Iniciando Geracao com {name} ---")
        kv_caches = [[[], []] for _ in range(num_layers)]

        # Prefill prompt
        t_start = time.time()
        for idx, tok_id in enumerate(prompt_tokens):
            logits = forward_token(tok_id, layers, is_bitnet, idx, kv_caches)

        # Gerar novos tokens
        generated = []
        curr_token = int(np.argmax(logits))
        generated.append(curr_token)

        for step in range(max_tokens - 1):
            pos = len(prompt_tokens) + step
            logits = forward_token(curr_token, layers, is_bitnet, pos, kv_caches)
            curr_token = int(np.argmax(logits))
            generated.append(curr_token)

        total_time = time.time() - t_start
        text = tok.decode(generated)
        print(f"Tokens gerados ({name}): {generated}")
        print(f"Texto gerado: \"{text}\"")
        print(f"Velocidade: {max_tokens / total_time:.2f} tokens/s (Tempo total: {total_time:.2f}s)")
        return generated, text

    gen_fp32, text_fp32 = generate(layers_fp32, False, "Qwen Original (FP32)")
    gen_bit, text_bit = generate(layers_bitnet, True, "Qwen BitNet b1.58 (Ternário)")

    print("\n==========================================================================")
    print("  COMPARATIVO FINAL DE RESULTADOS")
    print("==========================================================================")
    print(f"Prompt Original:        \"{prompt}\"")
    print(f"Continuacao FP32:       \"{text_fp32}\"")
    print(f"Continuacao BitNet 1.58:\"{text_bit}\"")
    print("==========================================================================")

if __name__ == "__main__":
    run_test()
