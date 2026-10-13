#!/usr/bin/env python3
"""Benchmark oficial de velocidade de geração (Tokens/s) do Qwen2.5-0.5B em FP32.

Mede o Throughput de Decode Auto-regressivo Token-a-Token utilizando o runtime do Nativa
com pesos residentes em RAM e KVCache Memory Arena (estilo llama2.c).
"""

import os
import sys
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa

MODEL_FILE = os.path.join("data", "qwen2.5_0.5b.safetensors")
TOKENIZER_FILE = os.path.join("data", "tokenizer_qwen.json")
ROPE_BASE = 1_000_000.0
HEAD_DIM = 64
NUM_HEADS = 14
NUM_KV_HEADS = 2
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS  # 7
D_MODEL = 896
NUM_LAYERS = 24


def rmsnorm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def apply_rope_single(vec, pos, heads):
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_BASE ** (np.arange(0, half, dtype=np.float32) / float(half)))
    angles = float(pos) * freqs
    cos = np.cos(angles)
    sin = np.sin(angles)
    
    v = vec.reshape(heads, HEAD_DIM)
    x1 = v[:, :half]
    x2 = v[:, half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return np.concatenate([out1, out2], axis=-1).reshape(-1)


def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def load_model_weights():
    print("[*] Carregando pesos oficiais do Qwen2.5-0.5B (24 camadas) para a RAM...")
    t0 = time.perf_counter()
    w = {}
    w["embed"] = nativa.load_safetensors_fn(MODEL_FILE, "model.embed_tokens.weight").data.astype(np.float32)
    w["norm"] = nativa.load_safetensors_fn(MODEL_FILE, "model.norm.weight").data.astype(np.float32)
    
    w["layers"] = []
    for l in range(NUM_LAYERS):
        prefix = f"model.layers.{l}."
        layer_w = {
            "norm1": nativa.load_safetensors_fn(MODEL_FILE, prefix + "input_layernorm.weight").data.astype(np.float32),
            "norm2": nativa.load_safetensors_fn(MODEL_FILE, prefix + "post_attention_layernorm.weight").data.astype(np.float32),
            "wq": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.q_proj.weight").data.astype(np.float32),
            "bq": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.q_proj.bias").data.astype(np.float32),
            "wk": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.k_proj.weight").data.astype(np.float32),
            "bk": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.k_proj.bias").data.astype(np.float32),
            "wv": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.v_proj.weight").data.astype(np.float32),
            "bv": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.v_proj.bias").data.astype(np.float32),
            "wo": nativa.load_safetensors_fn(MODEL_FILE, prefix + "self_attn.o_proj.weight").data.astype(np.float32),
            "wgate": nativa.load_safetensors_fn(MODEL_FILE, prefix + "mlp.gate_proj.weight").data.astype(np.float32),
            "wup": nativa.load_safetensors_fn(MODEL_FILE, prefix + "mlp.up_proj.weight").data.astype(np.float32),
            "wdown": nativa.load_safetensors_fn(MODEL_FILE, prefix + "mlp.down_proj.weight").data.astype(np.float32),
        }
        w["layers"].append(layer_w)
    
    elapsed = time.perf_counter() - t0
    total_mb = sum(arr.nbytes for k, arr in w.items() if isinstance(arr, np.ndarray))
    for lw in w["layers"]:
        total_mb += sum(arr.nbytes for arr in lw.values())
    total_mb /= (1024 * 1024)
    print(f"    -> {NUM_LAYERS} camadas carregadas na memória RAM em {elapsed:.2f}s ({total_mb:.1f} MB em FP32).")
    return w


def run_benchmark(num_tokens_generate=25):
    weights = load_model_weights()
    tokenizer = nativa.Tokenizer(TOKENIZER_FILE)
    
    # Prompt de teste
    prompt = "Artificial Intelligence is transforming"
    print(f"\n[*] Prompt: {prompt!r}")
    prompt_tokens = tokenizer.encode(prompt).tolist()
    prompt_len = len(prompt_tokens)
    print(f"[*] Tokens no prompt: {prompt_len}")
    
    # Inicializa os 24 KV-Caches com Memory Arena
    kv_caches = [nativa.KVCache(max_seq_len=1024, dim=NUM_KV_HEADS * HEAD_DIM) for _ in range(NUM_LAYERS)]
    
    # -------------------------------------------------------------
    # 1. PREFILL DO PROMPT
    # -------------------------------------------------------------
    print("[*] Executando Prefill do prompt...")
    t_prefill_start = time.perf_counter()
    
    # Fatias de embedding para todos os tokens do prompt
    h = weights["embed"][prompt_tokens].copy()  # shape (prompt_len, 896)
    
    # Processa cada token do prompt alimentando o KV Cache
    for pos, tok_id in enumerate(prompt_tokens):
        x = weights["embed"][tok_id].copy()
        for l in range(NUM_LAYERS):
            lw = weights["layers"][l]
            xn = rmsnorm(x, lw["norm1"])
            q = lw["wq"] @ xn + lw["bq"]
            k = lw["wk"] @ xn + lw["bk"]
            v = lw["wv"] @ xn + lw["bv"]
            
            q_rot = apply_rope_single(q, pos, NUM_HEADS)
            k_rot = apply_rope_single(k, pos, NUM_KV_HEADS)
            
            # Insere no KV Cache
            k_cache, v_cache = kv_caches[l].update(k_rot, v)
            cur_len = kv_caches[l].seq_len
            
            # GQA Attention
            Q = q_rot.reshape(NUM_HEADS, HEAD_DIM)
            K = k_cache.data.reshape(cur_len, NUM_KV_HEADS, HEAD_DIM).repeat(GROUP_SIZE, axis=1).transpose(1, 0, 2)
            V = v_cache.data.reshape(cur_len, NUM_KV_HEADS, HEAD_DIM).repeat(GROUP_SIZE, axis=1).transpose(1, 0, 2)
            
            # (14, 64) @ (14, 64, cur_len) -> (14, cur_len)
            scores = np.einsum('hd,hsd->hs', Q, K) / math.sqrt(HEAD_DIM)
            probs = softmax(scores)
            attn_out = np.einsum('hs,hsd->hd', probs, V).reshape(-1)
            
            x = x + (lw["wo"] @ attn_out)
            xn2 = rmsnorm(x, lw["norm2"])
            gate = lw["wgate"] @ xn2
            up = lw["wup"] @ xn2
            h_mlp = nativa.silu_fn(nativa.Tensor(gate)).data * up
            x = x + (lw["wdown"] @ h_mlp)
            
    # Projeção de saída do último token do prompt
    x_final = rmsnorm(x, weights["norm"])
    logits = weights["embed"] @ x_final
    next_token = int(np.argmax(logits))
    
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"    -> Prefill concluído em {t_prefill*1000.0:.2f} ms ({prompt_len / t_prefill:.1f} tokens/s).")
    print(f"    -> Primeiro token gerado: {tokenizer.decode([next_token])!r}")
    
    # -------------------------------------------------------------
    # 2. DECODE AUTO-REGRESSIVO (TOKEN-A-TOKEN COM KV-CACHE)
    # -------------------------------------------------------------
    print(f"\n[*] Gerando {num_tokens_generate} tokens em modo Decode Auto-regressivo FP32...")
    generated_tokens = [next_token]
    times = []
    
    for step in range(num_tokens_generate - 1):
        cur_pos = prompt_len + step
        t_step_start = time.perf_counter()
        
        cur_token = generated_tokens[-1]
        x = weights["embed"][cur_token].copy()
        
        for l in range(NUM_LAYERS):
            lw = weights["layers"][l]
            xn = rmsnorm(x, lw["norm1"])
            q = lw["wq"] @ xn + lw["bq"]
            k = lw["wk"] @ xn + lw["bk"]
            v = lw["wv"] @ xn + lw["bv"]
            
            q_rot = apply_rope_single(q, cur_pos, NUM_HEADS)
            k_rot = apply_rope_single(k, cur_pos, NUM_KV_HEADS)
            
            k_cache, v_cache = kv_caches[l].update(k_rot, v)
            cur_len = kv_caches[l].seq_len
            
            Q = q_rot.reshape(NUM_HEADS, HEAD_DIM)
            K = k_cache.data.reshape(cur_len, NUM_KV_HEADS, HEAD_DIM).repeat(GROUP_SIZE, axis=1).transpose(1, 0, 2)
            V = v_cache.data.reshape(cur_len, NUM_KV_HEADS, HEAD_DIM).repeat(GROUP_SIZE, axis=1).transpose(1, 0, 2)
            
            scores = np.einsum('hd,hsd->hs', Q, K) / math.sqrt(HEAD_DIM)
            probs = softmax(scores)
            attn_out = np.einsum('hs,hsd->hd', probs, V).reshape(-1)
            
            x = x + (lw["wo"] @ attn_out)
            xn2 = rmsnorm(x, lw["norm2"])
            gate = lw["wgate"] @ xn2
            up = lw["wup"] @ xn2
            h_mlp = nativa.silu_fn(nativa.Tensor(gate)).data * up
            x = x + (lw["wdown"] @ h_mlp)
            
        x_final = rmsnorm(x, weights["norm"])
        logits = weights["embed"] @ x_final
        next_token = int(np.argmax(logits))
        
        t_step = time.perf_counter() - t_step_start
        times.append(t_step)
        generated_tokens.append(next_token)
        sys.stdout.write(tokenizer.decode([next_token]))
        sys.stdout.flush()
        
    print("\n")
    avg_step_time = np.mean(times)
    tokens_per_sec = 1.0 / avg_step_time
    total_decode_time = sum(times)
    
    texto_completo = tokenizer.decode(prompt_tokens + generated_tokens)
    print("=" * 70)
    print("  RESULTADO DO BENCHMARK DE GERAÇÃO FP32 (QWEN2.5-0.5B NO NATIVA)")
    print("=" * 70)
    print(f"• Texto completo gerado:\n  {texto_completo!r}\n")
    print(f"• Tokens gerados no decode: {len(generated_tokens)}")
    print(f"• Tempo médio por token:    {avg_step_time * 1000.0:.2f} ms")
    print(f"• Velocidade de Decode:     {tokens_per_sec:.2f} TOKENS/SEGUNDO!")
    print(f"• Tempo total de Decode:    {total_decode_time:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(num_tokens_generate=25)
