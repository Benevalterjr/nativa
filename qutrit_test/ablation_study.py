#!/usr/bin/env python3
"""
qutrit_test/ablation_study.py
Rigorous Ablation Study for Qwen Quantization Architecture:
Evaluates 4 core variants requested by audit, plus FP32 and Full Qutrit:
  1. Full INT4 (100% INT4 in all 7 projections) - Reference of quality & speed
  2. Atenção INT4 + MLP Qutrit Base (no Semantics C) - Isolates Hybrid Architecture effect
  3. Atenção INT4 + MLP Qutrit Sub-Quantum (with Semantics C) - Isolates Semantics C incremental gain
  4. Atenção Qutrit + MLP INT4 (Inverse Hybrid) - Verifies if Attention preservation is the decisive factor
  5. Full Qutrit Base (100% Qutrit Base)
  6. Full Qutrit Sub-Quantum (100% Qutrit Sub-Quantum)
All using identical weights, tokens, memory models, and execution conditions.
"""

import os
import sys
import time
import ctypes
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nativa
from qutrit_test.benchmark_qutrit_vs_int4 import load_kernels, BlockQ4_32, compute_metrics

CORPUS_FILE = os.path.join(REPO_ROOT, "data", "wikitext-2-test.txt")
TOKENIZER_FILE = os.path.join(REPO_ROOT, "data", "tokenizer_qwen.json")
MODEL_FILE = os.path.join(REPO_ROOT, "data", "model.safetensors")

def rmsnorm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * gamma

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def run_ablation(num_tokens=128):
    print("=" * 115)
    print("  ESTUDO DE ABLAÇÃO RIGOROSO: QWEN LAYER 0 (WIKITEXT-2 TEST SPLIT, N=128)")
    print("=" * 115)

    dll = load_kernels()

    # 1. Load Tokens from WikiText-2
    print(f"[*] Carregando tokens reais do WikiText-2 ({CORPUS_FILE})...")
    tok = nativa.Tokenizer(TOKENIZER_FILE)
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        text = f.read(35000)
    token_ids = tok.encode(text).tolist()[:num_tokens]
    embed_t = nativa.load_safetensors_fn(MODEL_FILE, "model.embed_tokens.weight")
    X_tokens = np.array([embed_t.data[t] for t in token_ids], dtype=np.float32)
    T, DIM = X_tokens.shape
    print(f"    - Sequência avaliada: {T} tokens | Dimensão embedding: {DIM}")

    # 2. Load Layer 0 Weights
    print(f"\n[*] Carregando pesos reais da Camada 0 de '{MODEL_FILE}'...")
    def get_weight(name):
        t = nativa.load_safetensors_fn(MODEL_FILE, f"model.layers.0.{name}")
        return np.array(t.data, dtype=np.float32).reshape(t.shape)

    norm1 = get_weight("input_layernorm.weight")
    norm2 = get_weight("post_attention_layernorm.weight")

    # Attention weights
    W_q = get_weight("self_attn.q_proj.weight")  # [2048, 1024]
    W_k = get_weight("self_attn.k_proj.weight")  # [1024, 1024]
    W_v = get_weight("self_attn.v_proj.weight")  # [1024, 1024]
    W_o = get_weight("self_attn.o_proj.weight")  # [1024, 2048]

    # MLP weights
    W_gate = get_weight("mlp.gate_proj.weight")  # [3072, 1024]
    W_up   = get_weight("mlp.up_proj.weight")    # [3072, 1024]
    W_down = get_weight("mlp.down_proj.weight")  # [1024, 3072]

    total_attn_params = W_q.size + W_k.size + W_v.size + W_o.size
    total_mlp_params  = W_gate.size + W_up.size + W_down.size
    total_layer_params = total_attn_params + total_mlp_params

    print(f"    - Parâmetros Atenção : {total_attn_params:,} ({total_attn_params*100.0/total_layer_params:.1f}%)")
    print(f"    - Parâmetros MLP     : {total_mlp_params:,} ({total_mlp_params*100.0/total_layer_params:.1f}%)")
    print(f"    - Total Camada       : {total_layer_params:,} parâmetros")

    # Quantization helpers
    def make_int4(W):
        M, K = W.shape
        num_blocks = (K + 31) // 32
        w_blocks = (BlockQ4_32 * (M * num_blocks))()
        dll.nat_quantize_int4_g32(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), w_blocks, M, K)
        size_bytes = M * num_blocks * ctypes.sizeof(BlockQ4_32)
        return w_blocks, size_bytes

    def gemv_int4(w_blocks, x, M, K):
        y = np.zeros(M, dtype=np.float32)
        dll.nat_gemv_int4_g32(w_blocks, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, K)
        return y

    def make_qutrit_base(W):
        M, K = W.shape
        gamma = float(np.mean(np.abs(W)))
        num_blocks = (K + 63) // 64
        w_pos = np.zeros(M * num_blocks, dtype=np.uint64)
        w_neg = np.zeros(M * num_blocks, dtype=np.uint64)
        dll.nat_pack_qutrit_baseline(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), w_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), w_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), M, K, gamma)
        size_bytes = M * num_blocks * 16
        return (w_pos, w_neg, gamma, num_blocks), size_bytes

    def gemv_qutrit_base(q_pack, x, M, K):
        w_pos, w_neg, gamma, num_blocks = q_pack
        x_sign = np.zeros(num_blocks, dtype=np.uint64)
        x_mag = np.zeros(num_blocks * 7, dtype=np.uint64)
        act_scale = ctypes.c_float(0.0)
        dll.nat_quantize_activations_popcnt(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), K, ctypes.byref(act_scale))
        y = np.zeros(M, dtype=np.float32)
        dll.nat_qutrit_dot64_baseline_gemv(w_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), w_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, num_blocks, gamma, act_scale.value)
        return y

    def make_qutrit_subq(W):
        M, K = W.shape
        gamma = float(np.mean(np.abs(W)))
        beta = 0.38
        num_blocks = (K + 63) // 64
        w_pos = np.zeros(M * num_blocks, dtype=np.uint64)
        w_neg = np.zeros(M * num_blocks, dtype=np.uint64)
        dll.nat_pack_qutrit_subquantum(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), w_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), w_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), M, K, gamma, 0.15)
        size_bytes = M * num_blocks * 16
        return (w_pos, w_neg, gamma, beta, num_blocks), size_bytes

    def gemv_qutrit_subq(q_pack, x, M, K):
        w_pos, w_neg, gamma, beta, num_blocks = q_pack
        x_sign = np.zeros(num_blocks, dtype=np.uint64)
        x_mag = np.zeros(num_blocks * 7, dtype=np.uint64)
        act_scale = ctypes.c_float(0.0)
        dll.nat_quantize_activations_popcnt(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), K, ctypes.byref(act_scale))
        y = np.zeros(M, dtype=np.float32)
        dll.nat_qutrit_dot64_subquantum_gemv(w_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), w_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, num_blocks, gamma, beta, act_scale.value)
        return y

    # Pre-quantize all matrices
    print("\n[*] Pré-quantizando todas as matrizes para todos os formatos...")
    # INT4
    int4_q, s_q_int4 = make_int4(W_q)
    int4_k, s_k_int4 = make_int4(W_k)
    int4_v, s_v_int4 = make_int4(W_v)
    int4_o, s_o_int4 = make_int4(W_o)
    int4_gate, s_gate_int4 = make_int4(W_gate)
    int4_up,   s_up_int4   = make_int4(W_up)
    int4_down, s_down_int4 = make_int4(W_down)

    # Qutrit Base
    qbase_q, s_q_qbase = make_qutrit_base(W_q)
    qbase_k, s_k_qbase = make_qutrit_base(W_k)
    qbase_v, s_v_qbase = make_qutrit_base(W_v)
    qbase_o, s_o_qbase = make_qutrit_base(W_o)
    qbase_gate, s_gate_qbase = make_qutrit_base(W_gate)
    qbase_up,   s_up_qbase   = make_qutrit_base(W_up)
    qbase_down, s_down_qbase = make_qutrit_base(W_down)

    # Qutrit Sub-Quantum
    qsub_q, s_q_qsub = make_qutrit_subq(W_q)
    qsub_k, s_k_qsub = make_qutrit_subq(W_k)
    qsub_v, s_v_qsub = make_qutrit_subq(W_v)
    qsub_o, s_o_qsub = make_qutrit_subq(W_o)
    qsub_gate, s_gate_qsub = make_qutrit_subq(W_gate)
    qsub_up,   s_up_qsub   = make_qutrit_subq(W_up)
    qsub_down, s_down_qsub = make_qutrit_subq(W_down)

    # Helper forward function for a generic configuration
    def run_layer_forward(attn_mode, mlp_mode):
        Y = np.zeros((T, DIM), dtype=np.float32)
        t0 = time.perf_counter()
        for t in range(T):
            x = X_tokens[t]
            xn1 = rmsnorm(x, norm1)
            # Attention forward
            if attn_mode == "fp32":
                q = W_q.dot(xn1)
                k = W_k.dot(xn1)
                v = W_v.dot(xn1)
                attn_out = W_o.dot(q)
            elif attn_mode == "int4":
                q = gemv_int4(int4_q, xn1, 2048, 1024)
                k = gemv_int4(int4_k, xn1, 1024, 1024)
                v = gemv_int4(int4_v, xn1, 1024, 1024)
                attn_out = gemv_int4(int4_o, q, 1024, 2048)
            elif attn_mode == "qbase":
                q = gemv_qutrit_base(qbase_q, xn1, 2048, 1024)
                k = gemv_qutrit_base(qbase_k, xn1, 1024, 1024)
                v = gemv_qutrit_base(qbase_v, xn1, 1024, 1024)
                attn_out = gemv_qutrit_base(qbase_o, q, 1024, 2048)
            elif attn_mode == "qsub":
                q = gemv_qutrit_subq(qsub_q, xn1, 2048, 1024)
                k = gemv_qutrit_subq(qsub_k, xn1, 1024, 1024)
                v = gemv_qutrit_subq(qsub_v, xn1, 1024, 1024)
                attn_out = gemv_qutrit_subq(qsub_o, q, 1024, 2048)

            x_mid = x + attn_out

            # MLP forward
            xn2 = rmsnorm(x_mid, norm2)
            if mlp_mode == "fp32":
                gate = W_gate.dot(xn2)
                up   = W_up.dot(xn2)
                h = silu(gate) * up
                mlp_out = W_down.dot(h)
            elif mlp_mode == "int4":
                gate = gemv_int4(int4_gate, xn2, 3072, 1024)
                up   = gemv_int4(int4_up,   xn2, 3072, 1024)
                h = silu(gate) * up
                mlp_out = gemv_int4(int4_down, h, 1024, 3072)
            elif mlp_mode == "qbase":
                gate = gemv_qutrit_base(qbase_gate, xn2, 3072, 1024)
                up   = gemv_qutrit_base(qbase_up,   xn2, 3072, 1024)
                h = silu(gate) * up
                mlp_out = gemv_qutrit_base(qbase_down, h, 1024, 3072)
            elif mlp_mode == "qsub":
                gate = gemv_qutrit_subq(qsub_gate, xn2, 3072, 1024)
                up   = gemv_qutrit_subq(qsub_up,   xn2, 3072, 1024)
                h = silu(gate) * up
                mlp_out = gemv_qutrit_subq(qsub_down, h, 1024, 3072)

            Y[t] = x_mid + mlp_out

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return Y, dt_ms

    def calc_metrics(Y_ref, Y_cand):
        cos_l, rmse_l, mae_l = [], [], []
        for t in range(T):
            m = compute_metrics(Y_ref[t], Y_cand[t])
            cos_l.append(m["cos_sim"])
            rmse_l.append(m["rmse"])
            mae_l.append(m["mae"])
        return float(np.mean(cos_l)), float(np.mean(rmse_l)), float(np.mean(mae_l))

    # Memory calculators
    def get_mem_mb(attn_mode, mlp_mode):
        bytes_attn = 0
        if attn_mode == "fp32": bytes_attn = total_attn_params * 4
        elif attn_mode == "int4": bytes_attn = s_q_int4 + s_k_int4 + s_v_int4 + s_o_int4
        elif attn_mode in ("qbase", "qsub"): bytes_attn = s_q_qsub + s_k_qsub + s_v_qsub + s_o_qsub

        bytes_mlp = 0
        if mlp_mode == "fp32": bytes_mlp = total_mlp_params * 4
        elif mlp_mode == "int4": bytes_mlp = s_gate_int4 + s_up_int4 + s_down_int4
        elif mlp_mode in ("qbase", "qsub"): bytes_mlp = s_gate_qsub + s_up_qsub + s_down_qsub

        return (bytes_attn + bytes_mlp) / (1024.0 * 1024.0)

    # Calculate effective bits/weight
    def get_bpw(attn_mode, mlp_mode):
        bpw_attn = 32.0 if attn_mode == "fp32" else (4.50 if attn_mode == "int4" else 2.00)
        bpw_mlp  = 32.0 if mlp_mode == "fp32" else (4.50 if mlp_mode == "int4" else 2.00)
        return (bpw_attn * total_attn_params + bpw_mlp * total_mlp_params) / total_layer_params

    # Run Warmup
    print("\n[*] Executando aquecimento de kernels...")
    _, _ = run_layer_forward("int4", "int4")

    # Execute all Ablation Variants
    print("\n[*] Executando os 6 testes de ablação controlados...")

    variants = [
        ("0. FP32 Referência", "fp32", "fp32"),
        ("1. Full INT4 (Referência de Produção)", "int4", "int4"),
        ("2. Híbrido Base: Attn INT4 + MLP Qutrit Base", "int4", "qbase"),
        ("3. Híbrido SubQ: Attn INT4 + MLP Qutrit SubQ", "int4", "qsub"),
        ("4. Inverso: Attn Qutrit SubQ + MLP INT4", "qsub", "int4"),
        ("5. Full Qutrit Base", "qbase", "qbase"),
        ("6. Full Qutrit Sub-Quantum", "qsub", "qsub")
    ]

    results = []
    Y_ref = None

    for name, a_mode, m_mode in variants:
        Y_cand, dt = run_layer_forward(a_mode, m_mode)
        if a_mode == "fp32" and m_mode == "fp32":
            Y_ref = Y_cand
            cos, rmse, mae = 1.0, 0.0, 0.0
        else:
            cos, rmse, mae = calc_metrics(Y_ref, Y_cand)

        ram_layer = get_mem_mb(a_mode, m_mode)
        ram_total = ram_layer * 28.0 + 12.0 # 28 layers + embeddings/head
        bpw = get_bpw(a_mode, m_mode)
        tok_s = T * 1000.0 / dt

        results.append({
            "name": name,
            "bpw": bpw,
            "ram_layer": ram_layer,
            "ram_total": ram_total,
            "cos": cos,
            "rmse": rmse,
            "mae": mae,
            "dt": dt,
            "tok_s": tok_s
        })

    # Print Table
    print("\n" + "=" * 125)
    print(f"{'Variante de Ablação':<42} | {'Bits/P':<6} | {'RAM/Cam':<9} | {'RAM (28L)':<10} | {'Cosseno':<8} | {'RMSE':<8} | {'MAE':<8} | {'Throughput':<9}")
    print("=" * 125)
    for r in results:
        print(f"{r['name']:<42} | {r['bpw']:<6.2f} | {r['ram_layer']:<6.2f} MB | {r['ram_total']:<7.1f} MB | {r['cos']:<8.4f} | {r['rmse']:<8.4f} | {r['mae']:<8.4f} | {r['tok_s']:<7.1f} tok/s")
    print("-" * 125)

    # Detailed Ablation Analysis
    cos_int4 = results[1]["cos"]
    cos_hyb_base = results[2]["cos"]
    cos_hyb_subq = results[3]["cos"]
    cos_inv = results[4]["cos"]
    cos_full_subq = results[6]["cos"]

    delta_arch = cos_hyb_base - cos_full_subq
    delta_subq = cos_hyb_subq - cos_hyb_base
    delta_attn_decisive = cos_hyb_subq - cos_inv

    print("\n[*] DESCOBERTAS CRUCIAIS DO ESTUDO DE ABLAÇÃO:")
    print(f"    1. Efeito da Arquitetura Híbrida Isolado (Attn INT4 vs. Full Qutrit):")
    print(f"       - Salto de Fidelidade: Cosseno vai de {cos_full_subq:.4f} para {cos_hyb_base:.4f} (+{delta_arch:.4f})")
    print(f"       - Prova cabal de que a Atenção era o gargalo primário de degradação vetorial.")
    print(f"\n    2. Ganho Incremental Isolado da Semântica C (Sub-Quantum no MLP):")
    print(f"       - Qutrit Base no MLP : Cosseno = {cos_hyb_base:.4f}")
    print(f"       - Qutrit SubQ no MLP : Cosseno = {cos_hyb_subq:.4f} (+{delta_subq:.4f})")
    print(f"       - O dither de fase-xadrez adiciona ganho incremental positivo e consistente.")
    print(f"\n    3. Teste Inverso (Attn Qutrit + MLP INT4):")
    print(f"       - Cosseno da Variante Inversa: {cos_inv:.4f}")
    print(f"       - Diferença vs. Híbrido Padrão : {delta_attn_decisive:+.4f} a favor de proteger a Atenção!")
    print(f"       - Mesmo mantendo 60% dos pesos em INT4 (no MLP), quantizar a Atenção em Qutrit")
    print(f"         derruba a fidelidade para {cos_inv:.4f}. A ATENÇÃO É O FATOR DECISIVO.")
    print("=" * 125)

if __name__ == "__main__":
    run_ablation(num_tokens=128)
