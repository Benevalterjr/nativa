#!/usr/bin/env python3
"""
qutrit_test/benchmark_qwen3_1.7b_layer0.py
Rigorous Benchmark of Qwen3-1.7B Layer 0:
Evaluates FP32 vs Full INT4 vs Híbrido Base vs Híbrido SubQ vs Inverso vs Full Qutrit
using the REAL weights from data/qwen3_1.7b_layer0.safetensors:
  - hidden_size: 2048 (vs 1024 in 0.6B)
  - intermediate_size: 6144 (vs 3072 in 0.6B)
  - num_attention_heads: 16 (head_dim=128)
  - num_key_value_heads: 8 (GQA)
  - Layer 0 parameters: 50,331,648 (75% in MLP, 25% in Attention!)
"""

import os
import sys
import time
import math
import ctypes
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nativa
from qutrit_test.benchmark_qutrit_vs_int4 import load_kernels, BlockQ4_32, compute_metrics

MODEL_FILE = os.path.join(REPO_ROOT, "data", "qwen3_1.7b_layer0.safetensors")

DIM = 2048
INTER_DIM = 6144
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS # 2
ROPE_THETA = 1000000.0
RMS_NORM_EPS = 1e-6

def rmsnorm(x, gamma, eps=RMS_NORM_EPS):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * gamma

def rmsnorm_heads(vec, heads, head_dim, weight, eps=RMS_NORM_EPS):
    out = np.zeros_like(vec)
    for h in range(heads):
        idx = h * head_dim
        sub = vec[idx : idx + head_dim]
        rms = np.sqrt(np.mean(sub ** 2) + eps)
        out[idx : idx + head_dim] = (sub / rms) * weight
    return out

def compute_rope_freqs(head_dim, theta=ROPE_THETA):
    half = head_dim // 2
    freqs = np.zeros(half, dtype=np.float32)
    for i in range(half):
        freqs[i] = 1.0 / (theta ** (float(i) / float(half)))
    return freqs

ROPE_FREQS = compute_rope_freqs(HEAD_DIM, ROPE_THETA)

def apply_rope(vec, pos, num_heads, head_dim, freqs=ROPE_FREQS):
    half = head_dim // 2
    out = vec.copy()
    for h in range(num_heads):
        offset = h * head_dim
        v = out[offset : offset + head_dim]
        v0 = v[:half]
        v1 = v[half:]
        angle = pos * freqs
        cos_th = np.cos(angle)
        sin_th = np.sin(angle)
        out[offset : offset + half] = v0 * cos_th - v1 * sin_th
        out[offset + half : offset + head_dim] = v0 * sin_th + v1 * cos_th
    return out

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def run_qwen3_1_7b_benchmark(num_tokens=128, num_repeats=5):
    print("=" * 135)
    print("  BENCHMARK REAL DE QUANTIZAÇÃO: QWEN3-1.7B CAMADA 0 (DIM=2048, MLP=6144, 50.33M PARÂMETROS)")
    print("=" * 135)

    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Arquivo de pesos não encontrado: {MODEL_FILE}")
        return

    dll = load_kernels()

    # 1. Load Real Layer 0 Weights from Safetensors
    print(f"[*] Carregando pesos reais de '{MODEL_FILE}'...")
    def get_w(name):
        t = nativa.load_safetensors_fn(MODEL_FILE, f"model.layers.0.{name}")
        return np.array(t.data, dtype=np.float32).reshape(t.shape)

    norm1_w = get_w("input_layernorm.weight")
    norm2_w = get_w("post_attention_layernorm.weight")

    W_q = get_w("self_attn.q_proj.weight")  # [2048, 2048]
    W_k = get_w("self_attn.k_proj.weight")  # [1024, 2048]
    W_v = get_w("self_attn.v_proj.weight")  # [1024, 2048]
    W_o = get_w("self_attn.o_proj.weight")  # [2048, 2048]

    q_norm_w = get_w("self_attn.q_norm.weight")  # [128]
    k_norm_w = get_w("self_attn.k_norm.weight")  # [128]

    W_gate = get_w("mlp.gate_proj.weight")  # [6144, 2048]
    W_up   = get_w("mlp.up_proj.weight")    # [6144, 2048]
    W_down = get_w("mlp.down_proj.weight")  # [2048, 6144]

    total_attn_params = W_q.size + W_k.size + W_v.size + W_o.size
    total_mlp_params  = W_gate.size + W_up.size + W_down.size
    total_layer_params = total_attn_params + total_mlp_params

    print(f"    - Dimensão Oculta   : {DIM} | Dimensão MLP: {INTER_DIM}")
    print(f"    - Parâmetros Atenção: {total_attn_params:,} ({total_attn_params*100.0/total_layer_params:.1f}% da camada)")
    print(f"    - Parâmetros MLP    : {total_mlp_params:,} ({total_mlp_params*100.0/total_layer_params:.1f}% da camada)")
    print(f"    - Total da Camada   : {total_layer_params:,} parâmetros (3.2x maior que o 0.6B!)")

    # 2. Synthetic reproducible token stream (128 tokens x 2048 dim)
    np.random.seed(42)
    X_tokens = np.random.randn(num_tokens, DIM).astype(np.float32) * 0.08
    T = num_tokens

    # 3. Quantization Helpers
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
        return (w_pos, w_neg, gamma, num_blocks), M * num_blocks * 16

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
        return (w_pos, w_neg, gamma, beta, num_blocks), M * num_blocks * 16

    def gemv_qutrit_subq(q_pack, x, M, K):
        w_pos, w_neg, gamma, beta, num_blocks = q_pack
        x_sign = np.zeros(num_blocks, dtype=np.uint64)
        x_mag = np.zeros(num_blocks * 7, dtype=np.uint64)
        act_scale = ctypes.c_float(0.0)
        dll.nat_quantize_activations_popcnt(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), K, ctypes.byref(act_scale))
        y = np.zeros(M, dtype=np.float32)
        dll.nat_qutrit_dot64_subquantum_gemv(w_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), w_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, num_blocks, gamma, beta, act_scale.value)
        return y

    # Quantize matrices
    print("\n[*] Quantizando tensores do Qwen3-1.7B...")
    int4_q, s_q_i4 = make_int4(W_q)
    int4_k, s_k_i4 = make_int4(W_k)
    int4_v, s_v_i4 = make_int4(W_v)
    int4_o, s_o_i4 = make_int4(W_o)
    int4_gate, s_g_i4 = make_int4(W_gate)
    int4_up,   s_u_i4 = make_int4(W_up)
    int4_down, s_d_i4 = make_int4(W_down)

    qbase_q, _ = make_qutrit_base(W_q)
    qbase_k, _ = make_qutrit_base(W_k)
    qbase_v, _ = make_qutrit_base(W_v)
    qbase_o, _ = make_qutrit_base(W_o)
    qbase_gate, _ = make_qutrit_base(W_gate)
    qbase_up,   _ = make_qutrit_base(W_up)
    qbase_down, _ = make_qutrit_base(W_down)

    qsub_q, s_q_q = make_qutrit_subq(W_q)
    qsub_k, s_k_q = make_qutrit_subq(W_k)
    qsub_v, s_v_q = make_qutrit_subq(W_v)
    qsub_o, s_o_q = make_qutrit_subq(W_o)
    qsub_gate, s_g_q = make_qutrit_subq(W_gate)
    qsub_up,   s_u_q = make_qutrit_subq(W_up)
    qsub_down, s_d_q = make_qutrit_subq(W_down)

    # Memory calculations
    mem_fp32 = (total_layer_params * 4.0) / (1024 * 1024)
    mem_int4 = (s_q_i4 + s_k_i4 + s_v_i4 + s_o_i4 + s_g_i4 + s_u_i4 + s_d_i4) / (1024 * 1024)
    mem_hyb  = (s_q_i4 + s_k_i4 + s_v_i4 + s_o_i4 + s_g_q + s_u_q + s_d_q) / (1024 * 1024)
    mem_full_q = (s_q_q + s_k_q + s_v_q + s_o_q + s_g_q + s_u_q + s_d_q) / (1024 * 1024)

    # Forward pass across T tokens
    def run_layer_forward(attn_mode, mlp_mode):
        k_cache = np.zeros((NUM_KV_HEADS, T, HEAD_DIM), dtype=np.float32)
        v_cache = np.zeros((NUM_KV_HEADS, T, HEAD_DIM), dtype=np.float32)
        out_mid = np.zeros((T, DIM), dtype=np.float32)
        out_mlp = np.zeros((T, DIM), dtype=np.float32)
        out_final = np.zeros((T, DIM), dtype=np.float32)
        inv_sqrt_d = 1.0 / math.sqrt(HEAD_DIM)

        t_start = time.perf_counter()

        for t in range(T):
            x = X_tokens[t]
            xn1 = rmsnorm(x, norm1_w)

            # Attention GEMV
            if attn_mode == "fp32":
                q = W_q.dot(xn1)
                k = W_k.dot(xn1)
                v = W_v.dot(xn1)
            elif attn_mode == "int4":
                q = gemv_int4(int4_q, xn1, 2048, 2048)
                k = gemv_int4(int4_k, xn1, 1024, 2048)
                v = gemv_int4(int4_v, xn1, 1024, 2048)
            elif attn_mode == "qbase":
                q = gemv_qutrit_base(qbase_q, xn1, 2048, 2048)
                k = gemv_qutrit_base(qbase_k, xn1, 1024, 2048)
                v = gemv_qutrit_base(qbase_v, xn1, 1024, 2048)
            elif attn_mode == "qsub":
                q = gemv_qutrit_subq(qsub_q, xn1, 2048, 2048)
                k = gemv_qutrit_subq(qsub_k, xn1, 1024, 2048)
                v = gemv_qutrit_subq(qsub_v, xn1, 1024, 2048)

            q = rmsnorm_heads(q, NUM_HEADS, HEAD_DIM, q_norm_w)
            k = rmsnorm_heads(k, NUM_KV_HEADS, HEAD_DIM, k_norm_w)

            q = apply_rope(q, t, NUM_HEADS, HEAD_DIM)
            k = apply_rope(k, t, NUM_KV_HEADS, HEAD_DIM)

            for kv_h in range(NUM_KV_HEADS):
                k_cache[kv_h, t] = k[kv_h * HEAD_DIM : (kv_h + 1) * HEAD_DIM]
                v_cache[kv_h, t] = v[kv_h * HEAD_DIM : (kv_h + 1) * HEAD_DIM]

            attn_out_heads = np.zeros(NUM_HEADS * HEAD_DIM, dtype=np.float32)
            for h in range(NUM_HEADS):
                kv_h = h // HEADS_PER_KV
                q_h = q[h * HEAD_DIM : (h + 1) * HEAD_DIM]
                k_hist = k_cache[kv_h, : t + 1]
                v_hist = v_cache[kv_h, : t + 1]

                scores = k_hist.dot(q_h) * inv_sqrt_d
                scores_max = np.max(scores)
                exp_scores = np.exp(scores - scores_max)
                weights = exp_scores / np.sum(exp_scores)
                attn_out_heads[h * HEAD_DIM : (h + 1) * HEAD_DIM] = weights.dot(v_hist)

            if attn_mode == "fp32":
                attn_proj = W_o.dot(attn_out_heads)
            elif attn_mode == "int4":
                attn_proj = gemv_int4(int4_o, attn_out_heads, 2048, 2048)
            elif attn_mode == "qbase":
                attn_proj = gemv_qutrit_base(qbase_o, attn_out_heads, 2048, 2048)
            elif attn_mode == "qsub":
                attn_proj = gemv_qutrit_subq(qsub_o, attn_out_heads, 2048, 2048)

            x_mid = x + attn_proj
            out_mid[t] = x_mid

            xn2 = rmsnorm(x_mid, norm2_w)

            if mlp_mode == "fp32":
                gate = W_gate.dot(xn2)
                up   = W_up.dot(xn2)
                h_mlp = silu(gate) * up
                mlp_proj = W_down.dot(h_mlp)
            elif mlp_mode == "int4":
                gate = gemv_int4(int4_gate, xn2, 6144, 2048)
                up   = gemv_int4(int4_up,   xn2, 6144, 2048)
                h_mlp = silu(gate) * up
                mlp_proj = gemv_int4(int4_down, h_mlp, 2048, 6144)
            elif mlp_mode == "qbase":
                gate = gemv_qutrit_base(qbase_gate, xn2, 6144, 2048)
                up   = gemv_qutrit_base(qbase_up,   xn2, 6144, 2048)
                h_mlp = silu(gate) * up
                mlp_proj = gemv_qutrit_base(qbase_down, h_mlp, 2048, 6144)
            elif mlp_mode == "qsub":
                gate = gemv_qutrit_subq(qsub_gate, xn2, 6144, 2048)
                up   = gemv_qutrit_subq(qsub_up,   xn2, 6144, 2048)
                h_mlp = silu(gate) * up
                mlp_proj = gemv_qutrit_subq(qsub_down, h_mlp, 2048, 6144)

            out_mlp[t] = mlp_proj
            out_final[t] = x_mid + mlp_proj

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        return out_mid, out_mlp, out_final, dt_ms

    def score_sequence(ref_seq, cand_seq):
        c_list, r_list, m_list = [], [], []
        for t in range(T):
            res = compute_metrics(ref_seq[t], cand_seq[t])
            c_list.append(res["cos_sim"])
            r_list.append(res["rmse"])
            m_list.append(res["mae"])
        return float(np.mean(c_list)), float(np.mean(r_list)), float(np.mean(m_list))

    print("\n[*] Aquecimento do pipeline...")
    _, _, _, _ = run_layer_forward("int4", "int4")

    configs = [
        ("0. FP32 Referência", "fp32", "fp32", 32.0, mem_fp32),
        ("1. Full INT4 (Padrão Produção)", "int4", "int4", 4.50, mem_int4),
        ("2. Híbrido Base (Attn INT4 + MLP Qutrit Base)", "int4", "qbase", 2.62, mem_hyb),
        ("3. Híbrido SubQ (Attn INT4 + MLP Qutrit SubQ)", "int4", "qsub", 2.62, mem_hyb),
        ("4. Inverso (Attn Qutrit SubQ + MLP INT4)", "qsub", "int4", 3.88, mem_int4), # similar size
        ("5. Full Qutrit Base", "qbase", "qbase", 2.00, mem_full_q),
        ("6. Full Qutrit Sub-Quantum", "qsub", "qsub", 2.00, mem_full_q)
    ]

    print(f"\n[*] Executando {len(configs)} variantes ({num_repeats} repetições com mediana e desvio)...")

    results = []
    ref_mid, ref_mlp, ref_final = None, None, None

    for name, a_mode, m_mode, bpw, ram_mb in configs:
        times = []
        last_mid, last_mlp, last_final = None, None, None
        for r in range(num_repeats):
            mid, mlp, fin, dt = run_layer_forward(a_mode, m_mode)
            times.append(dt)
            last_mid, last_mlp, last_final = mid, mlp, fin

        med_dt = float(np.median(times))
        std_dt = float(np.std(times))

        if a_mode == "fp32" and m_mode == "fp32":
            ref_mid, ref_mlp, ref_final = last_mid, last_mlp, last_final
            cos_mid, cos_mlp, cos_fin = 1.0, 1.0, 1.0
            rmse_fin, mae_fin = 0.0, 0.0
        else:
            cos_mid, _, _ = score_sequence(ref_mid, last_mid)
            cos_mlp, _, _ = score_sequence(ref_mlp, last_mlp)
            cos_fin, rmse_fin, mae_fin = score_sequence(ref_final, last_final)

        results.append({
            "name": name,
            "bpw": bpw,
            "ram_mb": ram_mb,
            "med_dt": med_dt,
            "std_dt": std_dt,
            "tok_s": T * 1000.0 / med_dt,
            "cos_mid": cos_mid,
            "cos_mlp": cos_mlp,
            "cos_fin": cos_fin,
            "rmse_fin": rmse_fin,
            "mae_fin": mae_fin
        })

    print("\n" + "=" * 145)
    print(f"{'Variante Qwen3-1.7B':<42} | {'Bits/P':<6} | {'RAM/Cam':<9} | {'RAM (28L)':<10} | {'Cos Attn':<9} | {'Cos MLP':<8} | {'Cos Final':<10} | {'RMSE':<8} | {'Latência (ms)':<16} | {'Throughput':<9}")
    print("=" * 145)
    for r in results:
        ram_28l = r['ram_mb'] * 28.0 + 593.5 # 28 layers + 593MB embeddings
        print(f"{r['name']:<42} | {r['bpw']:<6.2f} | {r['ram_mb']:<6.2f} MB | {ram_28l:<7.1f} MB | {r['cos_mid']:<9.4f} | {r['cos_mlp']:<8.4f} | {r['cos_fin']:<10.4f} | {r['rmse_fin']:<8.4f} | {r['med_dt']:>6.1f} ± {r['std_dt']:<4.1f} ms | {r['tok_s']:<7.1f} tok/s")
    print("-" * 145)

    print("\n[*] ANÁLISE COMPARATIVA DE ESCALA (1.7B vs 0.6B):")
    int4_res = results[1]
    hyb_subq_res = results[3]
    full_q_res = results[6]
    speedup = int4_res['med_dt'] / hyb_subq_res['med_dt']
    ram_reduction = (1.0 - (hyb_subq_res['ram_mb'] / int4_res['ram_mb'])) * 100.0

    print(f"    1. Compressão Acentuada no 1.7B:")
    print(f"       - No 1.7B, o MLP representa 75.0% da camada (vs. 60.0% no 0.6B).")
    print(f"       - Redução de RAM por camada: {int4_res['ram_mb']:.1f} MB -> {hyb_subq_res['ram_mb']:.1f} MB ({ram_reduction:.1f}% MENOR!)")
    print(f"       - Taxa média ponderada de bits: {hyb_subq_res['bpw']:.2f} bits/peso!")
    print(f"\n    2. Aceleração por Hardware:")
    print(f"       - Latência Full INT4 : {int4_res['med_dt']:.1f} ms ({int4_res['tok_s']:.1f} tok/s)")
    print(f"       - Latência Híbrido   : {hyb_subq_res['med_dt']:.1f} ms ({hyb_subq_res['tok_s']:.1f} tok/s)")
    print(f"       - Speedup da Camada  : {speedup:.2f}x mais rápido no 1.7B!")
    print(f"\n    3. Fidelidade Numérica em Escala 1.7B:")
    print(f"       - Full INT4   : Cosseno = {int4_res['cos_fin']:.4f} | RMSE = {int4_res['rmse_fin']:.4f}")
    print(f"       - Híbrido     : Cosseno = {hyb_subq_res['cos_fin']:.4f} | RMSE = {hyb_subq_res['rmse_fin']:.4f}")
    print(f"       - Full Qutrit : Cosseno = {full_q_res['cos_fin']:.4f} | RMSE = {full_q_res['rmse_fin']:.4f}")
    print("=" * 145)

if __name__ == "__main__":
    run_qwen3_1_7b_benchmark(num_tokens=128, num_repeats=5)
