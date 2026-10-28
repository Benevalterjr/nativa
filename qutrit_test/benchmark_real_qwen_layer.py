#!/usr/bin/env python3
"""
qutrit_test/benchmark_real_qwen_layer.py
Rigorous Benchmark of Real Qwen Layer 0 Architecture:
Implements the TRUE Qwen/Qwen3 forward pass:
  1. Input RMSNorm
  2. Q, K, V Linear Projections (INT4 / Qutrit / FP32)
  3. Per-head QK-Norm (q_norm and k_norm weights)
  4. Rotary Position Embeddings (RoPE, theta=1e6, half-split)
  5. Causal KV-Cache Accumulation & Grouped Query Attention (GQA: 16 Q heads, 8 KV heads)
  6. Softmax & Value-weighted aggregation
  7. Output Projection Wo (INT4 / Qutrit / FP32)
  8. First Residual Connection (x_mid = x + attn_out) -> Tracks Stage 1 Error
  9. Post-Attention RMSNorm
  10. SwiGLU MLP: gate_proj, up_proj, silu(gate)*up, down_proj -> Tracks Stage 2 Error
  11. Second Residual Connection (y = x_mid + mlp_out) -> Tracks Stage 3 Final Error

Evaluates all 4 core ablation variants requested + Baselines:
  - FP32 Reference
  - Full INT4 (Reference)
  - Híbrido Base (Attn INT4 + MLP Qutrit Base)
  - Híbrido SubQ (Attn INT4 + MLP Qutrit Sub-Quantum)
  - Inverso (Attn Qutrit SubQ + MLP INT4)
  - Full Qutrit Base
  - Full Qutrit Sub-Quantum
With warmup + multiple repetitions (median & IQR/std), and stage-by-stage error breakdown.
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

CORPUS_FILE = os.path.join(REPO_ROOT, "data", "wikitext-2-test.txt")
TOKENIZER_FILE = os.path.join(REPO_ROOT, "data", "tokenizer_qwen.json")
MODEL_FILE = os.path.join(REPO_ROOT, "data", "model.safetensors")

# Model Architecture Parameters (Qwen3-0.6B / Qwen2.5-0.5B)
DIM = 1024
INTER_DIM = 3072
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

def run_real_qwen_benchmark(num_tokens=128, num_repeats=5):
    print("=" * 125)
    print("  AUDITORIA RIGOROSA: FORWARD REAL DA CAMADA QWEN COM ATENÇÃO GQA, ROPE E KV-CACHE")
    print("=" * 125)

    dll = load_kernels()

    # 1. Load Tokens from WikiText-2
    print(f"[*] Carregando sequência real do WikiText-2 ({CORPUS_FILE})...")
    tok = nativa.Tokenizer(TOKENIZER_FILE)
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        text = f.read(35000)
    token_ids = tok.encode(text).tolist()[:num_tokens]
    embed_t = nativa.load_safetensors_fn(MODEL_FILE, "model.embed_tokens.weight")
    X_tokens = np.array([embed_t.data[t] for t in token_ids], dtype=np.float32)
    T, D = X_tokens.shape
    print(f"    - Tokens avaliados: {T} | Dimensão embedding: {D}")

    # 2. Load Real Layer 0 Weights
    print(f"\n[*] Carregando pesos reais de '{MODEL_FILE}' (Layer 0)...")
    def get_w(name):
        t = nativa.load_safetensors_fn(MODEL_FILE, f"model.layers.0.{name}")
        return np.array(t.data, dtype=np.float32).reshape(t.shape)

    norm1_w = get_w("input_layernorm.weight")
    norm2_w = get_w("post_attention_layernorm.weight")

    W_q = get_w("self_attn.q_proj.weight")  # [2048, 1024]
    W_k = get_w("self_attn.k_proj.weight")  # [1024, 1024]
    W_v = get_w("self_attn.v_proj.weight")  # [1024, 1024]
    W_o = get_w("self_attn.o_proj.weight")  # [1024, 2048]

    q_norm_w = get_w("self_attn.q_norm.weight")  # [128]
    k_norm_w = get_w("self_attn.k_norm.weight")  # [128]

    W_gate = get_w("mlp.gate_proj.weight")  # [3072, 1024]
    W_up   = get_w("mlp.up_proj.weight")    # [3072, 1024]
    W_down = get_w("mlp.down_proj.weight")  # [1024, 3072]

    # Pre-quantize helpers
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

    print("\n[*] Quantizando pesos para os formatos sob teste...")
    # INT4
    int4_q, s_q_int4 = make_int4(W_q)
    int4_k, s_k_int4 = make_int4(W_k)
    int4_v, s_v_int4 = make_int4(W_v)
    int4_o, s_o_int4 = make_int4(W_o)
    int4_gate, s_gate_int4 = make_int4(W_gate)
    int4_up,   s_up_int4   = make_int4(W_up)
    int4_down, s_down_int4 = make_int4(W_down)

    # Qutrit Base
    qbase_q, _ = make_qutrit_base(W_q)
    qbase_k, _ = make_qutrit_base(W_k)
    qbase_v, _ = make_qutrit_base(W_v)
    qbase_o, _ = make_qutrit_base(W_o)
    qbase_gate, _ = make_qutrit_base(W_gate)
    qbase_up,   _ = make_qutrit_base(W_up)
    qbase_down, _ = make_qutrit_base(W_down)

    # Qutrit Sub-Quantum
    qsub_q, _ = make_qutrit_subq(W_q)
    qsub_k, _ = make_qutrit_subq(W_k)
    qsub_v, _ = make_qutrit_subq(W_v)
    qsub_o, _ = make_qutrit_subq(W_o)
    qsub_gate, _ = make_qutrit_subq(W_gate)
    qsub_up,   _ = make_qutrit_subq(W_up)
    qsub_down, _ = make_qutrit_subq(W_down)

    # TRUE Qwen Forward Pass across T tokens with causal KV-cache
    def run_full_qwen_forward(attn_mode, mlp_mode):
        # KV-cache: [8 heads, T, 128]
        k_cache = np.zeros((NUM_KV_HEADS, T, HEAD_DIM), dtype=np.float32)
        v_cache = np.zeros((NUM_KV_HEADS, T, HEAD_DIM), dtype=np.float32)

        # Output buffers
        out_mid = np.zeros((T, DIM), dtype=np.float32)   # after Attention + Residual 1
        out_mlp = np.zeros((T, DIM), dtype=np.float32)   # MLP contribution
        out_final = np.zeros((T, DIM), dtype=np.float32) # final layer output (Residual 2)

        inv_sqrt_d = 1.0 / math.sqrt(HEAD_DIM)

        t_start = time.perf_counter()

        for t in range(T):
            x = X_tokens[t]

            # 1. Input RMSNorm
            xn1 = rmsnorm(x, norm1_w)

            # 2. Linear Projections for Attention
            if attn_mode == "fp32":
                q = W_q.dot(xn1)
                k = W_k.dot(xn1)
                v = W_v.dot(xn1)
            elif attn_mode == "int4":
                q = gemv_int4(int4_q, xn1, 2048, 1024)
                k = gemv_int4(int4_k, xn1, 1024, 1024)
                v = gemv_int4(int4_v, xn1, 1024, 1024)
            elif attn_mode == "qbase":
                q = gemv_qutrit_base(qbase_q, xn1, 2048, 1024)
                k = gemv_qutrit_base(qbase_k, xn1, 1024, 1024)
                v = gemv_qutrit_base(qbase_v, xn1, 1024, 1024)
            elif attn_mode == "qsub":
                q = gemv_qutrit_subq(qsub_q, xn1, 2048, 1024)
                k = gemv_qutrit_subq(qsub_k, xn1, 1024, 1024)
                v = gemv_qutrit_subq(qsub_v, xn1, 1024, 1024)

            # 3. QK-Norm (per-head RMSNorm)
            q = rmsnorm_heads(q, NUM_HEADS, HEAD_DIM, q_norm_w)
            k = rmsnorm_heads(k, NUM_KV_HEADS, HEAD_DIM, k_norm_w)

            # 4. RoPE (Rotary Position Embeddings)
            q = apply_rope(q, t, NUM_HEADS, HEAD_DIM)
            k = apply_rope(k, t, NUM_KV_HEADS, HEAD_DIM)

            # 5. Store into KV-Cache at position t
            for kv_h in range(NUM_KV_HEADS):
                k_cache[kv_h, t] = k[kv_h * HEAD_DIM : (kv_h + 1) * HEAD_DIM]
                v_cache[kv_h, t] = v[kv_h * HEAD_DIM : (kv_h + 1) * HEAD_DIM]

            # 6. Grouped Query Attention (GQA: 16 Q heads, 8 KV heads)
            attn_out_heads = np.zeros(NUM_HEADS * HEAD_DIM, dtype=np.float32)
            for h in range(NUM_HEADS):
                kv_h = h // HEADS_PER_KV
                q_h = q[h * HEAD_DIM : (h + 1) * HEAD_DIM]

                # Causal attention scores against positions 0..t
                # k_history: [t+1, 128]
                k_hist = k_cache[kv_h, : t + 1]
                v_hist = v_cache[kv_h, : t + 1]

                scores = k_hist.dot(q_h) * inv_sqrt_d
                # Causal Softmax
                scores_max = np.max(scores)
                exp_scores = np.exp(scores - scores_max)
                weights = exp_scores / np.sum(exp_scores)

                # Context aggregation
                head_context = weights.dot(v_hist)
                attn_out_heads[h * HEAD_DIM : (h + 1) * HEAD_DIM] = head_context

            # 7. Output Projection Wo
            if attn_mode == "fp32":
                attn_proj = W_o.dot(attn_out_heads)
            elif attn_mode == "int4":
                attn_proj = gemv_int4(int4_o, attn_out_heads, 1024, 2048)
            elif attn_mode == "qbase":
                attn_proj = gemv_qutrit_base(qbase_o, attn_out_heads, 1024, 2048)
            elif attn_mode == "qsub":
                attn_proj = gemv_qutrit_subq(qsub_o, attn_out_heads, 1024, 2048)

            # 8. First Residual Connection
            x_mid = x + attn_proj
            out_mid[t] = x_mid

            # 9. Post-Attention RMSNorm
            xn2 = rmsnorm(x_mid, norm2_w)

            # 10. MLP (SwiGLU)
            if mlp_mode == "fp32":
                gate = W_gate.dot(xn2)
                up   = W_up.dot(xn2)
                h_mlp = silu(gate) * up
                mlp_proj = W_down.dot(h_mlp)
            elif mlp_mode == "int4":
                gate = gemv_int4(int4_gate, xn2, 3072, 1024)
                up   = gemv_int4(int4_up,   xn2, 3072, 1024)
                h_mlp = silu(gate) * up
                mlp_proj = gemv_int4(int4_down, h_mlp, 1024, 3072)
            elif mlp_mode == "qbase":
                gate = gemv_qutrit_base(qbase_gate, xn2, 3072, 1024)
                up   = gemv_qutrit_base(qbase_up,   xn2, 3072, 1024)
                h_mlp = silu(gate) * up
                mlp_proj = gemv_qutrit_base(qbase_down, h_mlp, 1024, 3072)
            elif mlp_mode == "qsub":
                gate = gemv_qutrit_subq(qsub_gate, xn2, 3072, 1024)
                up   = gemv_qutrit_subq(qsub_up,   xn2, 3072, 1024)
                h_mlp = silu(gate) * up
                mlp_proj = gemv_qutrit_subq(qsub_down, h_mlp, 1024, 3072)

            out_mlp[t] = mlp_proj

            # 11. Second Residual Connection
            y = x_mid + mlp_proj
            out_final[t] = y

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

    # Warmup
    print("\n[*] Executando aquecimento do pipeline de atenção real...")
    _, _, _, _ = run_full_qwen_forward("int4", "int4")

    # Define Configurations under Test
    configs = [
        ("0. FP32 Referência", "fp32", "fp32"),
        ("1. Full INT4 (Padrão de Produção)", "int4", "int4"),
        ("2. Híbrido Base (Attn INT4 + MLP Qutrit Base)", "int4", "qbase"),
        ("3. Híbrido SubQ (Attn INT4 + MLP Qutrit SubQ)", "int4", "qsub"),
        ("4. Inverso (Attn Qutrit SubQ + MLP INT4)", "qsub", "int4"),
        ("5. Full Qutrit Base", "qbase", "qbase"),
        ("6. Full Qutrit Sub-Quantum", "qsub", "qsub")
    ]

    print(f"\n[*] Executando {len(configs)} variantes com {num_repeats} repetições cada (mediana + desvio)...")

    results = []
    ref_mid, ref_mlp, ref_final = None, None, None

    for name, a_mode, m_mode in configs:
        times = []
        last_mid, last_mlp, last_final = None, None, None
        for r in range(num_repeats):
            mid, mlp, fin, dt = run_full_qwen_forward(a_mode, m_mode)
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
            "attn_mode": a_mode,
            "mlp_mode": m_mode,
            "med_dt": med_dt,
            "std_dt": std_dt,
            "tok_s": T * 1000.0 / med_dt,
            "cos_mid": cos_mid,
            "cos_mlp": cos_mlp,
            "cos_fin": cos_fin,
            "rmse_fin": rmse_fin,
            "mae_fin": mae_fin
        })

    # Display Table with Stage Breakdown
    print("\n" + "=" * 135)
    print(f"{'Variante':<40} | {'Cos Atenção':<11} | {'Cos MLP':<9} | {'Cosseno Final':<13} | {'RMSE':<8} | {'MAE':<8} | {'Latência (ms)':<15} | {'Throughput':<9}")
    print("=" * 135)
    for r in results:
        print(f"{r['name']:<40} | {r['cos_mid']:<11.4f} | {r['cos_mlp']:<9.4f} | {r['cos_fin']:<13.4f} | {r['rmse_fin']:<8.4f} | {r['mae_fin']:<8.4f} | {r['med_dt']:>6.1f} ± {r['std_dt']:<4.1f} ms | {r['tok_s']:<7.1f} tok/s")
    print("-" * 135)

    print("\n[*] DIAGNÓSTICO ESTÁGIO POR ESTÁGIO:")
    int4_res = results[1]
    hyb_base_res = results[2]
    hyb_subq_res = results[3]
    inv_res = results[4]
    full_q_res = results[6]

    print(f"    1. Efeito na Atenção (Resíduo 1, x_mid):")
    print(f"       - Full INT4   : Cosseno = {int4_res['cos_mid']:.4f}")
    print(f"       - Híbrido     : Cosseno = {hyb_subq_res['cos_mid']:.4f} (IDÊNTICO AO INT4, pois a atenção é 100% INT4!)")
    print(f"       - Inverso     : Cosseno = {inv_res['cos_mid']:.4f} (Qutrit na atenção degrada de imediato)")
    print(f"       - Full Qutrit : Cosseno = {full_q_res['cos_mid']:.4f}")
    print(f"\n    2. Efeito no MLP (mlp_out isolado):")
    print(f"       - Full INT4   : Cosseno = {int4_res['cos_mlp']:.4f}")
    print(f"       - Híbrido Base: Cosseno = {hyb_base_res['cos_mlp']:.4f}")
    print(f"       - Híbrido SubQ: Cosseno = {hyb_subq_res['cos_mlp']:.4f} (Delta SubQ vs Base = +{(hyb_subq_res['cos_mlp'] - hyb_base_res['cos_mlp']):.4f})")
    print(f"\n    3. Saída Final da Camada (Resíduo 2, y):")
    print(f"       - Full INT4   : Cosseno = {int4_res['cos_fin']:.4f}")
    print(f"       - Híbrido Base: Cosseno = {hyb_base_res['cos_fin']:.4f}")
    print(f"       - Híbrido SubQ: Cosseno = {hyb_subq_res['cos_fin']:.4f}")
    print(f"       - Inverso     : Cosseno = {inv_res['cos_fin']:.4f}")
    print(f"       - Full Qutrit : Cosseno = {full_q_res['cos_fin']:.4f}")
    print("=" * 135)

if __name__ == "__main__":
    run_real_qwen_benchmark(num_tokens=128, num_repeats=5)
