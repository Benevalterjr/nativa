#!/usr/bin/env python3
"""
qutrit_test/test_wikitext2_int4_vs_qutrit.py
Benchmark on WikiText-2 Test Split:
INT4 vs. BitNet b1.58 vs. Qutrit dot64 (Sub-Quantum Semantics C)
Using Real Qwen Weights and Official WikiText-2 Tokens.
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

def load_wikitext_tokens(max_tokens=128):
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"WikiText-2 corpus not found at {CORPUS_FILE}")
    if not os.path.exists(TOKENIZER_FILE):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_FILE}")

    tok = nativa.Tokenizer(TOKENIZER_FILE)
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        # Read clean paragraphs
        text_data = f.read(25000)

    enc = tok.encode(text_data)
    token_ids = enc.tolist() if isinstance(enc, nativa.Tensor) else list(enc)
    return token_ids[:max_tokens], tok

def evaluate_layer_projections_on_wikitext(num_tokens=128):
    print("=" * 96)
    print("  WIKITEXT-2 BENCHMARK: QWEN PROJECTIONS UNDER INT4 vs BITNET vs QUTRIT (SEM. C)")
    print("=" * 96)

    dll = load_kernels()

    # 1. Load Tokens and Tokenizer
    print(f"[*] Loading WikiText-2 tokens from '{CORPUS_FILE}'...")
    token_ids, tok = load_wikitext_tokens(max_tokens=num_tokens)
    print(f"    - Extracted sequence of {len(token_ids)} tokens from WikiText-2 test set.")

    # 2. Load Embeddings
    print(f"[*] Loading input embeddings from '{MODEL_FILE}'...")
    embed_t = nativa.load_safetensors_fn(MODEL_FILE, "model.embed_tokens.weight")
    # Shape: [num_tokens, 1024]
    X_seq = np.array([embed_t.data[t] for t in token_ids], dtype=np.float32)
    T, K = X_seq.shape
    print(f"    - Embedded sequence shape: [{T}, {K}] ({T} token vectors of dimension {K})")

    # 3. Test on Key Layer Projections: MLP (gate_proj) and Attention (q_proj)
    projections = [
        ("MLP Gate Projection", "model.layers.0.mlp.gate_proj.weight"),
        ("MLP Up Projection", "model.layers.0.mlp.up_proj.weight"),
        ("Self-Attention Q Proj", "model.layers.0.self_attn.q_proj.weight")
    ]

    all_summary = {}

    for proj_name, weight_key in projections:
        print("\n" + "#" * 96)
        print(f"  EVALUATING: {proj_name} ('{weight_key}') ON {T} WIKITEXT-2 TOKENS")
        print("#" * 96)

        W_t = nativa.load_safetensors_fn(MODEL_FILE, weight_key)
        W = np.array(W_t.data, dtype=np.float32).reshape(W_t.shape)
        M, K = W.shape
        gamma = float(np.mean(np.abs(W)))
        beta = 0.38

        # --- Quantize Weights ---
        # A) INT4
        num_blocks_int4 = (K + 31) // 32
        w_int4 = (BlockQ4_32 * (M * num_blocks_int4))()
        dll.nat_quantize_int4_g32(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), w_int4, M, K)
        size_int4_kb = (M * num_blocks_int4 * ctypes.sizeof(BlockQ4_32)) / 1024.0

        # B) BitNet b1.58 (veclut_5trits)
        bpr_5t = (K + 4) // 5
        W_trits = np.clip(np.round(W / gamma), -1, 1).astype(np.int8)
        w_packed_5t = np.zeros(M * bpr_5t, dtype=np.uint8)
        dll.nat_pack_5trits_2d(W_trits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), w_packed_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), M, K)
        size_5t_kb = (M * bpr_5t) / 1024.0

        # C) Qutrit Baseline (00, 01, 10)
        num_blocks_q = (K + 63) // 64
        W_pos_base = np.zeros(M * num_blocks_q, dtype=np.uint64)
        W_neg_base = np.zeros(M * num_blocks_q, dtype=np.uint64)
        dll.nat_pack_qutrit_baseline(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), W_pos_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), W_neg_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), M, K, gamma)
        size_q_kb = (M * num_blocks_q * 16) / 1024.0

        # D) Qutrit Sub-Quantum (Semântica C)
        W_pos_sub = np.zeros(M * num_blocks_q, dtype=np.uint64)
        W_neg_sub = np.zeros(M * num_blocks_q, dtype=np.uint64)
        dll.nat_pack_qutrit_subquantum(W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), M, K, gamma, 0.15)

        # Buffers for outputs across the whole WikiText sequence
        Y_fp32 = np.zeros((T, M), dtype=np.float32)
        Y_int4 = np.zeros((T, M), dtype=np.float32)
        Y_5t = np.zeros((T, M), dtype=np.float32)
        Y_q_base = np.zeros((T, M), dtype=np.float32)
        Y_q_sub = np.zeros((T, M), dtype=np.float32)

        # Execution and Timing over all T tokens
        # 1. FP32
        t0 = time.perf_counter()
        for t in range(T):
            Y_fp32[t] = W.dot(X_seq[t])
        t_fp32 = (time.perf_counter() - t0) * 1000.0

        # 2. INT4
        t0 = time.perf_counter()
        for t in range(T):
            xt = np.ascontiguousarray(X_seq[t], dtype=np.float32)
            yt = np.zeros(M, dtype=np.float32)
            dll.nat_gemv_int4_g32(w_int4, xt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), yt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, K)
            Y_int4[t] = yt
        t_int4 = (time.perf_counter() - t0) * 1000.0

        # 3. BitNet 5-trits
        t0 = time.perf_counter()
        for t in range(T):
            xt = np.ascontiguousarray(X_seq[t], dtype=np.float32)
            yt = np.zeros(M, dtype=np.float32)
            dll.nat_gemv_veclut_5trits(w_packed_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), xt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), yt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, K, gamma)
            Y_5t[t] = yt
        t_5t = (time.perf_counter() - t0) * 1000.0

        # 4. Qutrit Baseline (dot64)
        t0 = time.perf_counter()
        for t in range(T):
            xt = np.ascontiguousarray(X_seq[t], dtype=np.float32)
            yt = np.zeros(M, dtype=np.float32)
            x_sign = np.zeros(num_blocks_q, dtype=np.uint64)
            x_mag = np.zeros(num_blocks_q * 7, dtype=np.uint64)
            act_scale = ctypes.c_float(0.0)
            dll.nat_quantize_activations_popcnt(xt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), K, ctypes.byref(act_scale))
            dll.nat_qutrit_dot64_baseline_gemv(W_pos_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), W_neg_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), yt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, num_blocks_q, gamma, act_scale.value)
            Y_q_base[t] = yt
        t_q_base = (time.perf_counter() - t0) * 1000.0

        # 5. Qutrit Sub-Quantum (Semântica C)
        t0 = time.perf_counter()
        for t in range(T):
            xt = np.ascontiguousarray(X_seq[t], dtype=np.float32)
            yt = np.zeros(M, dtype=np.float32)
            x_sign = np.zeros(num_blocks_q, dtype=np.uint64)
            x_mag = np.zeros(num_blocks_q * 7, dtype=np.uint64)
            act_scale = ctypes.c_float(0.0)
            dll.nat_quantize_activations_popcnt(xt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), K, ctypes.byref(act_scale))
            dll.nat_qutrit_dot64_subquantum_gemv(W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), yt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, num_blocks_q, gamma, beta, act_scale.value)
            Y_q_sub[t] = yt
        t_q_sub = (time.perf_counter() - t0) * 1000.0

        # Compute average metrics across all T tokens
        def seq_metrics(Y_ref, Y_cand):
            cos_list = []
            rmse_list = []
            mae_list = []
            rel_list = []
            for t in range(T):
                m = compute_metrics(Y_ref[t], Y_cand[t])
                cos_list.append(m["cos_sim"])
                rmse_list.append(m["rmse"])
                mae_list.append(m["mae"])
                rel_list.append(m["rel_err"])
            return {
                "cos": float(np.mean(cos_list)),
                "rmse": float(np.mean(rmse_list)),
                "mae": float(np.mean(mae_list)),
                "rel": float(np.mean(rel_list)) * 100.0
            }

        m_int4 = seq_metrics(Y_fp32, Y_int4)
        m_5t = seq_metrics(Y_fp32, Y_5t)
        m_q_base = seq_metrics(Y_fp32, Y_q_base)
        m_q_sub = seq_metrics(Y_fp32, Y_q_sub)

        # Print detailed table for this projection
        fp32_size_kb = (M * K * 4.0) / 1024.0
        print(f"\n{'Quantization Mode':<34} | {'Bits':<5} | {'RAM (KB)':<9} | {'Mean Cosine':<11} | {'RMSE':<8} | {'MAE':<8} | {'Total (ms)':<10} | {'Tokens/s':<9}")
        print("-" * 112)
        print(f"{'FP32 Baseline (Reference)':<34} | {'32.0':<5} | {fp32_size_kb:<9.1f} | {'1.0000':<11} | {'0.0000':<8} | {'0.0000':<8} | {t_fp32:<10.2f} | {T*1000.0/t_fp32:<9.1f}")
        print(f"{'INT4 (Group-wise 32)':<34} | {'4.50':<5} | {size_int4_kb:<9.1f} | {m_int4['cos']:<11.4f} | {m_int4['rmse']:<8.4f} | {m_int4['mae']:<8.4f} | {t_int4:<10.2f} | {T*1000.0/t_int4:<9.1f}")
        print(f"{'BitNet b1.58: veclut_5trits':<34} | {'1.60':<5} | {size_5t_kb:<9.1f} | {m_5t['cos']:<11.4f} | {m_5t['rmse']:<8.4f} | {m_5t['mae']:<8.4f} | {t_5t:<10.2f} | {T*1000.0/t_5t:<9.1f}")
        print(f"{'Qutrit dot64: Baseline':<34} | {'2.00':<5} | {size_q_kb:<9.1f} | {m_q_base['cos']:<11.4f} | {m_q_base['rmse']:<8.4f} | {m_q_base['mae']:<8.4f} | {t_q_base:<10.2f} | {T*1000.0/t_q_base:<9.1f}")
        print(f"{'Qutrit dot64: Sub-Quantum (Sem. C)':<34} | {'2.00':<5} | {size_q_kb:<9.1f} | {m_q_sub['cos']:<11.4f} | {m_q_sub['rmse']:<8.4f} | {m_q_sub['mae']:<8.4f} | {t_q_sub:<10.2f} | {T*1000.0/t_q_sub:<9.1f}")
        print("-" * 112)

        gain_cos = m_q_sub['cos'] - m_q_base['cos']
        gain_rmse = (m_q_base['rmse'] - m_q_sub['rmse']) / m_q_base['rmse'] * 100.0
        gain_mae = (m_q_base['mae'] - m_q_sub['mae']) / m_q_base['mae'] * 100.0
        speedup_int4 = t_int4 / t_q_sub

        print(f"[*] Sub-Quantum (Semântica C) vs Ternário Base:")
        print(f"    - Ganho de Cosseno médio em WikiText-2: +{gain_cos:.4f}")
        print(f"    - Redução no RMSE:                      -{gain_rmse:.2f}% (menos erro)")
        print(f"    - Redução no MAE:                       -{gain_mae:.2f}% (menos erro)")
        print(f"    - Velocidade vs INT4:                   {speedup_int4:.2f}x mais rápido na CPU!")

        all_summary[proj_name] = {
            "int4": m_int4,
            "q_base": m_q_base,
            "q_sub": m_q_sub,
            "speedup": speedup_int4
        }

    return all_summary

if __name__ == "__main__":
    evaluate_layer_projections_on_wikitext(num_tokens=128)
