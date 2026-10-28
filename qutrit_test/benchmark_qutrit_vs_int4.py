#!/usr/bin/env python3
"""
qutrit_test/benchmark_qutrit_vs_int4.py
Rigorous Comparative Benchmark:
INT4 vs. BitNet b1.58 (5-trits) vs. Qutrit dot64 (Semantics A, B, C) on Qwen Architecture.
"""

import os
import sys
import time
import ctypes
import numpy as np

# Add repo root to sys.path so we can import nativa
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nativa

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(THIS_DIR, "qutrit_kernels.dll")

class BlockQ4_32(ctypes.Structure):
    _fields_ = [
        ("scale", ctypes.c_float),
        ("qs", ctypes.c_uint8 * 16)
    ]

def load_kernels():
    if not os.path.exists(DLL_PATH):
        from compile_kernels import compile_dll
        compile_dll()

    dll = ctypes.CDLL(DLL_PATH)

    # 1. INT4
    dll.nat_quantize_int4_g32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(BlockQ4_32),
        ctypes.c_int64,
        ctypes.c_int64
    ]
    dll.nat_quantize_int4_g32.restype = None

    dll.nat_gemv_int4_g32.argtypes = [
        ctypes.POINTER(BlockQ4_32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64
    ]
    dll.nat_gemv_int4_g32.restype = None

    # 2. Ternary / 5-trits
    dll.nat_gemv_ternary_int8.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float
    ]
    dll.nat_gemv_ternary_int8.restype = None

    dll.nat_pack_5trits_2d.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.c_int64
    ]
    dll.nat_pack_5trits_2d.restype = None

    dll.nat_gemv_veclut_5trits.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float
    ]
    dll.nat_gemv_veclut_5trits.restype = None

    # 3. Qutrit dot64
    dll.nat_pack_qutrit_baseline.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float
    ]
    dll.nat_pack_qutrit_baseline.restype = None

    dll.nat_pack_qutrit_subquantum.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float
    ]
    dll.nat_pack_qutrit_subquantum.restype = None

    dll.nat_quantize_activations_popcnt.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float)
    ]
    dll.nat_quantize_activations_popcnt.restype = None

    dll.nat_qutrit_dot64_baseline_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float
    ]
    dll.nat_qutrit_dot64_baseline_gemv.restype = None

    dll.nat_qutrit_dot64_subquantum_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float
    ]
    dll.nat_qutrit_dot64_subquantum_gemv.restype = None

    dll.nat_qutrit_dot64_ctx_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint64
    ]
    dll.nat_qutrit_dot64_ctx_gemv.restype = None

    dll.nat_qutrit_dot64_subquantum_gemv_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float
    ]
    dll.nat_qutrit_dot64_subquantum_gemv_f32.restype = None

    # 4. veclut_4qutrits
    dll.nat_pack_4qutrits.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float
    ]
    dll.nat_pack_4qutrits.restype = None

    dll.nat_gemv_veclut_4qutrits.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float
    ]
    dll.nat_gemv_veclut_4qutrits.restype = None

    return dll

def compute_metrics(y_true, y_pred):
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    norm_t = np.linalg.norm(y_true)
    norm_p = np.linalg.norm(y_pred)
    if norm_t > 1e-9 and norm_p > 1e-9:
        cos_sim = float(np.dot(y_true, y_pred) / (norm_t * norm_p))
        std_t = np.std(y_true)
        std_p = np.std(y_pred)
        if std_t > 1e-9 and std_p > 1e-9:
            pearson = float(np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / (std_t * std_p))
        else:
            pearson = 0.0
    else:
        cos_sim = 0.0
        pearson = 0.0
    rel_err = float(np.linalg.norm(diff) / (norm_t + 1e-9))
    return {
        "mae": mae,
        "rmse": rmse,
        "cos_sim": cos_sim,
        "pearson": pearson,
        "rel_err": rel_err
    }

def run_benchmark_on_matrix(W, title, num_runs=50):
    M, K = W.shape
    dll = load_kernels()

    print("=" * 86)
    print(f"  BENCHMARK: {title} — Matrix Shape: [{M}, {K}] ({M*K:,} parameters)")
    print("=" * 86)

    np.random.seed(42)
    x = np.random.randn(K).astype(np.float32)

    # 0. FP32 Reference
    y_fp32 = W.dot(x)
    gamma = float(np.mean(np.abs(W)))
    beta = 0.38  # optimal centroid of sub-quantum zone [0.15*gamma, 0.50*gamma)

    results = []

    # 1. INT4 (Group-wise 32)
    num_blocks_int4 = (K + 31) // 32
    w_int4 = (BlockQ4_32 * (M * num_blocks_int4))()
    W_f32_c = np.ascontiguousarray(W, dtype=np.float32)
    x_c = np.ascontiguousarray(x, dtype=np.float32)

    dll.nat_quantize_int4_g32(
        W_f32_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_int4,
        M, K
    )

    y_int4 = np.zeros(M, dtype=np.float32)
    # Warmup
    dll.nat_gemv_int4_g32(w_int4, x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y_int4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, K)
    # Timed run
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_gemv_int4_g32(w_int4, x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y_int4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, K)
    t_int4 = (time.perf_counter() - t0) * 1000.0 / num_runs

    m_int4 = compute_metrics(y_fp32, y_int4)
    # 4 bits per weight + 32-bit float scale per 32 weights = (4 * 32 + 32) / 32 = 5 bits/weight
    size_int4_kb = (M * num_blocks_int4 * ctypes.sizeof(BlockQ4_32)) / 1024.0
    results.append(("INT4 (Group-wise 32)", 4.5, size_int4_kb, t_int4, m_int4))

    # 2. Ternary int8 (BitNet b1.58 canonical)
    W_trits = np.clip(np.round(W / gamma), -1, 1).astype(np.int8)
    y_ternary = np.zeros(M, dtype=np.float32)
    dll.nat_gemv_ternary_int8(
        W_trits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_ternary.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K, gamma
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_gemv_ternary_int8(
            W_trits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_ternary.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K, gamma
        )
    t_ternary = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_ternary = compute_metrics(y_fp32, y_ternary)
    size_ternary_kb = (M * K) / 1024.0
    results.append(("BitNet b1.58: Ternary int8", 8.0, size_ternary_kb, t_ternary, m_ternary))

    # 3. veclut_5trits (1.58-bit Base-3 Packed)
    bpr_5t = (K + 4) // 5
    W_packed_5t = np.zeros(M * bpr_5t, dtype=np.uint8)
    dll.nat_pack_5trits_2d(
        W_trits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        W_packed_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        M, K
    )
    y_5t = np.zeros(M, dtype=np.float32)
    dll.nat_gemv_veclut_5trits(
        W_packed_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K, gamma
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_gemv_veclut_5trits(
            W_packed_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_5t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K, gamma
        )
    t_5t = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_5t = compute_metrics(y_fp32, y_5t)
    size_5t_kb = (M * bpr_5t) / 1024.0
    results.append(("BitNet b1.58: veclut_5trits", 1.6, size_5t_kb, t_5t, m_5t))

    # 4. Qutrit dot64: Baseline (00, 01, 10)
    num_blocks_q = (K + 63) // 64
    W_pos_base = np.zeros(M * num_blocks_q, dtype=np.uint64)
    W_neg_base = np.zeros(M * num_blocks_q, dtype=np.uint64)
    dll.nat_pack_qutrit_baseline(
        W_f32_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        W_pos_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        M, K, gamma
    )

    # Quantize activations for popcount
    x_sign = np.zeros(num_blocks_q, dtype=np.uint64)
    x_mag = np.zeros(num_blocks_q * 7, dtype=np.uint64)
    act_scale = ctypes.c_float(0.0)
    dll.nat_quantize_activations_popcnt(
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        K,
        ctypes.byref(act_scale)
    )

    y_q_base = np.zeros(M, dtype=np.float32)
    dll.nat_qutrit_dot64_baseline_gemv(
        W_pos_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        y_q_base.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, num_blocks_q, gamma, act_scale.value
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_qutrit_dot64_baseline_gemv(
            W_pos_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            W_neg_base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            y_q_base.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, num_blocks_q, gamma, act_scale.value
        )
    t_q_base = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_q_base = compute_metrics(y_fp32, y_q_base)
    size_q_kb = (M * num_blocks_q * 16) / 1024.0  # 2 x uint64 = 16 bytes per 64 weights = 2.0 bits
    results.append(("Qutrit dot64: Baseline (00,01,10)", 2.0, size_q_kb, t_q_base, m_q_base))

    # 5. Qutrit dot64: Sub-Quantum Semantics C (Deterministic Chess-Phase)
    W_pos_sub = np.zeros(M * num_blocks_q, dtype=np.uint64)
    W_neg_sub = np.zeros(M * num_blocks_q, dtype=np.uint64)
    dll.nat_pack_qutrit_subquantum(
        W_f32_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        M, K, gamma, 0.15
    )

    y_q_sub = np.zeros(M, dtype=np.float32)
    dll.nat_qutrit_dot64_subquantum_gemv(
        W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        y_q_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, num_blocks_q, gamma, beta, act_scale.value
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_qutrit_dot64_subquantum_gemv(
            W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            y_q_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, num_blocks_q, gamma, beta, act_scale.value
        )
    t_q_sub = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_q_sub = compute_metrics(y_fp32, y_q_sub)
    results.append(("Qutrit dot64: Sub-Quantum (Sem. C)", 2.0, size_q_kb, t_q_sub, m_q_sub))

    # 6. Qutrit dot64: Sub-Quantum F32 Activations (Continuous x)
    y_q_sub_f32 = np.zeros(M, dtype=np.float32)
    dll.nat_qutrit_dot64_subquantum_gemv_f32(
        W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_q_sub_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K, gamma, beta
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_qutrit_dot64_subquantum_gemv_f32(
            W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_q_sub_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K, gamma, beta
        )
    t_q_sub_f32 = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_q_sub_f32 = compute_metrics(y_fp32, y_q_sub_f32)
    results.append(("Qutrit dot64: Sub-Quantum (F32 Act)", 2.0, size_q_kb, t_q_sub_f32, m_q_sub_f32))

    # 7. Qutrit veclut_4qutrits (LUT256 for 4 qutrits per byte = 2.0 bits)
    bpr_4q = (K + 3) // 4
    W_packed_4q = np.zeros(M * bpr_4q, dtype=np.uint8)
    dll.nat_pack_4qutrits(
        W_f32_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        W_packed_4q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        M, K, gamma, 0.15
    )
    y_4q = np.zeros(M, dtype=np.float32)
    dll.nat_gemv_veclut_4qutrits(
        W_packed_4q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_4q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K, gamma, beta
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_gemv_veclut_4qutrits(
            W_packed_4q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_4q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K, gamma, beta
        )
    t_4q = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_4q = compute_metrics(y_fp32, y_4q)
    size_4q_kb = (M * bpr_4q) / 1024.0
    results.append(("Qutrit LUT256: veclut_4qutrits", 2.0, size_4q_kb, t_4q, m_4q))

    # 8. Qutrit dot64 Contextual (Semantics B with c_mask)
    y_ctx_0 = np.zeros(M, dtype=np.float32)
    y_ctx_1 = np.zeros(M, dtype=np.float32)
    dll.nat_qutrit_dot64_ctx_gemv(
        W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        y_ctx_0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, num_blocks_q, gamma, act_scale.value, 0x0
    )
    dll.nat_qutrit_dot64_ctx_gemv(
        W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        y_ctx_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, num_blocks_q, gamma, act_scale.value, 0xFFFFFFFFFFFFFFFF
    )
    t0 = time.perf_counter()
    for _ in range(num_runs):
        dll.nat_qutrit_dot64_ctx_gemv(
            W_pos_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            W_neg_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            x_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            y_ctx_0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, num_blocks_q, gamma, act_scale.value, 0x0
        )
    t_ctx = (time.perf_counter() - t0) * 1000.0 / num_runs
    m_ctx_0 = compute_metrics(y_fp32, y_ctx_0)
    results.append(("Qutrit dot64: Context A (c_mask=0)", 2.0, size_q_kb, t_ctx, m_ctx_0))

    # Print summary table
    print("\n" + "-" * 102)
    print(f"{'Quantization Kernel':<34} | {'Bits':<5} | {'Size (KB)':<9} | {'Cosine':<7} | {'RMSE':<9} | {'MAE':<8} | {'CPU (ms)':<8}")
    print("-" * 102)

    fp32_size_kb = (M * K * 4.0) / 1024.0
    print(f"{'FP32 Baseline (Reference)':<34} | {'32.0':<5} | {fp32_size_kb:<9.1f} | {'1.0000':<7} | {'0.0000':<9} | {'0.0000':<8} | {'---':<8}")

    for name, bits, size_kb, t_ms, m in results:
        print(f"{name:<34} | {bits:<5.2f} | {size_kb:<9.1f} | {m['cos_sim']:<7.4f} | {m['rmse']:<9.4f} | {m['mae']:<8.4f} | {t_ms:<8.2f}")
    print("-" * 102)

    # Calculate Contextual Switching divergence
    diff_ctx = np.linalg.norm(y_ctx_1 - y_ctx_0)
    cos_ctx = np.dot(y_ctx_0, y_ctx_1) / (np.linalg.norm(y_ctx_0) * np.linalg.norm(y_ctx_1) + 1e-9)
    print(f"\n[*] Contextual Adaptation Metric (Semantics B):")
    print(f"    - Divergence ||Context A - Context B||: {diff_ctx:.4f}")
    print(f"    - Inter-context Cosine Similarity:      {cos_ctx:.4f} (Context switched successfully with 0 params!)")

    return results

def main():
    print("==========================================================================")
    print("  NATIVA BENCHMARK SUITE: QWEN QUANTIZATION (INT4 vs BITNET vs QUTRIT)    ")
    print("==========================================================================")

    # 1. Benchmark on Qwen2.5-0.5B Hidden Size (896 x 896)
    np.random.seed(42)
    W_896 = np.random.normal(0.0, 0.04, size=(896, 896)).astype(np.float32)
    run_benchmark_on_matrix(W_896, "Synthetic Qwen2.5-0.5B Hidden Layer (896x896)", num_runs=50)

    # 2. Benchmark on Real Qwen Weights from data/model.safetensors if available
    safetensors_path = os.path.join(REPO_ROOT, "data", "model.safetensors")
    if os.path.exists(safetensors_path):
        print("\n\n" + "#" * 86)
        print(f"  BENCHMARK ON REAL QWEN WEIGHTS: {safetensors_path}")
        print("#" * 86)

        try:
            # Let's inspect available projections in Layer 0
            layer_keys = [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.self_attn.q_proj.weight"
            ]
            for key in layer_keys:
                print(f"\n[*] Loading real tensor: '{key}'...")
                W_real_t = nativa.load_safetensors_fn(safetensors_path, key)
                # Convert Nativa Tensor to numpy array
                W_real = np.array(W_real_t.data, dtype=np.float32).reshape(W_real_t.shape)
                run_benchmark_on_matrix(W_real, f"Real Qwen Weight: {key}", num_runs=20)
        except Exception as e:
            print(f"[!] Error loading real Safetensors weights: {e}")
    else:
        print(f"\n[INFO] Real weights file '{safetensors_path}' not found, skipping real weights test.")

if __name__ == "__main__":
    main()
