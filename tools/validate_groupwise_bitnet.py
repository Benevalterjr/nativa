#!/usr/bin/env python3
"""
Validation script for Group-Wise BitNet b1.58 Scaling in Nativa.
Addresses Issue #6: [Quantization]: Implement group-wise BitNet b1.58 scaling (group_size=128).

Compares:
1. FP32 baseline
2. Canonical Global BitNet b1.58 (group_size=None)
3. Group-wise BitNet b1.58 (group_size=128)
4. Group-wise BitNet b1.58 (group_size=64)

Evaluates:
- Cosine similarity against FP32 across typical LLM projection dimensions
- Mean Absolute Error (MAE) and Relative Error
- End-to-end matrix-vector product agreement (t2 and t2_packed_5t)
"""

import os
import sys
import numpy as np

# Ensure repository root is on path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import nativa
from nativa import Tensor, quantize_bitnet_fn, cosine_sim_fn, mae_fn, relative_error_fn


def evaluate_layer_fidelity(name: str, M: int, K: int, rng: np.random.RandomState):
    print(f"\n[*] Evaluating Layer: {name} (Shape: {M} x {K})")
    print("-" * 75)

    # Simulate realistic LLM weights with heterogeneous channel variance
    channel_scales = np.linspace(0.2, 4.0, K, dtype=np.float32)
    w_raw = (rng.randn(M, K).astype(np.float32)) * channel_scales
    x_raw = rng.randn(K).astype(np.float32)

    t_w = Tensor(w_raw, dtype="f32")
    t_x = Tensor(x_raw, dtype="f32")

    # FP32 baseline output
    y_fp32 = t_w @ t_x
    y_fp32_arr = np.asarray(y_fp32.data, dtype=np.float64)

    configs = [
        ("Global BitNet (b1.58)", None),
        ("Group-wise BitNet (G=128)", 128),
        ("Group-wise BitNet (G=64)", 64),
    ]

    print(f"{'Configuration':<28} | {'Cosine Sim':<12} | {'MAE':<12} | {'Rel Error':<12}")
    print("-" * 75)

    for desc, group_size in configs:
        q = t_w.quantize_bitnet(group_size=group_size)
        y_q = q @ t_x
        y_q_arr = np.asarray(y_q.data, dtype=np.float64)

        cos_sim = cosine_sim_fn(y_q_arr, y_fp32_arr)
        mae = mae_fn(y_q_arr, y_fp32_arr)
        rel_err = relative_error_fn(y_q_arr, y_fp32_arr)

        print(f"{desc:<28} | {cos_sim:<12.5f} | {mae:<12.5f} | {rel_err:<12.5f}")

        # Validate packed 5-trits parity
        p = q.pack_5trits()
        y_p = p @ t_x
        np.testing.assert_allclose(y_p.data, y_q.data, rtol=1e-4, atol=1e-4)


def main():
    print("=" * 75)
    print("  NATIVA BITNET b1.58 GROUP-WISE QUANTIZATION VALIDATION SUITE")
    print("=" * 75)

    rng = np.random.RandomState(42)

    # Standard LLM projection dimensions (Qwen, Llama, BitNet)
    test_layers = [
        ("MLP Gate/Up Projection (Qwen2.5-0.5B)", 4864, 896),
        ("MLP Down Projection (Qwen2.5-0.5B)", 896, 4864),
        ("Self-Attention Q/K/V Projection", 896, 896),
        ("Dense GEMM Test Layer", 512, 2048),
    ]

    for name, M, K in test_layers:
        evaluate_layer_fidelity(name, M, K, rng)

    print("\n[OK] All layer evaluations completed. Group-wise BitNet scaling verified successfully!")


if __name__ == "__main__":
    main()
