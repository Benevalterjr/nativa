#!/usr/bin/env python3
"""
tools/experimento_2_isolar_quantizacao.py — Experiment 2: Isolate Quantization
Keeps the cache policy FIXED (Full KV Cache: 100% token retention)
and strictly varies the K and V quantization precision:
1. FP32 (IEEE 754 baseline)
2. FP16 (Half precision)
3. INT8 (Symmetric linear quantization per scale)
4. TurboQuant 4-bit (RHT + Lloyd-Max Gaussian)
5. TurboQuant 2-bit (RHT + Lloyd-Max Gaussian)

Measures: Real PPL, NLL, Top-1 Acc, theoretical KV Cache memory, and real peak RAM (psutil).
"""

import os
import sys
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa
from tools.experimento_real_cortex_hipocampo import (
    QwenModelLoaded,
    rmsnorm,
    apply_rope_heads,
    MODEL_DEFAULT_PATH,
    TOKENIZER_DEFAULT_PATH,
    CORPUS_DEFAULT_PATH,
    NUM_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GROUP_SIZE
)


def quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric INT8 quantization with scale factor."""
    max_val = float(np.max(np.abs(x)))
    if max_val < 1e-12:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = max_val / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


@dataclass
class QuantizedKVEntry:
    pos: int
    precision: str
    # FP32 / FP16
    k_raw: Optional[np.ndarray] = None
    v_raw: Optional[np.ndarray] = None
    # INT8
    k_int8: Optional[np.ndarray] = None
    v_int8: Optional[np.ndarray] = None
    scale_k: float = 1.0
    scale_v: float = 1.0
    # TurboQuant (4b and 2b)
    comp_k: Optional[List[dict]] = None
    comp_v: Optional[List[dict]] = None


class QuantizedKVCache:
    """KV Cache with configurable precision and 100% Full Cache policy."""
    def __init__(self, precision: str):
        self.precision = precision
        self.entries: List[QuantizedKVEntry] = []

    def append(self, pos: int, k_rot: np.ndarray, v_val: np.ndarray):
        entry = QuantizedKVEntry(pos=pos, precision=self.precision)

        if self.precision == "fp32":
            entry.k_raw = k_rot.astype(np.float32)
            entry.v_raw = v_val.astype(np.float32)

        elif self.precision == "fp16":
            entry.k_raw = k_rot.astype(np.float16)
            entry.v_raw = v_val.astype(np.float16)

        elif self.precision == "int8":
            q_k, s_k = quantize_int8(k_rot)
            q_v, s_v = quantize_int8(v_val)
            entry.k_int8 = q_k
            entry.v_int8 = q_v
            entry.scale_k = s_k
            entry.scale_v = s_v

        elif self.precision == "turboquant_4b":
            entry.comp_k = [nativa.turbo_quant_fn(k_rot[h], bits=4) for h in range(NUM_KV_HEADS)]
            entry.comp_v = [nativa.turbo_quant_fn(v_val[h], bits=4) for h in range(NUM_KV_HEADS)]

        elif self.precision == "turboquant_2b":
            entry.comp_k = [nativa.turbo_quant_fn(k_rot[h], bits=2) for h in range(NUM_KV_HEADS)]
            entry.comp_v = [nativa.turbo_quant_fn(v_val[h], bits=2) for h in range(NUM_KV_HEADS)]

        self.entries.append(entry)

    def get_keys_and_values(self) -> Tuple[np.ndarray, np.ndarray]:
        k_list = []
        v_list = []
        for e in self.entries:
            if self.precision in ("fp32", "fp16"):
                k_list.append(e.k_raw.astype(np.float32))
                v_list.append(e.v_raw.astype(np.float32))
            elif self.precision == "int8":
                k_list.append(dequantize_int8(e.k_int8, e.scale_k))
                v_list.append(dequantize_int8(e.v_int8, e.scale_v))
            elif self.precision in ("turboquant_4b", "turboquant_2b"):
                k_decomp = np.stack([np.asarray(nativa.turbo_dequant_fn(e.comp_k[h]), dtype=np.float32) for h in range(NUM_KV_HEADS)], axis=0)
                v_decomp = np.stack([np.asarray(nativa.turbo_dequant_fn(e.comp_v[h]), dtype=np.float32) for h in range(NUM_KV_HEADS)], axis=0)
                k_list.append(k_decomp)
                v_list.append(v_decomp)
        K = np.stack(k_list, axis=0)
        V = np.stack(v_list, axis=0)
        return K, V

    def memory_bytes(self) -> int:
        n = len(self.entries)
        dim = NUM_KV_HEADS * HEAD_DIM  # 128 coords
        if self.precision == "fp32":
            return n * dim * 4 * 2  # 1024 bytes/token
        elif self.precision == "fp16":
            return n * dim * 2 * 2  # 512 bytes/token
        elif self.precision == "int8":
            return n * (dim * 1 * 2 + 8)  # 256 bytes + 8 bytes scales = 264 bytes/token
        elif self.precision == "turboquant_4b":
            return n * ((dim * 4 * 2) // 8 + 8)  # 128 bytes + 8 bytes scales = 136 bytes/token
        elif self.precision == "turboquant_2b":
            return n * ((dim * 2 * 2) // 8 + 8)  # 64 bytes + 8 bytes scales = 72 bytes/token
        return 0


def evaluate_precision(
    precision: str,
    tokens: List[int],
    model: QwenModelLoaded
) -> Dict[str, Any]:
    print(f"\n---> Evaluating KV Precision: {precision.upper()}...")
    caches = [QuantizedKVCache(precision=precision) for _ in range(model.num_layers)]
    
    n_tokens = len(tokens)
    nll_list = []
    correct_count = 0
    t_start = time.perf_counter()
    proc = psutil.Process(os.getpid())
    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)

    for i in range(n_tokens - 1):
        target_token = tokens[i + 1]
        pos = i
        token_id = tokens[i]

        # Forward pass
        x = model.embed_tokens[token_id].data.astype(np.float32)

        for l_idx, layer in enumerate(model.layers):
            norm_x = rmsnorm(x, layer.norm1)
            q = (layer.wq.data @ norm_x) + (layer.bq.data if layer.bq is not None else 0.0)
            k = (layer.wk.data @ norm_x) + (layer.bk.data if layer.bk is not None else 0.0)
            v = (layer.wv.data @ norm_x) + (layer.bv.data if layer.bv is not None else 0.0)

            q = q.reshape(NUM_HEADS, HEAD_DIM)
            k = k.reshape(NUM_KV_HEADS, HEAD_DIM)
            v = v.reshape(NUM_KV_HEADS, HEAD_DIM)

            q_rot = apply_rope_heads(q, pos)
            k_rot = apply_rope_heads(k, pos)

            cache = caches[l_idx]
            cache.append(pos, k_rot, v)

            K_all, V_all = cache.get_keys_and_values()
            K_exp = np.repeat(K_all, GROUP_SIZE, axis=1)
            V_exp = np.repeat(V_all, GROUP_SIZE, axis=1)

            scores = np.einsum('hd,chd->hc', q_rot, K_exp) / math.sqrt(HEAD_DIM)
            max_s = np.max(scores, axis=-1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            attn_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

            out_heads = np.einsum('hc,chd->hd', attn_probs, V_exp)
            out_flat = out_heads.reshape(-1)
            x = x + (layer.wo.data @ out_flat)

            norm_x2 = rmsnorm(x, layer.norm2)
            gate = layer.wgate.data @ norm_x2
            up = layer.wup.data @ norm_x2
            gated = (gate / (1.0 + np.exp(-np.clip(gate, -30.0, 30.0)))) * up
            x = x + (layer.wdown.data @ gated)

        final_x = rmsnorm(x, model.final_norm)
        logits = model.embed_tokens.data @ final_x

        # Cross entropy
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l)
        prob_target = float(probs[target_token])
        nll = -math.log(max(prob_target, 1e-15))
        nll_list.append(nll)

        if int(np.argmax(logits)) == target_token:
            correct_count += 1

        rss_mb = proc.memory_info().rss / (1024 * 1024)
        if rss_mb > peak_rss_mb:
            peak_rss_mb = rss_mb

        if (i + 1) % 25 == 0 or (i + 1) == (n_tokens - 1):
            curr_ppl = math.exp(np.mean(nll_list))
            mem_kb = sum(c.memory_bytes() for c in caches) / 1024.0
            print(f"    Step {i + 1:2d}/{n_tokens - 1} | Partial PPL: {curr_ppl:.2f} | KV: {mem_kb:.1f} KB | Process RAM: {rss_mb:.1f} MB")

    t_end = time.perf_counter()
    mean_nll = float(np.mean(nll_list))
    ppl = math.exp(mean_nll)
    acc = 100.0 * correct_count / (n_tokens - 1)
    total_kv_kb = sum(c.memory_bytes() for c in caches) / 1024.0

    return {
        "precision": precision,
        "ppl": ppl,
        "nll": mean_nll,
        "accuracy": acc,
        "kv_kb": total_kv_kb,
        "peak_rss_mb": peak_rss_mb,
        "elapsed_s": t_end - t_start
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Isolate KV Cache Quantization")
    parser.add_argument("--tokens", type=int, default=80, help="Number of WikiText-2 tokens")
    parser.add_argument("--layers", type=int, default=24, help="Number of layers")
    args = parser.parse_args()

    print("=" * 85)
    print("EXPERIMENT 2: ISOLATING KV CACHE QUANTIZATION (100% RETENTION POLICY)")
    print(f"Model: Qwen2.5-0.5B ({args.layers} layers) | Dataset: WikiText-2 | Tokens: {args.tokens}")
    print("=" * 85)

    tok = nativa.Tokenizer(TOKENIZER_DEFAULT_PATH)
    with open(CORPUS_DEFAULT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 200 and not p.strip().startswith("=")]
    target_text = "\n\n".join(paragraphs[:3])
    eval_tokens = tok.encode(target_text).tolist()[:args.tokens]

    model = QwenModelLoaded(MODEL_DEFAULT_PATH, num_layers=args.layers)

    precisions = ["fp32", "fp16", "int8", "turboquant_4b", "turboquant_2b"]
    results = []
    for prec in precisions:
        res = evaluate_precision(prec, eval_tokens, model)
        results.append(res)

    print("\n" + "=" * 95)
    print(f"{'KV PRECISION':<16} | {'PPL':<14} | {'NLL (nats)':<10} | {'TOP-1 ACC':<10} | {'KV CACHE':<12} | {'REAL PEAK RAM':<14} | {'COMPRESSION'}")
    print("=" * 95)

    base_kv = results[0]["kv_kb"]
    base_ppl = results[0]["ppl"]

    for r in results:
        diff_ppl = r["ppl"] - base_ppl
        diff_str = f"(+{diff_ppl:.2f})" if diff_ppl > 0.001 else "(Base)"
        ratio = base_kv / max(r["kv_kb"], 0.001)
        ratio_str = f"{ratio:.1f}x" if ratio > 1.0 else "1.0x (Base)"
        print(f"{r['precision']:<16} | {r['ppl']:<6.2f} {diff_str:<7} | {r['nll']:<10.3f} | {r['accuracy']:<6.1f}%    | {r['kv_kb']:<8.1f} KB   | {r['peak_rss_mb']:<10.1f} MB   | {ratio_str}")
    print("=" * 95)


if __name__ == "__main__":
    main()
