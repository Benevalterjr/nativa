#!/usr/bin/env python3
"""Short next-token loss evaluation of Qwen using Nativa runtime operations.

Python only coordinates reading the parquet file and the token loop. Tensors, weights,
matmuls, RMSNorm, RoPE, SiLU, and softmax come from the runtime in nativa.py.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa


MODEL_FILE = os.path.join("data", "qwen2.5_0.5b.safetensors")
TOKENIZER_FILE = os.path.join("data", "tokenizer_qwen.json")
CORPUS_FILE = os.path.join("data", "wikitext-2-raw-v1-test.parquet")
ROPE_BASE = 1_000_000.0
HEAD_DIM = 64
NUM_HEADS = 14
NUM_KV_HEADS = 2


def tensor_from_file(name):
    return nativa.load_safetensors_fn(MODEL_FILE, name)


def linear(weights, values, bias=None):
    result = weights @ values
    if bias is None:
        return result
    if result.ndim == 2:
        return nativa.Tensor(result.data + bias.data[:, None])
    return result + bias


def rmsnorm_rows(values, weight):
    rms = np.sqrt(np.mean(values ** 2, axis=-1, keepdims=True) + 1e-6)
    return (values / rms) * weight.data


def rotate_heads(values, heads, seq_len):
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_BASE ** (np.arange(0, half, dtype=np.float32) / float(half)))
    pos_arr = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = pos_arr * freqs
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]

    v_3d = values.reshape(seq_len, heads, HEAD_DIM)
    x1 = v_3d[..., :half]
    x2 = v_3d[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return np.concatenate([out1, out2], axis=-1).reshape(seq_len, -1)


def forward_layer(hidden, layer_index):
    prefix = f"model.layers.{layer_index}."
    norm1 = tensor_from_file(prefix + "input_layernorm.weight")
    norm2 = tensor_from_file(prefix + "post_attention_layernorm.weight")
    wq = tensor_from_file(prefix + "self_attn.q_proj.weight")
    bq = tensor_from_file(prefix + "self_attn.q_proj.bias")
    wk = tensor_from_file(prefix + "self_attn.k_proj.weight")
    bk = tensor_from_file(prefix + "self_attn.k_proj.bias")
    wv = tensor_from_file(prefix + "self_attn.v_proj.weight")
    bv = tensor_from_file(prefix + "self_attn.v_proj.bias")
    wo = tensor_from_file(prefix + "self_attn.o_proj.weight")
    wgate = tensor_from_file(prefix + "mlp.gate_proj.weight")
    wup = tensor_from_file(prefix + "mlp.up_proj.weight")
    wdown = tensor_from_file(prefix + "mlp.down_proj.weight")

    # Batch linear projections using the Nativa Tensor @ implementation.
    normalized = rmsnorm_rows(hidden, norm1)
    normalized_t = nativa.Tensor(normalized.T)
    q = linear(wq, normalized_t, bq).data.T
    k = linear(wk, normalized_t, bk).data.T
    v = linear(wv, normalized_t, bv).data.T

    seq_len = hidden.shape[0]
    q_rot = rotate_heads(q, NUM_HEADS, seq_len)
    k_rot = rotate_heads(k, NUM_KV_HEADS, seq_len)

    # Causal grouped-query attention (7 query heads per KV head).
    group_size = NUM_HEADS // NUM_KV_HEADS
    Q = q_rot.reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
    K = k_rot.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM).repeat(group_size, axis=1).transpose(1, 0, 2)
    V = v.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM).repeat(group_size, axis=1).transpose(1, 0, 2)

    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(HEAD_DIM)
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    scores = scores + mask
    max_s = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    attention = np.matmul(probs, V).transpose(1, 0, 2).reshape(seq_len, -1)

    attended = (wo @ nativa.Tensor(attention.T)).data.T
    residual = hidden + attended

    normalized2 = rmsnorm_rows(residual, norm2)
    normalized2_t = nativa.Tensor(normalized2.T)
    gate = (wgate @ normalized2_t).data.T
    up = (wup @ normalized2_t).data.T
    gated = nativa.silu_fn(nativa.Tensor(gate)).data * up
    mlp = (wdown @ nativa.Tensor(gated.T)).data.T
    return residual + mlp


def load_test_tokens(max_tokens):
    table = pq.read_table(CORPUS_FILE, columns=["text"])
    paragraphs = [text for text in table.column("text").to_pylist() if text and text.strip()]
    tokenizer = nativa.Tokenizer(TOKENIZER_FILE)
    token_ids = []
    for paragraph in paragraphs:
        token_ids.extend(tokenizer.encode(paragraph.rstrip("\r\n") + "\n\n").tolist())
        if max_tokens and len(token_ids) >= max_tokens:
            break
    if len(token_ids) < 2:
        raise RuntimeError("The WikiText split did not produce enough tokens.")
    if max_tokens:
        token_ids = token_ids[:max_tokens]
    return token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="tokens from the test split (default: 16; 0 uses the entire split)")
    args = parser.parse_args()

    for path in (MODEL_FILE, TOKENIZER_FILE, CORPUS_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    if args.max_tokens < 0:
        raise ValueError("--max-tokens must be >= 0")

    token_ids = load_test_tokens(args.max_tokens)
    print("Qwen2.5-0.5B / WikiText-2 raw test / Nativa runtime")
    print(f"Evaluated tokens: {len(token_ids)} (next-token targets: {len(token_ids) - 1})")
    print("Full causal context of the slice; BF16 weights converted per layer on demand.")

    started = time.perf_counter()
    embeddings = tensor_from_file("model.embed_tokens.weight")
    hidden = np.stack([embeddings[token_id].data for token_id in token_ids]).astype(np.float32)

    for layer in range(24):
        hidden = forward_layer(hidden, layer)
        print(f"Layer {layer + 1:02d}/24 completed")

    final_norm = tensor_from_file("model.norm.weight")
    hidden = rmsnorm_rows(hidden, final_norm)
    # The model uses tied embeddings for the language head (no lm_head.weight in the file).
    logits = embeddings @ nativa.Tensor(hidden[:-1].T)

    nll_sum = 0.0
    correct = 0
    for col, target in enumerate(token_ids[1:]):
        probabilities = nativa.softmax_fn(logits[:, col])
        target_probability = float(probabilities.data[target])
        nll_sum -= math.log(max(target_probability, 1e-45))
        correct += int(int(np.argmax(logits.data[:, col])) == target)

    count = len(token_ids) - 1
    mean_nll = nll_sum / count
    elapsed = time.perf_counter() - started
    print(f"Mean NLL (nats/token): {mean_nll:.6f}")
    print(f"Slice perplexity: {math.exp(mean_nll):.4f}")
    print(f"Top-1 next-token accuracy: {100.0 * correct / count:.2f}%")
    print(f"Total time: {elapsed:.2f}s | {count / elapsed:.2f} target-tokens/s")
    print("Note: short evaluation of the test split prefix; not the official full-corpus perplexity.")
    nativa.close_safetensors_mmap(MODEL_FILE)


if __name__ == "__main__":
    main()
