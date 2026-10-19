#!/usr/bin/env python3
"""Converts Qwen2/Qwen3 CausalLM to NATQ4X4 (Q4 weights, BF16 norms)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from typing import Any

import numpy as np

HEADER_SIZE = 16_384
HEADER_VERSION = 3
ARCH_QWEN2 = 1
ARCH_QWEN3 = 6
MAX_LAYERS = 128
LAYER_OFFSET = 100
LAYER_STRIDE = 96
EOS_FIELDS_OFFSET = LAYER_OFFSET + MAX_LAYERS * LAYER_STRIDE
QK_NORM_OFFSET = EOS_FIELDS_OFFSET + 8
TOKENIZER_VOCAB_SIZE_OFFSET = QK_NORM_OFFSET + MAX_LAYERS * 16
MAX_CONTEXT = 32_768
TOKENIZER_JSON = "tokenizer.json"


def _safe_resolve_path(base_dir: str, *segments: str) -> str:
    base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base, *segments))
    if os.path.commonpath([base, target]) != base:
        raise ValueError(f"Directory traversal attempt detected: {segments}")
    return target


def read_safe_header(path: str) -> tuple[int, dict[str, Any]]:
    clean_path = os.path.abspath(path)
    with open(clean_path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Truncated SafeTensors header: {clean_path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        file_size = os.fstat(f.fileno()).st_size
        if header_len <= 0 or header_len > file_size - 8:
            raise ValueError(f"Invalid SafeTensors header size: {clean_path}")
        header = json.loads(f.read(header_len).decode("utf-8"))
    return 8 + header_len, header


def make_tensor_map(model_dir: str) -> dict[str, tuple[str, int, dict[str, Any]]]:
    clean_model_dir = os.path.abspath(model_dir)
    index_path = _safe_resolve_path(clean_model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        if not weight_map:
            raise ValueError("SafeTensors index does not contain weight_map.")
    else:
        single = _safe_resolve_path(clean_model_dir, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError("Could not find model.safetensors or model.safetensors.index.json.")
        weight_map = {key: "model.safetensors" for key in read_safe_header(single)[1] if key != "__metadata__"}

    shard_headers: dict[str, tuple[int, dict[str, Any]]] = {}
    tensors: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for key, shard_name in weight_map.items():
        shard_path = _safe_resolve_path(clean_model_dir, shard_name)
        if shard_name not in shard_headers:
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            shard_headers[shard_name] = read_safe_header(shard_path)
        base, header = shard_headers[shard_name]
        entry = header.get(key)
        if not isinstance(entry, dict) or "data_offsets" not in entry:
            raise ValueError(f"Tensor {key} does not exist in shard {shard_name}.")
        tensors[key] = (shard_path, base, entry)
    return tensors


class SafeTensorReader:
    def __init__(self, tensor_map: dict[str, tuple[str, int, dict[str, Any]]]):
        self.tensor_map = tensor_map
        self.handles: dict[str, Any] = {}

    def close(self) -> None:
        for f in self.handles.values():
            f.close()
        self.handles.clear()

    def float32(self, key: str, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
        if key not in self.tensor_map:
            raise ValueError(f"Missing required weight: {key}")
        path, base, entry = self.tensor_map[key]
        dtype_name = entry.get("dtype")
        shape = tuple(int(n) for n in entry.get("shape", ()))
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"Shape of {key}: expected {expected_shape}, got {shape}.")
        start, end = map(int, entry["data_offsets"])
        if start < 0 or end < start:
            raise ValueError(f"Invalid offsets for {key}.")
        dtype_map = {"BF16": np.dtype("<u2"), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
        if dtype_name not in dtype_map:
            raise ValueError(f"Type {dtype_name!r} not supported in {key}; use BF16, F16, or F32.")
        dtype = dtype_map[dtype_name]
        count = int(np.prod(shape, dtype=np.int64))
        if end - start != count * dtype.itemsize:
            raise ValueError(f"Data size mismatch with shape of {key}.")
        f = self.handles.get(path)
        if f is None:
            f = open(path, "rb")
            self.handles[path] = f
        f.seek(base + start)
        raw = f.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Truncated data for {key}.")
        values = np.frombuffer(raw, dtype=dtype)
        if dtype_name == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        else:
            values = values.astype(np.float32, copy=False)
        return values.reshape(shape)


def q4x4_matrix(values: np.ndarray) -> bytes:
    rows, cols = values.shape
    if rows % 4 or cols % 32:
        raise ValueError(f"Dimensions {rows}x{cols} are not compatible with Q4x4.")
    blocks = values.reshape(rows // 4, 4, cols // 32, 32)
    amax = np.max(np.abs(blocks), axis=-1)
    scale32 = (amax / np.float32(7.0)).astype(np.float32)
    scale_bits = scale32.view(np.uint32) >> 16
    scale_bf16 = scale_bits.astype(np.uint16)
    scale_recon = (scale_bits << 16).view(np.float32)
    scale_recon = np.where(scale_recon > 1e-12, scale_recon, scale32)
    denom = np.where(scale_recon > 1e-12, scale_recon, 1.0)
    scaled = blocks / denom[..., None]
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    quantized = np.clip(rounded + 8, 0, 15).astype(np.uint8)
    packed = quantized[..., :16] | (quantized[..., 16:] << 4)

    groups, _, num_blocks, _ = blocks.shape
    result = np.empty((groups, num_blocks, 72), dtype=np.uint8)
    scale_trans = np.ascontiguousarray(scale_bf16.transpose(0, 2, 1)).astype("<u2", copy=False)
    result[:, :, :8] = scale_trans.view(np.uint8).reshape(groups, num_blocks, 8)
    result[:, :, 8:] = np.ascontiguousarray(packed.transpose(0, 2, 1, 3)).reshape(groups, num_blocks, 64)
    return result.tobytes()


def bf16_array(values: np.ndarray) -> bytes:
    f32 = np.asarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    return (bits >> 16).astype("<u2").tobytes()


def q4_size(rows: int, cols: int) -> int:
    if rows % 4 or cols % 32:
        raise ValueError(f"Dimensions {rows}x{cols} are not compatible with Q4x4.")
    return rows * (cols // 32) * 18


def eos_ids(model_dir: str, cfg: dict[str, Any]) -> tuple[int, int]:
    clean_model_dir = os.path.abspath(model_dir)
    raw = cfg.get("eos_token_id")
    candidates = raw if isinstance(raw, list) else [raw]
    ids = [int(x) for x in candidates if x is not None]
    tok_path = _safe_resolve_path(clean_model_dir, TOKENIZER_JSON)
    if os.path.isfile(tok_path):
        with open(tok_path, "r", encoding="utf-8") as f:
            tok = json.load(f)
        for added in tok.get("added_tokens", []):
            if added.get("content") == "<|endoftext|>":
                ids.append(int(added["id"]))
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("config.json does not declare eos_token_id and tokenizer.json does not provide an EOS token.")
    if len(ids) > 2:
        raise ValueError("The NATQ4 runtime accepts up to two EOS stop IDs per model.")
    if any(i < 0 or i >= int(cfg["vocab_size"]) for i in ids):
        raise ValueError("eos_token_id is outside the declared vocabulary.")
    return ids[0], ids[1] if len(ids) > 1 else 0xFFFFFFFF


def tokenizer_vocab_size(model_dir: str, model_vocab_size: int) -> int:
    clean_model_dir = os.path.abspath(model_dir)
    path = _safe_resolve_path(clean_model_dir, TOKENIZER_JSON)
    if not os.path.isfile(path):
        return model_vocab_size
    with open(path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    vocab = tokenizer.get("model", {}).get("vocab", {})
    ids = [int(value) for value in vocab.values()]
    ids.extend(int(item["id"]) for item in tokenizer.get("added_tokens", []) if "id" in item)
    if not ids:
        raise ValueError("tokenizer.json does not contain recognizable vocabulary IDs.")
    unique_ids = set(ids)
    size = max(ids) + 1
    if size < 4 or size > model_vocab_size:
        raise ValueError(f"Tokenizer covers {size} IDs, incompatible with vocab_size={model_vocab_size}.")
    if min(unique_ids) < 0 or len(unique_ids) != size:
        raise ValueError("Tokenizer IDs are sparse; runtime requires contiguous IDs starting from zero.")
    return size


def quantize_qwen(model_dir: str, out_file: str, max_seq_len: int = 2048) -> None:
    clean_model_dir = os.path.abspath(model_dir)
    config_path = _safe_resolve_path(clean_model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {clean_model_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type")
    archs = cfg.get("architectures", [])
    arch_by_type = {"qwen2": ("Qwen2ForCausalLM", ARCH_QWEN2), "qwen3": ("Qwen3ForCausalLM", ARCH_QWEN3)}
    if model_type not in arch_by_type or not any(arch_by_type[model_type][0] == a for a in archs):
        raise ValueError("This converter accepts Qwen2ForCausalLM and Qwen3ForCausalLM.")
    arch_id = arch_by_type[model_type][1]
    if cfg.get("hidden_act", "silu") != "silu":
        raise ValueError("The NATQ4X4 engine implements SwiGLU/SiLU; incompatible hidden_act.")
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    if rope and rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear"):
        raise ValueError("RoPE configuration is not supported by this engine.")
    if rope and rope.get("rope_type", rope.get("type", "default")) == "linear":
        raise ValueError("Linear RoPE is not supported by this engine; use default RoPE.")
    if cfg.get("use_sliding_window") or "sliding_attention" in cfg.get("layer_types", []):
        raise ValueError("Qwen3 layers with sliding window are not supported yet.")

    dim = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    layers_n = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    head_dim = int(cfg.get("head_dim", dim // heads))
    q_dim = heads * head_dim
    vocab = int(cfg["vocab_size"])
    model_context = int(cfg.get("max_position_embeddings", MAX_CONTEXT))
    if (layers_n < 1 or layers_n > MAX_LAYERS or dim < 32 or dim % 32 or inter < 32 or inter % 32
            or vocab < 4 or vocab % 4 or heads < 1 or kv_heads < 1 or heads % kv_heads
            or (model_type != "qwen3" and head_dim != dim // heads) or q_dim % 32 != 0 or head_dim % 16 or (kv_heads * head_dim) % 4):
        raise ValueError("Model dimensions are incompatible with the Q4x4 layout/runtime.")
    if max_seq_len < 2 or max_seq_len > model_context or max_seq_len > MAX_CONTEXT:
        raise ValueError(f"max_seq_len must be between 2 and {min(model_context, MAX_CONTEXT)}.")

    tie_embeddings = bool(cfg.get("tie_word_embeddings", model_type == "qwen2"))
    has_bias = bool(cfg.get("attention_bias", model_type == "qwen2"))
    if model_type == "qwen3" and has_bias:
        raise ValueError("Qwen3 with attention_bias=true requires o_proj bias support, which is not yet implemented.")
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(rope.get("rope_theta", cfg.get("rope_theta", 1_000_000.0)))
    eos1, eos2 = eos_ids(model_dir, cfg)
    tokenizer_size = tokenizer_vocab_size(model_dir, vocab)
    tensor_map = make_tensor_map(model_dir)
    reader = SafeTensorReader(tensor_map)

    def write_matrix(out: Any, key: str, rows: int, cols: int) -> int:
        values = reader.float32(key, (rows, cols))
        blob = q4x4_matrix(values)
        offset = out.tell()
        out.write(blob)
        return offset

    def write_norm(out: Any, key: str, size: int) -> int:
        values = reader.float32(key, (size,))
        offset = out.tell()
        out.write(bf16_array(values))
        return offset

    out_abs = os.path.abspath(out_file)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".natq4-", suffix=".tmp", dir=os.path.dirname(out_abs))
    try:
        with os.fdopen(fd, "w+b") as out:
            out.write(bytes(HEADER_SIZE))
            embed_off = write_matrix(out, "model.embed_tokens.weight", vocab, dim)
            norm_off = write_norm(out, "model.norm.weight", dim)
            layer_offsets: list[dict[str, int]] = []

            for li in range(layers_n):
                prefix = f"model.layers.{li}."
                lo: dict[str, int] = {}
                lo["norm1"] = write_norm(out, prefix + "input_layernorm.weight", dim)
                lo["norm2"] = write_norm(out, prefix + "post_attention_layernorm.weight", dim)

                for short, proj, size in (("bq", "q_proj", q_dim), ("bk", "k_proj", kv_heads * head_dim), ("bv", "v_proj", kv_heads * head_dim)):
                    key = prefix + f"self_attn.{proj}.bias"
                    if has_bias:
                        lo[short] = out.tell()
                        out.write(bf16_array(reader.float32(key, (size,))))
                    else:
                        lo[short] = 0

                for short, key, rows, cols in (
                    ("wq", "q_proj", q_dim, dim),
                    ("wk", "k_proj", kv_heads * head_dim, dim),
                    ("wv", "v_proj", kv_heads * head_dim, dim),
                    ("wo", "o_proj", dim, q_dim),
                ):
                    lo[short] = write_matrix(out, prefix + f"self_attn.{key}.weight", rows, cols)

                for short, name, rows, cols in (
                    ("wgate", "gate_proj", inter, dim),
                    ("wup", "up_proj", inter, dim),
                    ("wdown", "down_proj", dim, inter),
                ):
                    lo[short] = write_matrix(out, prefix + f"mlp.{name}.weight", rows, cols)
                q_norm_off = k_norm_off = 0
                if model_type == "qwen3":
                    q_norm_off = write_norm(out, prefix + "self_attn.q_norm.weight", head_dim)
                    k_norm_off = write_norm(out, prefix + "self_attn.k_norm.weight", head_dim)
                lo["q_norm"] = q_norm_off
                lo["k_norm"] = k_norm_off
                layer_offsets.append(lo)

            lm_head_off = embed_off if tie_embeddings else write_matrix(out, "lm_head.weight", vocab, dim)
            file_size = out.tell()

            hdr = bytearray(HEADER_SIZE)
            struct.pack_into(
                "<8s13I2f4Q", hdr, 0,
                b"NATQ4X4\0", HEADER_VERSION, arch_id, vocab, dim, inter, layers_n, heads, kv_heads,
                head_dim, int(has_bias), int(tie_embeddings), 1, max_seq_len, rope_theta, eps,
                embed_off, norm_off, lm_head_off, HEADER_SIZE,
            )
            for li, lo in enumerate(layer_offsets):
                values = [lo[k] for k in ("norm1", "norm2", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "wgate", "wup", "wdown")]
                struct.pack_into("<12Q", hdr, LAYER_OFFSET + li * LAYER_STRIDE, *values)
                struct.pack_into("<2Q", hdr, QK_NORM_OFFSET + li * 16, lo["q_norm"], lo["k_norm"])
            struct.pack_into("<2I", hdr, EOS_FIELDS_OFFSET, eos1, eos2)
            struct.pack_into("<I", hdr, TOKENIZER_VOCAB_SIZE_OFFSET, tokenizer_size)
            out.seek(0)
            out.write(hdr)
            out.flush()
            os.fsync(out.fileno())
        reader.close()
        os.replace(temp_path, out_abs)
        tokenizer_src = os.path.join(model_dir, TOKENIZER_JSON)
        if os.path.isfile(tokenizer_src):
            tokenizer_dst = out_abs + ".tokenizer.json"
            fd_tok, temp_tok = tempfile.mkstemp(prefix=".tokenizer-", suffix=".tmp", dir=os.path.dirname(out_abs))
            os.close(fd_tok)
            try:
                shutil.copyfile(tokenizer_src, temp_tok)
                os.replace(temp_tok, tokenizer_dst)
            except Exception:
                try:
                    os.unlink(temp_tok)
                except OSError:
                    pass
                raise
    except Exception:
        reader.close()
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

    print(f"[OK] {model_type.upper()} quantized to NATQ4X4: {out_abs}")
    print(f"     {layers_n} layers | {file_size:,} bytes | context {max_seq_len} | EOS {eos1}, {eos2}")
    print(f"     Vocabulary: {tokenizer_size} active IDs out of {vocab} embedding rows")
    if os.path.isfile(out_abs + ".tokenizer.json"):
        print(f"     Tokenizer saved to {out_abs}.tokenizer.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantizes Qwen2/Qwen3ForCausalLM BF16/F16/F32 to NATQ4X4 Q4x4.")
    parser.add_argument("model_dir", help="Directory containing config.json and SafeTensors")
    parser.add_argument("output_pos", nargs="?", help="Output .natq4 file (optional)")
    parser.add_argument("--out", dest="output_flag", help="Output .natq4 file")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Context allocated by the engine (default: 2048)")
    args = parser.parse_args()
    if args.output_pos and args.output_flag:
        parser.error("provide destination as a positional argument or with --out, not both")
    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        parser.error(f"Model directory not found: {args.model_dir}")
    output = os.path.abspath(args.output_flag or args.output_pos or os.path.join(model_dir, "model.natq4"))
    try:
        quantize_qwen(model_dir, output, args.max_seq_len)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"[ERROR] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
