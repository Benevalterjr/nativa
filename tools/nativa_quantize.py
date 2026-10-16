#!/usr/bin/env python3
"""Converte Qwen2/Qwen3 CausalLM para NATQ4X4 (Q4 weights, BF16 norms)."""

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
        raise ValueError(f"Tentativa de travessia de diretório detectada: {segments}")
    return target


def read_safe_header(path: str) -> tuple[int, dict[str, Any]]:
    clean_path = os.path.abspath(path)
    with open(clean_path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Cabeçalho SafeTensors truncado: {clean_path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        file_size = os.fstat(f.fileno()).st_size
        if header_len <= 0 or header_len > file_size - 8:
            raise ValueError(f"Tamanho de cabeçalho SafeTensors inválido: {clean_path}")
        header = json.loads(f.read(header_len).decode("utf-8"))
    return 8 + header_len, header


def make_tensor_map(model_dir: str) -> dict[str, tuple[str, int, dict[str, Any]]]:
    clean_model_dir = os.path.abspath(model_dir)
    index_path = _safe_resolve_path(clean_model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        if not weight_map:
            raise ValueError("O índice SafeTensors não contém weight_map.")
    else:
        single = _safe_resolve_path(clean_model_dir, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError("Não encontrei model.safetensors nem model.safetensors.index.json.")
        weight_map = {key: "model.safetensors" for key in read_safe_header(single)[1] if key != "__metadata__"}

    shard_headers: dict[str, tuple[int, dict[str, Any]]] = {}
    tensors: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for key, shard_name in weight_map.items():
        shard_path = _safe_resolve_path(clean_model_dir, shard_name)
        if shard_name not in shard_headers:
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard ausente: {shard_path}")
            shard_headers[shard_name] = read_safe_header(shard_path)
        base, header = shard_headers[shard_name]
        entry = header.get(key)
        if not isinstance(entry, dict) or "data_offsets" not in entry:
            raise ValueError(f"Tensor {key} não existe no shard {shard_name}.")
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
            raise ValueError(f"Peso obrigatório ausente: {key}")
        path, base, entry = self.tensor_map[key]
        dtype_name = entry.get("dtype")
        shape = tuple(int(n) for n in entry.get("shape", ()))
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"Forma de {key}: esperada {expected_shape}, recebida {shape}.")
        start, end = map(int, entry["data_offsets"])
        if start < 0 or end < start:
            raise ValueError(f"Offsets inválidos para {key}.")
        dtype_map = {"BF16": np.dtype("<u2"), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
        if dtype_name not in dtype_map:
            raise ValueError(f"Tipo {dtype_name!r} não suportado em {key}; use BF16, F16 ou F32.")
        dtype = dtype_map[dtype_name]
        count = int(np.prod(shape, dtype=np.int64))
        if end - start != count * dtype.itemsize:
            raise ValueError(f"Tamanho de dados incompatível com a forma de {key}.")
        f = self.handles.get(path)
        if f is None:
            f = open(path, "rb")
            self.handles[path] = f
        f.seek(base + start)
        raw = f.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Dados truncados para {key}.")
        values = np.frombuffer(raw, dtype=dtype)
        if dtype_name == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        else:
            values = values.astype(np.float32, copy=False)
        return values.reshape(shape)


def q4x4_matrix(values: np.ndarray) -> bytes:
    rows, cols = values.shape
    if rows % 4 or cols % 32:
        raise ValueError(f"Dimensões {rows}x{cols} não são compatíveis com Q4x4.")
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
        raise ValueError(f"Dimensões {rows}x{cols} não são compatíveis com Q4x4.")
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
        raise ValueError("config.json não declara eos_token_id e tokenizer.json não fornece um token EOS.")
    if len(ids) > 2:
        raise ValueError("O runtime NATQ4 aceita até dois IDs de parada EOS por modelo.")
    if any(i < 0 or i >= int(cfg["vocab_size"]) for i in ids):
        raise ValueError("eos_token_id está fora do vocabulário declarado.")
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
        raise ValueError("tokenizer.json não contém IDs de vocabulário reconhecíveis.")
    unique_ids = set(ids)
    size = max(ids) + 1
    if size < 4 or size > model_vocab_size:
        raise ValueError(f"O tokenizer cobre {size} IDs, incompatíveis com vocab_size={model_vocab_size}.")
    if min(unique_ids) < 0 or len(unique_ids) != size:
        raise ValueError("Os IDs do tokenizer são esparsos; o runtime exige IDs contíguos a partir de zero.")
    return size


def quantize_qwen(model_dir: str, out_file: str, max_seq_len: int = 2048) -> None:
    clean_model_dir = os.path.abspath(model_dir)
    config_path = _safe_resolve_path(clean_model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json não encontrado em {clean_model_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type")
    archs = cfg.get("architectures", [])
    arch_by_type = {"qwen2": ("Qwen2ForCausalLM", ARCH_QWEN2), "qwen3": ("Qwen3ForCausalLM", ARCH_QWEN3)}
    if model_type not in arch_by_type or not any(arch_by_type[model_type][0] == a for a in archs):
        raise ValueError("Este conversor aceita Qwen2ForCausalLM e Qwen3ForCausalLM.")
    arch_id = arch_by_type[model_type][1]
    if cfg.get("hidden_act", "silu") != "silu":
        raise ValueError("O motor NATQ4X4 implementa SwiGLU/SiLU; hidden_act incompatível.")
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    if rope and rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear"):
        raise ValueError("A configuração de RoPE não é suportada por este motor.")
    if rope and rope.get("rope_type", rope.get("type", "default")) == "linear":
        raise ValueError("RoPE linear não é suportado por este motor; use RoPE padrão.")
    if cfg.get("use_sliding_window") or "sliding_attention" in cfg.get("layer_types", []):
        raise ValueError("Camadas Qwen3 com sliding window ainda não são suportadas.")

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
        raise ValueError("Dimensões do modelo incompatíveis com o layout/runtime Q4x4.")
    if max_seq_len < 2 or max_seq_len > model_context or max_seq_len > MAX_CONTEXT:
        raise ValueError(f"max_seq_len deve estar entre 2 e {min(model_context, MAX_CONTEXT)}.")

    tie_embeddings = bool(cfg.get("tie_word_embeddings", model_type == "qwen2"))
    has_bias = bool(cfg.get("attention_bias", model_type == "qwen2"))
    if model_type == "qwen3" and has_bias:
        raise ValueError("Qwen3 com attention_bias=true requer suporte ao bias de o_proj, ainda não implementado.")
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

    print(f"[OK] {model_type.upper()} quantizado para NATQ4X4: {out_abs}")
    print(f"     {layers_n} camadas | {file_size:,} bytes | contexto {max_seq_len} | EOS {eos1}, {eos2}")
    print(f"     Vocabulário: {tokenizer_size} IDs ativos de {vocab} linhas de embedding")
    if os.path.isfile(out_abs + ".tokenizer.json"):
        print(f"     Tokenizer salvo em {out_abs}.tokenizer.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantiza Qwen2/Qwen3ForCausalLM BF16/F16/F32 para NATQ4X4 Q4x4.")
    parser.add_argument("model_dir", help="Diretório com config.json e SafeTensors")
    parser.add_argument("output_pos", nargs="?", help="Arquivo .natq4 de saída (opcional)")
    parser.add_argument("--out", dest="output_flag", help="Arquivo .natq4 de saída")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Contexto alocado pelo motor (padrão: 2048)")
    args = parser.parse_args()
    if args.output_pos and args.output_flag:
        parser.error("informe o destino como argumento posicional ou com --out, não ambos")
    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        parser.error(f"Diretório do modelo não encontrado: {args.model_dir}")
    output = os.path.abspath(args.output_flag or args.output_pos or os.path.join(model_dir, "model.natq4"))
    try:
        quantize_qwen(model_dir, output, args.max_seq_len)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"[ERRO] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
