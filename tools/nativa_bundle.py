#!/usr/bin/env python3
"""
Nativa Bundle Packager - Generates single, self-contained executables (.exe / ELF)
Inspired by Mozilla llamafile (Cosmopolitan APE).
Packs: C99 Engine + Vocabulary + .natq4 Weights + 64B Trailer aligned to 64KB.
"""

import argparse
import os
import pathlib
import stat
import struct
import sys
from typing import Optional

# Alignment and format constants
NATIVA_BUNDLE_MAGIC = 0x4E4154495641464C  # "NATIVAF\0"
NATIVA_BUNDLE_ALIGN = 65536               # 64 KB alignment (MapViewOfFile / mmap page)
NATIVA_BUNDLE_VERSION = 1
TRAILER_FORMAT = "<QQQQQQII8s"
TRAILER_SIZE = struct.calcsize(TRAILER_FORMAT)  # exactly 64 bytes

# Add project root to sys.path to import nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import nativa


def export_vocab_bytes(tokenizer) -> bytes:
    """Exports the vocabulary to bytes in the binary format read by the C engine."""
    vocab_size = max(tokenizer.inv_vocab, default=-1) + 1
    chunks = [struct.pack("<I", vocab_size)]
    for token_id in range(vocab_size):
        piece = tokenizer.inv_vocab.get(token_id, "")
        if piece.startswith("<0x") and len(piece) == 6 and piece.endswith(">"):
            try:
                token = bytes([int(piece[3:5], 16)])
            except ValueError:
                token = piece.encode("utf-8", errors="replace")
        elif piece and hasattr(tokenizer, "byte_decoder") and all(c in tokenizer.byte_decoder for c in piece):
            token = bytes(tokenizer.byte_decoder[c] for c in piece)
        else:
            token = piece.encode("utf-8", errors="replace")
        if len(token) > 65535:
            raise ValueError(f"Token {token_id} exceeds 65535 bytes.")
        chunks.append(struct.pack("<H", len(token)))
        chunks.append(token)
    return b"".join(chunks)


def write_vocab_file(tokenizer, path: str) -> None:
    """Exports the vocabulary to the binary file read by the C engine."""
    data = export_vocab_bytes(tokenizer)
    with open(path, "wb") as f:
        f.write(data)


def align_up(offset: int, alignment: int = NATIVA_BUNDLE_ALIGN) -> int:
    """Calculates the next offset aligned to a multiple of alignment."""
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def inspect_bundle(bundle_path: str) -> Optional[dict]:
    """Inspects an existing bundle executable and returns trailer metadata if valid."""
    safe_path = pathlib.Path(bundle_path).resolve()
    if not safe_path.is_file():
        return None
    file_size = safe_path.stat().st_size
    if file_size < TRAILER_SIZE:
        return None

    with safe_path.open("rb") as f:
        f.seek(-TRAILER_SIZE, os.SEEK_END)
        data = f.read(TRAILER_SIZE)

    try:
        magic, engine_size, vocab_offset, vocab_size, model_offset, model_size, version, flags, _ = struct.unpack(
            TRAILER_FORMAT, data
        )
    except struct.error:
        return None

    if magic != NATIVA_BUNDLE_MAGIC or version != NATIVA_BUNDLE_VERSION:
        return None

    return {
        "bundle_path": str(safe_path),
        "total_size": file_size,
        "engine_size": engine_size,
        "vocab_offset": vocab_offset,
        "vocab_size": vocab_size,
        "model_offset": model_offset,
        "model_size": model_size,
        "version": version,
        "flags": flags,
    }


def create_bundle(
    model_path: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    flags: int = 0,
) -> str:
    """Creates a single self-contained executable by appending the vocabulary and model weights."""
    safe_model = pathlib.Path(model_path).resolve()
    if not safe_model.is_file():
        raise FileNotFoundError(f"Model file not found: {safe_model}")

    # Resolve the C engine executable
    if not engine_path:
        is_win = sys.platform.startswith("win")
        default_exe = "nativa_engine.exe" if is_win else "nativa_engine"
        engine_path = os.path.join(ROOT_DIR, "tools", default_exe)

    safe_engine = pathlib.Path(engine_path).resolve()
    if not safe_engine.is_file():
        raise FileNotFoundError(f"C engine executable not found: {safe_engine}")

    # Resolve the tokenizer
    if not tokenizer_path:
        candidate_same_dir = pathlib.Path(f"{safe_model}.tokenizer.json").resolve()
        candidate_qwen = pathlib.Path(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json").resolve()
        if candidate_same_dir.is_file():
            safe_tok = candidate_same_dir
        elif candidate_qwen.is_file():
            safe_tok = candidate_qwen
        else:
            raise FileNotFoundError(f"Tokenizer not found for model: {safe_model}")
    else:
        safe_tok = pathlib.Path(tokenizer_path).resolve()
        if not safe_tok.is_file():
            raise FileNotFoundError(f"Tokenizer not found for model: {safe_tok}")

    print(f"[*] Generating binary vocabulary from: {safe_tok}")
    tok = nativa.Tokenizer(str(safe_tok))
    vocab_bytes = export_vocab_bytes(tok)

    with safe_engine.open("rb") as ef:
        engine_bytes = ef.read()

    with safe_model.open("rb") as mf:
        model_bytes = mf.read()

    engine_size = len(engine_bytes)
    vocab_size = len(vocab_bytes)
    model_size = len(model_bytes)

    # Calculate offsets aligned to 64 KB
    vocab_offset = align_up(engine_size, NATIVA_BUNDLE_ALIGN)
    pad1_len = vocab_offset - engine_size

    model_offset = align_up(vocab_offset + vocab_size, NATIVA_BUNDLE_ALIGN)
    pad2_len = model_offset - (vocab_offset + vocab_size)

    # Build the fixed 64-byte trailer
    trailer = struct.pack(
        TRAILER_FORMAT,
        NATIVA_BUNDLE_MAGIC,
        engine_size,
        vocab_offset,
        vocab_size,
        model_offset,
        model_size,
        NATIVA_BUNDLE_VERSION,
        flags,
        b"\x00" * 8,
    )

    print("[*] Assembling self-contained bundle:")
    print(f"    -> C99 Engine    : {engine_size:,} bytes")
    print(f"    -> Vocabulary    : {vocab_size:,} bytes (offset: {vocab_offset:,}, aligned to 64KB)")
    print(f"    -> .natq4 Model  : {model_size:,} bytes (offset: {model_offset:,}, aligned to 64KB)")
    print(f"    -> Trailer       : {len(trailer)} bytes (Magic: 0x{NATIVA_BUNDLE_MAGIC:X})")

    safe_output = pathlib.Path(output_path).resolve()
    safe_output.parent.mkdir(parents=True, exist_ok=True)

    with safe_output.open("wb") as out_f:
        out_f.write(engine_bytes)
        if pad1_len > 0:
            out_f.write(b"\x00" * pad1_len)
        out_f.write(vocab_bytes)
        if pad2_len > 0:
            out_f.write(b"\x00" * pad2_len)
        out_f.write(model_bytes)
        out_f.write(trailer)

    # Grant safe execution permissions on POSIX environments (SonarCloud S2612: 0o750)
    try:
        safe_output.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    except OSError:
        pass

    total_mb = safe_output.stat().st_size / (1024 * 1024)
    print(f"[OK] Bundle created successfully: {safe_output} ({total_mb:.2f} MB)")
    return str(safe_output)


def main():
    parser = argparse.ArgumentParser(
        description="Nativa Bundle Packager - Generates single self-contained llamafile-style executables"
    )
    parser.add_argument("model", help="Path to the .natq4 model to package")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to the generated output executable (default: <model>.exe)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default=None,
        help="Path to the corresponding tokenizer.json",
    )
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        help="Path to the base nativa_engine executable to use",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspects and displays metadata of an existing bundle",
    )

    args = parser.parse_args()

    if args.inspect:
        info = inspect_bundle(args.model)
        if not info:
            print(f"[ERROR] File {args.model} is not a valid Nativa bundle.", file=sys.stderr)
            sys.exit(1)
        print("=== NATIVA BUNDLE METADATA ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    out_name = args.output
    if not out_name:
        is_win = sys.platform.startswith("win")
        base = os.path.splitext(args.model)[0]
        out_name = f"{base}.exe" if is_win else base

    create_bundle(
        model_path=args.model,
        output_path=out_name,
        tokenizer_path=args.tokenizer,
        engine_path=args.engine,
    )


if __name__ == "__main__":
    main()
