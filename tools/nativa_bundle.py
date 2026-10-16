#!/usr/bin/env python3
"""
Nativa Bundle Packager - Gera executáveis únicos e autocontidos (.exe / ELF)
Inspirado no conceito do Mozilla llamafile (Cosmopolitan APE).
Empacota: Motor C99 + Vocabulário + Pesos .natq4 + Trailer de 64B alinhado a 64KB.
"""

import argparse
import os
import pathlib
import stat
import struct
import sys
from typing import Optional

# Constantes de alinhamento e formato
NATIVA_BUNDLE_MAGIC = 0x4E4154495641464C  # "NATIVAF\0"
NATIVA_BUNDLE_ALIGN = 65536               # 64 KB alignment (MapViewOfFile / mmap page)
NATIVA_BUNDLE_VERSION = 1
TRAILER_FORMAT = "<QQQQQQII8s"
TRAILER_SIZE = struct.calcsize(TRAILER_FORMAT)  # exatamente 64 bytes

# Adiciona a raiz do projeto ao sys.path para importar nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import nativa


def export_vocab_bytes(tokenizer) -> bytes:
    """Exporta o vocabulário para bytes no formato binário lido pelo motor C."""
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
            raise ValueError(f"Token {token_id} excede 65535 bytes.")
        chunks.append(struct.pack("<H", len(token)))
        chunks.append(token)
    return b"".join(chunks)


def write_vocab_file(tokenizer, path: str) -> None:
    """Exporta o vocabulário para o arquivo binário lido pelo motor C."""
    data = export_vocab_bytes(tokenizer)
    with open(path, "wb") as f:
        f.write(data)


def align_up(offset: int, alignment: int = NATIVA_BUNDLE_ALIGN) -> int:
    """Calcula o próximo offset alinhado ao múltiplo de alignment."""
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def inspect_bundle(bundle_path: str) -> Optional[dict]:
    """Inspeciona um executável bundle existente e retorna os metadados do trailer se for válido."""
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
    """Cria um executável único autocontido anexando o vocabulário e os pesos do modelo."""
    safe_model = pathlib.Path(model_path).resolve()
    if not safe_model.is_file():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {safe_model}")

    # Resolve o executável do motor C
    if not engine_path:
        is_win = sys.platform.startswith("win")
        default_exe = "nativa_engine.exe" if is_win else "nativa_engine"
        engine_path = os.path.join(ROOT_DIR, "tools", default_exe)

    safe_engine = pathlib.Path(engine_path).resolve()
    if not safe_engine.is_file():
        raise FileNotFoundError(f"Executável do motor C não encontrado: {safe_engine}")

    # Resolve o tokenizer
    if not tokenizer_path:
        candidate_same_dir = pathlib.Path(f"{safe_model}.tokenizer.json").resolve()
        candidate_qwen = pathlib.Path(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json").resolve()
        if candidate_same_dir.is_file():
            safe_tok = candidate_same_dir
        elif candidate_qwen.is_file():
            safe_tok = candidate_qwen
        else:
            raise FileNotFoundError(f"Tokenizer não localizado para o modelo: {safe_model}")
    else:
        safe_tok = pathlib.Path(tokenizer_path).resolve()
        if not safe_tok.is_file():
            raise FileNotFoundError(f"Tokenizer não localizado para o modelo: {safe_tok}")

    print(f"[*] Gerando vocabulário binário a partir de: {safe_tok}")
    tok = nativa.Tokenizer(str(safe_tok))
    vocab_bytes = export_vocab_bytes(tok)

    with safe_engine.open("rb") as ef:
        engine_bytes = ef.read()

    with safe_model.open("rb") as mf:
        model_bytes = mf.read()

    engine_size = len(engine_bytes)
    vocab_size = len(vocab_bytes)
    model_size = len(model_bytes)

    # Calcula offsets com alinhamento a 64 KB
    vocab_offset = align_up(engine_size, NATIVA_BUNDLE_ALIGN)
    pad1_len = vocab_offset - engine_size

    model_offset = align_up(vocab_offset + vocab_size, NATIVA_BUNDLE_ALIGN)
    pad2_len = model_offset - (vocab_offset + vocab_size)

    # Constrói o trailer fixo de 64 bytes
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

    print("[*] Montando bundle autocontido:")
    print(f"    -> Motor C99     : {engine_size:,} bytes")
    print(f"    -> Vocabulário   : {vocab_size:,} bytes (offset: {vocab_offset:,}, alinhado a 64KB)")
    print(f"    -> Modelo .natq4 : {model_size:,} bytes (offset: {model_offset:,}, alinhado a 64KB)")
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

    # Dá permissão de execução segura em ambientes POSIX (SonarCloud S2612: 0o750)
    try:
        safe_output.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    except OSError:
        pass

    total_mb = safe_output.stat().st_size / (1024 * 1024)
    print(f"[OK] Bundle criado com sucesso: {safe_output} ({total_mb:.2f} MB)")
    return str(safe_output)


def main():
    parser = argparse.ArgumentParser(
        description="Nativa Bundle Packager - Gera executáveis únicos autocontidos estilo llamafile"
    )
    parser.add_argument("model", help="Caminho do modelo .natq4 a empacotar")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Caminho do executável de saída gerado (padrão: <modelo>.exe)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default=None,
        help="Caminho do tokenizer.json correspondente",
    )
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        help="Caminho do executável nativa_engine base a usar",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspeciona e exibe metadados de um bundle existente",
    )

    args = parser.parse_args()

    if args.inspect:
        info = inspect_bundle(args.model)
        if not info:
            print(f"[ERRO] O arquivo {args.model} não é um bundle Nativa válido.", file=sys.stderr)
            sys.exit(1)
        print("=== METADADOS DO BUNDLE NATIVA ===")
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
