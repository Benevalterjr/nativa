#!/usr/bin/env python3
"""
tools/baixar_qwen.py
Permite baixar o modelo oficial Qwen2.5-0.5B da Alibaba Cloud (pesos + tokenizador)
diretamente do Hugging Face com exibição de progresso e velocidade em tempo real.
Uso:
    python tools/baixar_qwen.py --full          # baixa o modelo completo (942 MB com 24 camadas + tokenizer)
    python tools/baixar_qwen.py --layers 2      # baixa apenas 2 camadas (~56 MB)
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.request

QWEN_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors"
TOKENIZER_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json"

def download_file_with_progress(url: str, dest: str, desc: str = "Arquivo"):
    if os.path.exists(dest):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
            if os.path.getsize(dest) == total_size and total_size > 0:
                print(f"[OK] {desc} já baixado: {dest} ({total_size / (1024*1024):.2f} MB)")
                return
        except Exception:
            pass

    print(f"[*] Baixando {desc} para {dest}...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 4 * 1024 * 1024
        last_percent = -1
        start_time = time.time()
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int(downloaded * 100 / total_size)
                if pct != last_percent and (pct % 10 == 0 or pct == 100):
                    elapsed = time.time() - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                    print(f"    [{pct:3d}%] {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)")
                    last_percent = pct
    print(f"[OK] {desc} concluido com sucesso: {dest} ({os.path.getsize(dest) / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Baixar camadas ou modelo completo do Qwen2.5-0.5B")
    parser.add_argument("--layers", type=int, default=0, help="Numero de camadas para baixar (ex: 2, 4)")
    parser.add_argument("--full", action="store_true", help="Baixar o arquivo model.safetensors completo (24 camadas)")
    args = parser.parse_args()

    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Baixar Tokenizer oficial do Qwen
    tok_dest = os.path.abspath(os.path.join(data_dir, "tokenizer_qwen.json"))
    download_file_with_progress(TOKENIZER_URL, tok_dest, "Tokenizador oficial (tokenizer.json)")

    # 2. Baixar Pesos
    if args.full or args.layers == 0:
        model_dest = os.path.abspath(os.path.join(data_dir, "qwen2.5_0.5b.safetensors"))
        download_file_with_progress(QWEN_URL, model_dest, "Pesos do Qwen2.5-0.5B Completo (24 camadas)")
        return

    if not (1 <= args.layers <= 24):
        parser.error("O número de camadas deve estar entre 1 e 24.")

    n_layers = args.layers
    dest = os.path.abspath(os.path.join(data_dir, f"qwen2.5_{n_layers}camadas.safetensors"))
    if os.path.commonpath([data_dir, dest]) != data_dir:
        parser.error("Caminho de destino fora do diretório permitido.")

    print(f"[*] Conectando ao Hugging Face para baixar {n_layers} camadas do Qwen2.5-0.5B...")

    req_header = urllib.request.Request(QWEN_URL, headers={"Range": "bytes=0-50000"})
    with urllib.request.urlopen(req_header) as resp:
        head_data = resp.read()

    header_len = struct.unpack("<Q", head_data[:8])[0]
    full_header = json.loads(head_data[8:8 + header_len])
    data_start = 8 + header_len

    selected_prefixes = [f"model.layers.{i}." for i in range(n_layers)]
    selected_tensors = {k: v for k, v in full_header.items() if any(k.startswith(p) for p in selected_prefixes)}

    print(f"[*] Selecionados {len(selected_tensors)} tensores ({n_layers} camadas):")
    for i in range(n_layers):
        cnt = sum(1 for k in selected_tensors if f"layers.{i}." in k)
        print(f"    - Camada {i}: {cnt} tensores")

    min_offset = min(v["data_offsets"][0] for v in selected_tensors.values())
    max_offset = max(v["data_offsets"][1] for v in selected_tensors.values())
    total_bytes = max_offset - min_offset
    print(f"[*] Tamanho total do trecho binário: {total_bytes / (1024 * 1024):.2f} MB")

    start_byte = data_start + min_offset
    end_byte = data_start + max_offset - 1
    print(f"[*] Baixando bytes [{start_byte} .. {end_byte}]...")

    req_data = urllib.request.Request(QWEN_URL, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    with urllib.request.urlopen(req_data) as resp:
        layer_bytes = resp.read()

    print(f"[+] Download concluido: {len(layer_bytes)} bytes recebidos!")

    new_header = {}
    for k, v in selected_tensors.items():
        orig_s, orig_e = v["data_offsets"]
        new_header[k] = {
            "dtype": v["dtype"],
            "shape": v["shape"],
            "data_offsets": [orig_s - min_offset, orig_e - min_offset]
        }

    new_header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    new_header_len = len(new_header_json)

    with open(dest, "wb") as f:
        f.write(struct.pack("<Q", new_header_len))
        f.write(new_header_json)
        f.write(layer_bytes)

    print(f"[OK] {n_layers} Camadas salvas com sucesso em: {dest}")
    print(f"[OK] Tamanho final no disco: {os.path.getsize(dest) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
