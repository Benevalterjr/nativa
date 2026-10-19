#!/usr/bin/env python3
"""
tools/baixar_qwen.py
Downloads the official Alibaba Cloud Qwen2.5-0.5B model (weights + tokenizer)
directly from Hugging Face with real-time progress and speed display.
Usage:
    python tools/baixar_qwen.py --full          # download full model (942 MB with 24 layers + tokenizer)
    python tools/baixar_qwen.py --layers 2      # download only 2 layers (~56 MB)
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

def download_file_with_progress(url: str, dest: str, desc: str = "File"):
    if os.path.exists(dest):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
            if os.path.getsize(dest) == total_size and total_size > 0:
                print(f"[OK] {desc} already downloaded: {dest} ({total_size / (1024*1024):.2f} MB)")
                return
        except Exception:
            pass

    print(f"[*] Downloading {desc} to {dest}...")
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
    print(f"[OK] {desc} completed successfully: {dest} ({os.path.getsize(dest) / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download layers or the full Qwen2.5-0.5B model")
    parser.add_argument("--layers", type=int, default=0, help="Number of layers to download (e.g., 2, 4)")
    parser.add_argument("--full", action="store_true", help="Download the full model.safetensors file (24 layers)")
    args = parser.parse_args()

    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download official Qwen Tokenizer
    tok_dest = os.path.abspath(os.path.join(data_dir, "tokenizer_qwen.json"))
    download_file_with_progress(TOKENIZER_URL, tok_dest, "Official tokenizer (tokenizer.json)")

    # 2. Download Weights
    if args.full or args.layers == 0:
        model_dest = os.path.abspath(os.path.join(data_dir, "qwen2.5_0.5b.safetensors"))
        download_file_with_progress(QWEN_URL, model_dest, "Full Qwen2.5-0.5B weights (24 layers)")
        return

    if not (1 <= args.layers <= 24):
        parser.error("The number of layers must be between 1 and 24.")

    n_layers = args.layers
    dest = os.path.abspath(os.path.join(data_dir, f"qwen2.5_{n_layers}camadas.safetensors"))
    if os.path.commonpath([data_dir, dest]) != data_dir:
        parser.error("Destination path is outside the allowed directory.")

    print(f"[*] Connecting to Hugging Face to download {n_layers} layers of Qwen2.5-0.5B...")

    req_header = urllib.request.Request(QWEN_URL, headers={"Range": "bytes=0-50000"})
    with urllib.request.urlopen(req_header) as resp:
        head_data = resp.read()

    header_len = struct.unpack("<Q", head_data[:8])[0]
    full_header = json.loads(head_data[8:8 + header_len])
    data_start = 8 + header_len

    selected_prefixes = [f"model.layers.{i}." for i in range(n_layers)]
    selected_tensors = {k: v for k, v in full_header.items() if any(k.startswith(p) for p in selected_prefixes)}

    print(f"[*] Selected {len(selected_tensors)} tensors ({n_layers} layers):")
    for i in range(n_layers):
        cnt = sum(1 for k in selected_tensors if f"layers.{i}." in k)
        print(f"    - Layer {i}: {cnt} tensors")

    min_offset = min(v["data_offsets"][0] for v in selected_tensors.values())
    max_offset = max(v["data_offsets"][1] for v in selected_tensors.values())
    total_bytes = max_offset - min_offset
    print(f"[*] Total binary slice size: {total_bytes / (1024 * 1024):.2f} MB")

    start_byte = data_start + min_offset
    end_byte = data_start + max_offset - 1
    print(f"[*] Downloading bytes [{start_byte} .. {end_byte}]...")

    req_data = urllib.request.Request(QWEN_URL, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    with urllib.request.urlopen(req_data) as resp:
        layer_bytes = resp.read()

    print(f"[+] Download completed: {len(layer_bytes)} bytes received!")

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

    print(f"[OK] {n_layers} layers successfully saved to: {dest}")
    print(f"[OK] Final size on disk: {os.path.getsize(dest) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
