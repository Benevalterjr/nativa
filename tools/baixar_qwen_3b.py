#!/usr/bin/env python3
"""tools/baixar_qwen_3b.py
Downloads the official Qwen2.5-3B-Instruct shards from Hugging Face with resume support (HTTP Range).
"""

import os
import sys
import time
import urllib.request

REPO_BASE = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main"

FILES = [
    ("model.safetensors.index.json", os.path.join("data", "qwen3b", "model.safetensors.index.json")),
    ("config.json", os.path.join("data", "qwen3b", "config.json")),
    ("model-00001-of-00002.safetensors", os.path.join("data", "qwen3b", "model-00001-of-00002.safetensors")),
    ("model-00002-of-00002.safetensors", os.path.join("data", "qwen3b", "model-00002-of-00002.safetensors")),
]


def download_file(url: str, dest_path: str, desc: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "NativaAI/1.0"})
    with urllib.request.urlopen(head_req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    total_mb = total_size / (1024 * 1024)
    print(f"\n[*] File: {desc} ({total_mb:.1f} MB)")

    downloaded = 0
    if os.path.exists(dest_path):
        downloaded = os.path.getsize(dest_path)
        if downloaded == total_size:
            print(f"[OK] {desc} is already complete on disk!")
            return True
        elif downloaded < total_size:
            print(f"[*] Resuming from {downloaded / (1024*1024):.1f} MB...")
        else:
            downloaded = 0

    mode = "ab" if downloaded > 0 else "wb"
    headers = {"User-Agent": "NativaAI/1.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    chunk_size = 4 * 1024 * 1024

    t_start = time.time()
    t_last = t_start
    bytes_since_last = 0
    last_pct = -1

    with urllib.request.urlopen(req) as resp, open(dest_path, mode) as out_f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            bytes_since_last += len(chunk)

            now = time.time()
            pct = int(downloaded * 100 / total_size) if total_size > 0 else 0

            if (pct != last_pct and pct % 5 == 0) or (now - t_last >= 4.0):
                elapsed_step = max(now - t_last, 0.001)
                speed_mb = (bytes_since_last / (1024 * 1024)) / elapsed_step
                eta_sec = (total_size - downloaded) / max(bytes_since_last / elapsed_step, 1.0) if total_size > downloaded else 0
                print(f"    [{pct:3d}%] {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB | {speed_mb:.2f} MB/s | ETA: {eta_sec:.0f}s")
                last_pct = pct
                t_last = now
                bytes_since_last = 0

    print(f"[OK] {desc} completed!")
    return True


def main():
    print("=" * 70)
    print("  DOWNLOADING QWEN 2.5 (3B INSTRUCT) SAFETENSORS")
    print("=" * 70)
    for filename, local_path in FILES:
        url = f"{REPO_BASE}/{filename}"
        download_file(url, local_path, filename)
    print("\n" + "=" * 70)
    print("  ALL QWEN 2.5 3B FILES DOWNLOADED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
