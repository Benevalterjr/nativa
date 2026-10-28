#!/usr/bin/env python3
"""
Downloads Qwen3-1.7B safetensors and metadata from Hugging Face with parallel chunk downloading.
Repository: https://huggingface.co/Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Tuple

BASE_URL = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main"
DEFAULT_DEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "qwen3_1.7b"))

METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
]

LARGE_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

NUM_WORKERS = 8


def get_remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download_small_file(url: str, dest_path: str, filename: str) -> None:
    if os.path.exists(dest_path):
        remote_sz = get_remote_size(url)
        if remote_sz > 0 and os.path.getsize(dest_path) == remote_sz:
            print(f"  [OK] {filename} already exists ({remote_sz:,} bytes)")
            return

    print(f"  [*] Downloading {filename}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print(f"  [OK] {filename} saved ({os.path.getsize(dest_path):,} bytes)")


def download_chunk_with_retry(url: str, dest_path: str, start: int, end: int, chunk_id: int, progress_dict: Dict[int, int]) -> int:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={start}-{end}"
            })
            downloaded = 0
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(dest_path, "r+b") as f:
                    f.seek(start)
                    while True:
                        buf = resp.read(512 * 1024)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        progress_dict[chunk_id] = downloaded
            return downloaded
        except Exception as err:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to download chunk {chunk_id} [{start}-{end}] after {max_retries} attempts: {err}") from err
    return 0


def download_large_file_parallel(url: str, dest_path: str, filename: str) -> None:
    remote_sz = get_remote_size(url)
    if remote_sz <= 0:
        raise ValueError(f"Could not determine file size for {url}")

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == remote_sz:
        print(f"  [OK] {filename} already fully downloaded ({remote_sz / (1024*1024):.2f} MB)")
        return

    print(f"\n[*] Preparing {filename} ({remote_sz / (1024*1024):.2f} MB)...")
    with open(dest_path, "wb") as f:
        f.truncate(remote_sz)

    chunk_size = (remote_sz + NUM_WORKERS - 1) // NUM_WORKERS
    chunks: List[Tuple[int, int, int]] = []
    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end = min((i + 1) * chunk_size - 1, remote_sz - 1)
        if start <= end:
            chunks.append((i, start, end))

    print(f"[*] Downloading {filename} with {len(chunks)} parallel connections...")
    progress_dict: Dict[int, int] = {i: 0 for i, _, _ in chunks}
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(download_chunk_with_retry, url, dest_path, start, end, chunk_id, progress_dict)
            for chunk_id, start, end in chunks
        ]

        while any(not f.done() for f in futures):
            total_dl = sum(progress_dict.values())
            pct = (total_dl * 100.0 / remote_sz) if remote_sz > 0 else 0
            elap = time.time() - t0
            speed = (total_dl / (1024 * 1024)) / max(elap, 0.001)
            print(f"\r  -> [{pct:5.1f}%] {total_dl / (1024*1024):.1f}/{remote_sz / (1024*1024):.1f} MB ({speed:5.2f} MB/s)", end="", flush=True)
            time.sleep(0.5)

        for f in futures:
            f.result()

    total_elap = time.time() - t0
    final_speed = (remote_sz / (1024 * 1024)) / max(total_elap, 0.001)
    print(f"\r  -> [100.0%] {remote_sz / (1024*1024):.1f}/{remote_sz / (1024*1024):.1f} MB ({final_speed:5.2f} MB/s) in {total_elap:.1f}s")
    print(f"  [OK] {filename} completed successfully!")


def main() -> None:
    dest_dir = DEFAULT_DEST
    if len(sys.argv) > 1:
        dest_dir = os.path.abspath(sys.argv[1])

    os.makedirs(dest_dir, exist_ok=True)
    print(f"=== Downloading Qwen3-1.7B to {dest_dir} ===")

    print("\n--- 1. Downloading Metadata & Tokenizer ---")
    for f in METADATA_FILES:
        url = f"{BASE_URL}/{f}"
        dest = os.path.join(dest_dir, f)
        download_small_file(url, dest, f)

    print("\n--- 2. Downloading SafeTensors Shards ---")
    for f in LARGE_FILES:
        url = f"{BASE_URL}/{f}"
        dest = os.path.join(dest_dir, f)
        download_large_file_parallel(url, dest, f)

    print("\n=== All Qwen3-1.7B files downloaded successfully! ===")


if __name__ == "__main__":
    main()
