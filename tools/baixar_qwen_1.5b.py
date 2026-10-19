#!/usr/bin/env python3
"""tools/baixar_qwen_1.5b.py
Downloads the official Qwen2.5-1.5B-Instruct model in safetensors format directly from Hugging Face.
Supports resumable downloads (HTTP Range), progress bar, and speed measurement in MB/s.
"""

import os
import sys
import time
import urllib.request

URL_MODEL_1_5B = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"
DEST_MODEL = os.path.join("data", "qwen2.5_1.5b_instruct.safetensors")


def download_with_resume(url: str, dest_path: str, desc: str = "Model"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # 1. Get total size via HEAD request
    print(f"[*] Checking metadata on Hugging Face: {url}...")
    head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "NativaAI/1.0"})
    with urllib.request.urlopen(head_req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    total_mb = total_size / (1024 * 1024)
    print(f"[*] Total size of {desc}: {total_mb:.2f} MB ({total_size:,} bytes)")

    # 2. Check if already fully or partially downloaded
    downloaded = 0
    if os.path.exists(dest_path):
        downloaded = os.path.getsize(dest_path)
        if downloaded == total_size:
            print(f"[OK] {desc} is already completely downloaded at {dest_path}!")
            return True
        elif downloaded < total_size:
            print(f"[*] Resuming download from {downloaded / (1024*1024):.2f} MB...")
        else:
            # Larger/corrupted file, restart from scratch
            downloaded = 0

    mode = "ab" if downloaded > 0 else "wb"
    headers = {"User-Agent": "NativaAI/1.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    chunk_size = 4 * 1024 * 1024  # 4 MB per chunk

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

            # Update log every 5% or every 3 seconds
            if (pct != last_pct and pct % 5 == 0) or (now - t_last >= 3.0):
                elapsed_step = max(now - t_last, 0.001)
                speed_mb = (bytes_since_last / (1024 * 1024)) / elapsed_step
                eta_sec = (total_size - downloaded) / max(bytes_since_last / elapsed_step, 1.0) if total_size > downloaded else 0
                print(f"    [{pct:3d}%] {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB | {speed_mb:.2f} MB/s | ETA: {eta_sec:.0f}s")
                last_pct = pct
                t_last = now
                bytes_since_last = 0

    final_size = os.path.getsize(dest_path)
    total_time = time.time() - t_start
    avg_speed = (final_size / (1024 * 1024)) / max(total_time, 0.001)

    print(f"\n[OK] Download completed successfully!")
    print(f"• Saved file: {dest_path}")
    print(f"• Final size: {final_size / (1024*1024):.2f} MB")
    print(f"• Total time: {total_time:.1f}s (Average: {avg_speed:.2f} MB/s)")
    return True


if __name__ == "__main__":
    download_with_resume(URL_MODEL_1_5B, DEST_MODEL, "Qwen2.5-1.5B-Instruct Safetensors")
