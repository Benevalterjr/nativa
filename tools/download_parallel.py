#!/usr/bin/env python3
import concurrent.futures
import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
DEST = os.path.abspath("data/qwen3_0.6b_q4_k_m.gguf")
NUM_WORKERS = 8

def get_content_length(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.headers.get("Content-Length", 0))

def download_chunk(url, dest_path, start, end, chunk_id, progress_dict):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end}"
    })
    downloaded = 0
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(dest_path, "r+b") as f:
            f.seek(start)
            while True:
                buf = resp.read(256 * 1024)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                progress_dict[chunk_id] = downloaded
    return downloaded

def main():
    print(f"[*] Checking remote size of {URL}...")
    total_size = get_content_length(URL)
    print(f"[*] Total size: {total_size / (1024*1024):.2f} MB ({total_size} bytes)")

    # Prepare destination file with total size
    if not os.path.exists(DEST) or os.path.getsize(DEST) != total_size:
        with open(DEST, "wb") as f:
            f.truncate(total_size)

    chunk_size = (total_size + NUM_WORKERS - 1) // NUM_WORKERS
    chunks = []
    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end = min((i + 1) * chunk_size - 1, total_size - 1)
        if start <= end:
            chunks.append((i, start, end))

    print(f"[*] Starting accelerated download with {len(chunks)} concurrent connections...")
    progress_dict = {i: 0 for i, _, _ in chunks}
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(download_chunk, URL, DEST, start, end, chunk_id, progress_dict)
            for chunk_id, start, end in chunks
        ]

        while any(not f.done() for f in futures):
            total_dl = sum(progress_dict.values())
            pct = (total_dl * 100.0 / total_size) if total_size > 0 else 0
            elap = time.time() - t0
            speed = (total_dl / (1024 * 1024)) / max(elap, 0.001)
            print(f"\r  -> Progress: [{pct:5.1f}%] {total_dl/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({speed:4.2f} MB/s)", end="", flush=True)
            time.sleep(0.5)

        for f in futures:
            f.result()

    total_elap = time.time() - t0
    final_speed = (total_size / (1024 * 1024)) / max(total_elap, 0.001)
    print(f"\n[OK] Download completed successfully in {total_elap:.1f}s ({final_speed:.2f} MB/s): {DEST}")

if __name__ == "__main__":
    main()
