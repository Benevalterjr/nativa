#!/usr/bin/env python3
"""
tools/baixar_qwen_camada.py
Downloads exclusively the real Layer 0 of the official Alibaba Cloud Qwen2.5-0.5B model
directly from Hugging Face via HTTP Range request (only ~28 MB instead of 1 GB).
Generates a 100% canonical SafeTensors file at data/qwen2.5_camada0.safetensors.
"""

import json
import os
import struct
import sys
import urllib.request

QWEN_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors"
OUTPUT_FILE = os.path.join("data", "qwen2.5_camada0.safetensors")

def main():
    print(f"[*] Connecting to Hugging Face to inspect Qwen2.5-0.5B...")
    # 1. Fetch the first 50KB to extract the header
    req_header = urllib.request.Request(QWEN_URL, headers={"Range": "bytes=0-50000"})
    with urllib.request.urlopen(req_header) as resp:
        head_data = resp.read()

    header_len = struct.unpack("<Q", head_data[:8])[0]
    full_header = json.loads(head_data[8:8 + header_len])
    data_start_offset = 8 + header_len

    # 2. Filter layer 0 tensors
    layer0_tensors = {k: v for k, v in full_header.items() if "layers.0." in k}
    print(f"[*] Found {len(layer0_tensors)} tensors in Layer 0 of Qwen2.5-0.5B:")
    for k in sorted(layer0_tensors.keys()):
        print(f"    - {k}: shape {layer0_tensors[k]['shape']} ({layer0_tensors[k]['dtype']})")

    min_offset = min(v["data_offsets"][0] for v in layer0_tensors.values())
    max_offset = max(v["data_offsets"][1] for v in layer0_tensors.values())
    total_bytes = max_offset - min_offset
    print(f"[*] Exact size of Layer 0 in BF16: {total_bytes / (1024 * 1024):.2f} MB")

    # 3. Download the exact binary slice of Layer 0
    start_byte = data_start_offset + min_offset
    end_byte = data_start_offset + max_offset - 1
    print(f"[*] Downloading bytes [{start_byte} .. {end_byte}] via HTTP Range...")

    req_data = urllib.request.Request(QWEN_URL, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    with urllib.request.urlopen(req_data) as resp:
        layer_bytes = resp.read()

    print(f"[+] Download completed: {len(layer_bytes)} bytes received!")

    # 4. Build new SafeTensors header remapping local offsets
    new_header = {}
    for k, v in layer0_tensors.items():
        orig_s, orig_e = v["data_offsets"]
        new_header[k] = {
            "dtype": v["dtype"],
            "shape": v["shape"],
            "data_offsets": [orig_s - min_offset, orig_e - min_offset]
        }

    new_header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    new_header_len = len(new_header_json)

    # 5. Save SafeTensors file to data/
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        f.write(struct.pack("<Q", new_header_len))
        f.write(new_header_json)
        f.write(layer_bytes)

    print(f"[OK] Qwen Layer 0 saved successfully to: {OUTPUT_FILE}")
    print(f"[OK] Final size on disk: {os.path.getsize(OUTPUT_FILE) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
