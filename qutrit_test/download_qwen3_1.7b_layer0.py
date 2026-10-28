#!/usr/bin/env python3
"""
qutrit_test/download_qwen3_1.7b_layer0.py
Downloads Layer 0 of Qwen3-1.7B-Base directly from Hugging Face via HTTP Range request.
Generates a valid, self-contained SafeTensors file at data/qwen3_1.7b_layer0.safetensors (~96 MB).
"""

import os
import sys
import time
import json
import struct
import urllib.request

QWEN3_URL = "https://huggingface.co/Qwen/Qwen3-1.7B-Base/resolve/main/model.safetensors"
DEST_FILE = os.path.join("data", "qwen3_1.7b_layer0.safetensors")

def download_layer0():
    os.makedirs("data", exist_ok=True)
    print("=" * 90)
    print("  DOWNLOAD DA CAMADA 0 DO QWEN3-1.7B-BASE (VIA HTTP RANGE)")
    print("=" * 90)
    print(f"[*] Conectando a {QWEN3_URL}...")

    # 1. Fetch header length
    req_len = urllib.request.Request(QWEN3_URL, headers={"Range": "bytes=0-7", "User-Agent": "NativaAI/1.0"})
    with urllib.request.urlopen(req_len) as resp:
        head_bytes = resp.read()
    hlen = struct.unpack("<Q", head_bytes)[0]
    print(f"[*] Tamanho do cabeçalho SafeTensors: {hlen:,} bytes")

    # 2. Fetch full JSON header
    req_hdr = urllib.request.Request(QWEN3_URL, headers={"Range": f"bytes=8-{8 + hlen - 1}", "User-Agent": "NativaAI/1.0"})
    with urllib.request.urlopen(req_hdr) as resp:
        hdr_json_bytes = resp.read()
    full_header = json.loads(hdr_json_bytes.decode("utf-8"))

    # 3. Filter Layer 0 tensors
    l0_tensors = {k: v for k, v in full_header.items() if "layers.0." in k}
    print(f"[*] Encontrados {len(l0_tensors)} tensores na Camada 0:")
    for k in sorted(l0_tensors.keys()):
        print(f"    - {k:<45}: {str(l0_tensors[k]['shape']):<16} ({l0_tensors[k]['dtype']})")

    min_offset = min(v["data_offsets"][0] for v in l0_tensors.values())
    max_offset = max(v["data_offsets"][1] for v in l0_tensors.values())
    total_bytes = max_offset - min_offset
    total_mb = total_bytes / (1024 * 1024)
    print(f"\n[*] Offset de dados relativo: [{min_offset:,} .. {max_offset:,}]")
    print(f"[*] Tamanho exato da fatia binária: {total_mb:.2f} MB ({total_bytes:,} bytes)")

    data_start_offset = 8 + hlen
    abs_start = data_start_offset + min_offset
    abs_end = data_start_offset + max_offset - 1
    print(f"[*] Bytes absolutos no arquivo remoto: [{abs_start:,} .. {abs_end:,}]")

    # 4. Check if raw slice already downloaded
    raw_slice_path = DEST_FILE + ".raw"
    downloaded = 0
    if os.path.exists(raw_slice_path):
        downloaded = os.path.getsize(raw_slice_path)
        if downloaded == total_bytes:
            print(f"[OK] Dados brutos já baixados completamente ({downloaded:,} bytes).")
        elif downloaded < total_bytes:
            print(f"[*] Retomando download dos dados brutos a partir de {downloaded / (1024*1024):.2f} MB...")
        else:
            downloaded = 0

    if downloaded < total_bytes:
        mode = "ab" if downloaded > 0 else "wb"
        chunk_start = abs_start + downloaded
        req_slice = urllib.request.Request(
            QWEN3_URL,
            headers={
                "Range": f"bytes={chunk_start}-{abs_end}",
                "User-Agent": "NativaAI/1.0"
            }
        )
        chunk_size = 4 * 1024 * 1024
        t0 = time.time()
        t_last = t0
        bytes_step = 0
        last_pct = -1

        print(f"[*] Baixando {total_mb:.2f} MB em blocos de 4 MB...")
        with urllib.request.urlopen(req_slice) as resp, open(raw_slice_path, mode) as out_f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                bytes_step += len(chunk)

                now = time.time()
                pct = int(downloaded * 100 / total_bytes)
                if (pct != last_pct and pct % 10 == 0) or (now - t_last >= 5.0):
                    speed = (bytes_step / (1024 * 1024)) / max(now - t_last, 0.001)
                    eta = (total_bytes - downloaded) / max(bytes_step / max(now - t_last, 0.001), 1.0) if total_bytes > downloaded else 0
                    print(f"    [{pct:3d}%] {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB | {speed:.2f} MB/s | ETA: {eta:.0f}s")
                    last_pct = pct
                    t_last = now
                    bytes_step = 0

        dt_tot = time.time() - t0
        print(f"[+] Download dos dados brutos concluído em {dt_tot:.1f}s ({total_mb / max(dt_tot, 0.001):.2f} MB/s)!")

    # 5. Build New SafeTensors file
    print(f"\n[*] Montando arquivo SafeTensors canônico: '{DEST_FILE}'...")
    new_header = {}
    for k, v in l0_tensors.items():
        s, e = v["data_offsets"]
        new_header[k] = {
            "dtype": v["dtype"],
            "shape": v["shape"],
            "data_offsets": [s - min_offset, e - min_offset]
        }

    # Add metadata if present
    if "__metadata__" in full_header:
        new_header["__metadata__"] = full_header["__metadata__"]

    new_header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Align to 8 bytes
    pad_len = (8 - (len(new_header_json) % 8)) % 8
    new_header_json += b" " * pad_len
    new_header_len = len(new_header_json)

    with open(raw_slice_path, "rb") as raw_f, open(DEST_FILE, "wb") as out_st:
        out_st.write(struct.pack("<Q", new_header_len))
        out_st.write(new_header_json)
        # Stream raw slice
        while True:
            b = raw_f.read(8 * 1024 * 1024)
            if not b:
                break
            out_st.write(b)

    final_size = os.path.getsize(DEST_FILE)
    print(f"[OK] SafeTensors da Camada 0 gerado com sucesso!")
    print(f"    - Arquivo: {DEST_FILE}")
    print(f"    - Tamanho: {final_size / (1024*1024):.2f} MB ({final_size:,} bytes)")

    # 6. Sanity check: load with nativa
    import nativa
    test_t = nativa.load_safetensors_fn(DEST_FILE, "model.layers.0.mlp.gate_proj.weight")
    print(f"[*] Verificação de integridade: gate_proj shape={test_t.shape}, min={float(test_t.data.min()):.4f}, max={float(test_t.data.max()):.4f}")
    print("=" * 90)
    return True

if __name__ == "__main__":
    download_layer0()
