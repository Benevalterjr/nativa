#!/usr/bin/env python3
"""
tools/baixar_qwen_camada.py
Baixa exclusivamente a Camada 0 real do modelo oficial Qwen2.5-0.5B da Alibaba Cloud
diretamente do Hugging Face via HTTP Range request (apenas ~28 MB em vez de 1 GB).
Gera um arquivo SafeTensors 100% canônico em data/qwen2.5_camada0.safetensors.
"""

import json
import os
import struct
import sys
import urllib.request

QWEN_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors"
OUTPUT_FILE = os.path.join("data", "qwen2.5_camada0.safetensors")

def main():
    print(f"[*] Conectando ao Hugging Face para inspecionar Qwen2.5-0.5B...")
    # 1. Obter os primeiros 50KB para extrair o cabeçalho
    req_header = urllib.request.Request(QWEN_URL, headers={"Range": "bytes=0-50000"})
    with urllib.request.urlopen(req_header) as resp:
        head_data = resp.read()

    header_len = struct.unpack("<Q", head_data[:8])[0]
    full_header = json.loads(head_data[8:8 + header_len])
    data_start_offset = 8 + header_len

    # 2. Filtrar tensores da camada 0
    layer0_tensors = {k: v for k, v in full_header.items() if "layers.0." in k}
    print(f"[*] Encontrados {len(layer0_tensors)} tensores na Camada 0 do Qwen2.5-0.5B:")
    for k in sorted(layer0_tensors.keys()):
        print(f"    - {k}: shape {layer0_tensors[k]['shape']} ({layer0_tensors[k]['dtype']})")

    min_offset = min(v["data_offsets"][0] for v in layer0_tensors.values())
    max_offset = max(v["data_offsets"][1] for v in layer0_tensors.values())
    total_bytes = max_offset - min_offset
    print(f"[*] Tamanho exato da Camada 0 em BF16: {total_bytes / (1024 * 1024):.2f} MB")

    # 3. Baixar exatamente o trecho binário da Camada 0
    start_byte = data_start_offset + min_offset
    end_byte = data_start_offset + max_offset - 1
    print(f"[*] Baixando bytes [{start_byte} .. {end_byte}] via HTTP Range...")

    req_data = urllib.request.Request(QWEN_URL, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    with urllib.request.urlopen(req_data) as resp:
        layer_bytes = resp.read()

    print(f"[+] Download concluído: {len(layer_bytes)} bytes recebidos!")

    # 4. Construir novo cabeçalho SafeTensors remapeando os offsets locais
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

    # 5. Salvar arquivo SafeTensors em data/
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        f.write(struct.pack("<Q", new_header_len))
        f.write(new_header_json)
        f.write(layer_bytes)

    print(f"[OK] Camada 0 do Qwen salva com sucesso em: {OUTPUT_FILE}")
    print(f"[OK] Tamanho final no disco: {os.path.getsize(OUTPUT_FILE) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
