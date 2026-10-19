#!/usr/bin/env python3
"""
gerar_safetensors.py
Generates a large-scale synthetic .safetensors file simulating layers of a modern LLM.
"""
import json
import struct
import numpy as np

def criar_safetensors_sintetico(caminho: str):
    print(f"[*] Generating synthetic tensors in SafeTensors format...")
    
    # 1. MLP Gate projection matrix (1024 x 2048 = 2,097,152 floats ~ 8.4 MB)
    # Initialized with a normal distribution N(0, 0.02) typical of LLMs
    rng = np.random.RandomState(42)
    w_gate = rng.normal(0.0, 0.02, size=(1024, 2048)).astype(np.float32)
    
    # 2. Self-Attention Q projection matrix (1024 x 1024 = 1,048,576 floats ~ 4.2 MB)
    w_q = rng.normal(0.0, 0.02, size=(1024, 1024)).astype(np.float32)

    # 3. Up projection matrix (512 x 2048 = 1,048,576 floats ~ 4.2 MB)
    w_up = rng.normal(0.0, 0.02, size=(512, 2048)).astype(np.float32)

    tensors = {
        "model.layers.0.mlp.gate_proj.weight": w_gate,
        "model.layers.0.mlp.up_proj.weight": w_up,
        "model.layers.0.self_attn.q_proj.weight": w_q,
    }

    header = {}
    current_offset = 0
    raw_buffers = []

    for name, arr in tensors.items():
        data_bytes = arr.tobytes()
        raw_buffers.append(data_bytes)
        length = len(data_bytes)
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + length]
        }
        current_offset += length

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    with open(caminho, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for buf in raw_buffers:
            f.write(buf)

    total_mb = (8 + header_len + current_offset) / (1024 * 1024)
    print(f"[OK] File '{caminho}' generated successfully!")
    print(f"     Total size: {total_mb:.2f} MB")
    print(f"     Included tensors: {list(tensors.keys())}")

if __name__ == "__main__":
    criar_safetensors_sintetico("grande_modelo.safetensors")
