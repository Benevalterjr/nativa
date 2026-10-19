#!/usr/bin/env python3
"""
gerar_llama7b_block.py
Generates a Transformer block with the REAL dimensions of a LLaMA-7B / Mistral-7B:
- d_model = 4096
- d_intermediate = 11008
Total of ~202 million parameters (~809 MB in FP32).
Generated via direct streaming to disk, keeping RAM usage below 100 MB.
"""
import json
import struct
import time
import numpy as np

def gerar_bloco_llama7b(caminho: str):
    print("==================================================================")
    print("  LLaMA-7B / Mistral-7B BLOCK GENERATOR (SafeTensors ~809 MB)     ")
    print("==================================================================")

    d_model = 4096
    d_ff = 11008

    camadas_spec = [
        ("model.layers.0.mlp.gate_proj.weight", (d_ff, d_model)),
        ("model.layers.0.mlp.up_proj.weight",   (d_ff, d_model)),
        ("model.layers.0.mlp.down_proj.weight", (d_model, d_ff)),
        ("model.layers.0.self_attn.q_proj.weight", (d_model, d_model)),
        ("model.layers.0.self_attn.k_proj.weight", (d_model, d_model)),
        ("model.layers.0.self_attn.v_proj.weight", (d_model, d_model)),
        ("model.layers.0.self_attn.o_proj.weight", (d_model, d_model)),
    ]

    header = {}
    current_offset = 0
    total_params = 0

    for name, shape in camadas_spec:
        n_elems = shape[0] * shape[1]
        byte_len = n_elems * 4  # float32 = 4 bytes
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [current_offset, current_offset + byte_len]
        }
        current_offset += byte_len
        total_params += n_elems

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    print(f"[*] Layers to generate: {len(camadas_spec)}")
    print(f"[*] Total parameters: {total_params:,} (~{total_params / 1e6:.1f} Million)")
    print(f"[*] Binary payload size: {current_offset / (1024*1024):.2f} MB")
    print(f"[*] Writing '{caminho}' via continuous streaming...")

    t0 = time.perf_counter()
    rng = np.random.RandomState(42)

    with open(caminho, "wb") as f:
        # 1. Write header length and JSON header
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)

        # 2. Generate and write each matrix in chunks (chunk streaming)
        for i, (name, shape) in enumerate(camadas_spec, start=1):
            rows, cols = shape
            mb_camada = (rows * cols * 4) / (1024 * 1024)
            print(f"    [{i}/{len(camadas_spec)}] {name} ({rows}x{cols} ~ {mb_camada:.1f} MB)...", end="", flush=True)
            
            # Generate in chunks of 1024 rows to avoid inflating memory
            chunk_rows = 1024
            for r in range(0, rows, chunk_rows):
                cur_r = min(chunk_rows, rows - r)
                # Normal distribution N(0, 0.02)
                chunk_data = rng.normal(0.0, 0.02, size=(cur_r, cols)).astype(np.float32)
                f.write(chunk_data.tobytes())
            print(" Done!")

    tempo = time.perf_counter() - t0
    total_mb = (8 + header_len + current_offset) / (1024 * 1024)
    print(f"\n[OK] File '{caminho}' successfully generated in {tempo:.2f}s!")
    print(f"     Total disk size: {total_mb:.2f} MB")

if __name__ == "__main__":
    gerar_bloco_llama7b("llama7b_bloco.safetensors")
