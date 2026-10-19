"""repack_q4x4.py - Reorganizes layer Q4 matrices into interleaved 4-row blocks.

Original format (per row):   [scale(2B) qs(16B)] x num_blocks, row after row.
Q4x4 format (per 4-row group, per block):
    scale[4] (8 bytes BF16) + qs[4][16] (64 bytes) = 72 bytes
The size of each tensor does not change, so the header offsets remain valid.
The embedding (also used by the LM Head) is interleaved in the same way.
"""
import os
import re
import shutil
import sys
import time

import numpy as np

SRC = os.path.join("data", "qwen2.5_1.5b_q4.bin")
DST = os.path.join("data", "qwen2.5_1.5b_q4x4.bin")
HDR = os.path.join("tools", "qwen_offsets_1.5b_q4.h")

DIM, INTER, KV, VOCAB = 1536, 8960, 256, 151936
# name -> (rows, cols)
SHAPES = {
    "wq": (DIM, DIM), "wk": (KV, DIM), "wv": (KV, DIM), "wo": (DIM, DIM),
    "wgate": (INTER, DIM), "wup": (INTER, DIM), "wdown": (DIM, INTER),
}


def parse_layer_offsets():
    text = open(HDR, encoding="utf-8").read()
    layers = []
    for block in re.findall(r"// (?:Camada|Layer) \d+\s*\{(.*?)\}", text, re.S):
        d = dict((k, int(v)) for k, v in re.findall(r"\.(\w+)\s*=\s*(\d+)ULL", block))
        layers.append(d)
    return layers


def interleave(raw, rows, cols):
    nb = cols // 32
    a = raw.reshape(rows // 4, 4, nb, 18)
    scales = a[..., :2].transpose(0, 2, 1, 3).reshape(rows // 4, nb, 8)
    qs = a[..., 2:].transpose(0, 2, 1, 3).reshape(rows // 4, nb, 64)
    return np.concatenate([scales, qs], axis=2).reshape(-1)


def main():
    layers = parse_layer_offsets()
    if len(layers) != 28:
        print(f"[ERROR] Expected 28 layers in header, found {len(layers)}")
        return 1

    t0 = time.time()
    print(f"[*] Copying {SRC} -> {DST} ...")
    shutil.copyfile(SRC, DST)

    src = np.memmap(SRC, dtype=np.uint8, mode="r")
    dst = np.memmap(DST, dtype=np.uint8, mode="r+")
    total = 0
    # Embedding / LM Head (offset 0, 151936 x 1536)
    emb_size = VOCAB * (DIM // 32) * 18
    dst[0:emb_size] = interleave(np.asarray(src[0:emb_size]), VOCAB, DIM)
    total += emb_size
    for li, lo in enumerate(layers):
        for name, (rows, cols) in SHAPES.items():
            off = lo[name]
            size = rows * (cols // 32) * 18
            dst[off:off + size] = interleave(np.asarray(src[off:off + size]), rows, cols)
            total += size
        sys.stdout.write(f"\r    Layer {li + 1}/28")
        sys.stdout.flush()
    dst.flush()
    del dst
    print(f"\n[OK] {total / 1e6:.1f} MB reorganized in {time.time() - t0:.1f} s -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
