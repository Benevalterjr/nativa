import os
import subprocess
import json
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import nativa

TEXT_FILE = os.path.join(ROOT, "data", "wikitext-2-test.txt")
TOKENIZER_JSON = os.path.join(ROOT, "data", "tokenizer_qwen.json")
GGUF_MODEL = os.path.join(ROOT, "data", "qwen3_0.6b_q4_k_m.gguf")
LLAMA_TOKENIZE_EXE = os.path.join(ROOT, "tools", "llama_cpp", "llama-tokenize.exe")

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Nativa Tokenizer
tok_nativa = nativa.Tokenizer(TOKENIZER_JSON)
ids_nativa = tok_nativa.encode(text)
print(f"Total Nativa tokens: {len(ids_nativa)}")
print(f"Nativa (first 20): {ids_nativa[:20].tolist() if hasattr(ids_nativa, 'tolist') else list(ids_nativa[:20])}")

# 2. Official llama-tokenize with --ids and --no-bos
# To compare quickly and safely without overloading stdout, we can save a text sample of the first ~2500 characters (or temporary file)
# Or pass the entire file and read the IDs
print("\nRunning official llama-tokenize.exe...")
cmd = [LLAMA_TOKENIZE_EXE, "-m", GGUF_MODEL, "-f", TEXT_FILE, "--ids", "--no-bos"]
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

if res.returncode != 0:
    print("ERROR running llama-tokenize:")
    print(res.stderr)
    sys.exit(1)

# llama-tokenize --ids prints a list formatted as JSON / Python: [1, 2, 3, ...]
try:
    ids_llama = json.loads(res.stdout.strip())
except Exception as e:
    # Fallback if there are line breaks or extra characters
    raw = res.stdout.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        ids_llama = json.loads(raw[start:end+1])
    else:
        raise e

print(f"Total llama.cpp tokens: {len(ids_llama)}")
print(f"llama.cpp (first 20): {ids_llama[:20]}")

# 3. Parity Comparison
ids_n = list(ids_nativa[:512])
ids_l = ids_llama[:512]

print(f"\nAre the first 512 IDs identical? {ids_n == ids_l}")

divergences = 0
for i, (a, b) in enumerate(zip(ids_n, ids_l)):
    if a != b:
        if divergences == 0:
            print(f"FIRST DIVERGENCE at index {i}: Nativa={a} vs llama.cpp={b}")
        divergences += 1
        if divergences <= 10:
            tok_a = tok_nativa.decode([a])
            print(f"  [{i:3d}] Nativa ID={a:<7} ({tok_a!r}) vs llama.cpp ID={b}")

if divergences == 0:
    print("[100% PARITY CONFIRMED] All initial 512 tokens have the exact same IDs!")
else:
    print(f"Total divergences in the first 512 tokens: {divergences}")
