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
print(f"Total tokens Nativa: {len(ids_nativa)}")
print(f"Nativa (primeiros 20): {ids_nativa[:20].tolist() if hasattr(ids_nativa, 'tolist') else list(ids_nativa[:20])}")

# 2. llama-tokenize oficial com --ids e --no-bos
# Para comparar de forma rápida e segura sem sobrecarregar stdout, salvamos uma amostra de texto dos primeiros ~2500 caracteres (ou arquivo temporário)
# Ou passamos o arquivo inteiro e lemos os IDs
print("\nExecutando llama-tokenize.exe oficial...")
cmd = [LLAMA_TOKENIZE_EXE, "-m", GGUF_MODEL, "-f", TEXT_FILE, "--ids", "--no-bos"]
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

if res.returncode != 0:
    print("ERRO ao executar llama-tokenize:")
    print(res.stderr)
    sys.exit(1)

# llama-tokenize --ids imprime uma lista formatada como JSON / Python: [1, 2, 3, ...]
try:
    ids_llama = json.loads(res.stdout.strip())
except Exception as e:
    # Fallback se tiver quebras de linha ou caracteres extras
    raw = res.stdout.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        ids_llama = json.loads(raw[start:end+1])
    else:
        raise e

print(f"Total tokens llama.cpp: {len(ids_llama)}")
print(f"llama.cpp (primeiros 20): {ids_llama[:20]}")

# 3. Comparação de Paridade
ids_n = list(ids_nativa[:512])
ids_l = ids_llama[:512]

print(f"\nPrimeiros 512 IDs são idênticos? {ids_n == ids_l}")

divergences = 0
for i, (a, b) in enumerate(zip(ids_n, ids_l)):
    if a != b:
        if divergences == 0:
            print(f"PRIMEIRA DIVERGÊNCIA no índice {i}: Nativa={a} vs llama.cpp={b}")
        divergences += 1
        if divergences <= 10:
            tok_a = tok_nativa.decode([a])
            print(f"  [{i:3d}] Nativa ID={a:<7} ({tok_a!r}) vs llama.cpp ID={b}")

if divergences == 0:
    print("[PARIDADE CONFIRMADA 100%] Todos os 512 tokens iniciais têm exatamente os mesmos IDs!")
else:
    print(f"Total de divergências nos primeiros 512 tokens: {divergences}")
