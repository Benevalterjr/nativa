#!/usr/bin/env python3
"""
tools/compare_token_nll.py — Diagnóstico Token a Token: Nativa vs llama.cpp
Mede e compara o NLL exato de cada token em condições idênticas de contexto.
"""

import ctypes
import os
import sys
import re
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import nativa

TOKENS_FILE = os.path.join(ROOT, "data", "wiki_512_ids.txt")
TOKENIZER_JSON = os.path.join(ROOT, "data", "tokenizer_qwen.json")
GGUF_MODEL = os.path.join(ROOT, "data", "qwen3_0.6b_q4_k_m.gguf")
NATIVE_LOG = os.path.join(
    r"C:\Users\pa_alisson\.gemini\antigravity\brain\9d689257-6f98-49a8-b53c-d5643c275cea\.system_generated\tasks",
    "task-2676.log"
)

# 1. Carregar IDs e Tokenizer para decodificar texto
with open(TOKENS_FILE, "r") as f:
    token_ids = [int(x) for x in f.read().split()]

tok = nativa.Tokenizer(TOKENIZER_JSON)

# 2. Carregar NLLs do Nativa
nativa_nlls = {}
with open(NATIVE_LOG, "r") as f:
    for line in f:
        m = re.search(r"\[NLL_NATIVA\] pos=\s*(\d+)\s+input=\s*(\d+)\s+target=\s*(\d+)\s+nll=([0-9.]+)\s+best=\s*(\d+)", line)
        if m:
            pos = int(m.group(1))
            nativa_nlls[pos] = {
                "input": int(m.group(2)),
                "target": int(m.group(3)),
                "nll": float(m.group(4)),
                "best": int(m.group(5)),
            }

# 3. Inicializar llama.dll com logs silenciados
abs_dir = os.path.join(ROOT, "tools", "llama_cpp")
os.add_dll_directory(abs_dir)
ggml = ctypes.CDLL(os.path.join(abs_dir, "ggml.dll"))
ggml.ggml_backend_load_all_from_path(abs_dir.encode("utf-8"))

llama = ctypes.CDLL(os.path.join(abs_dir, "llama.dll"))

# Silenciar logs do llama.cpp
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def dummy_log(level, text, user_data): pass
cb_log = LOG_CALLBACK(dummy_log)
llama.llama_log_set(cb_log, None)

llama.llama_backend_init()

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

mparams = (ctypes.c_char * 512)()
llama.llama_model_default_params(ctypes.byref(mparams))
llama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
llama.llama_model_load_from_file.restype = ctypes.c_void_p
model = llama.llama_model_load_from_file(GGUF_MODEL.encode("utf-8"), ctypes.byref(mparams))

cparams = (ctypes.c_char * 1024)()
llama.llama_context_default_params(ctypes.byref(cparams))
llama.llama_init_from_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
llama.llama_init_from_model.restype = ctypes.c_void_p
ctx = llama.llama_init_from_model(model, ctypes.byref(cparams))

llama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
llama.llama_batch_init.restype = llama_batch
batch = llama.llama_batch_init(1, 0, 1)

llama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
llama.llama_decode.restype = ctypes.c_int32

llama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
llama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

n_vocab = 151936
llama_nlls = {}

N_EVAL = min(len(token_ids) - 1, 280)  # cobrir pos 0..32 e pos 256..275
print(f"[*] Avaliando {N_EVAL} tokens no llama.cpp (token a token com KV cache progressivo)...")

for t in range(N_EVAL):
    batch.n_tokens = 1
    batch.token[0] = token_ids[t]
    batch.pos[0] = t
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1

    ret = llama.llama_decode(ctx, batch)
    if ret != 0:
        print(f"Erro no llama_decode na pos {t}")
        break

    logits_ptr = llama.llama_get_logits_ith(ctx, 0)
    logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
    logits[151669:] = -1e9  # mascara tokens de padding fora do vocab do Qwen3

    target = token_ids[t + 1]
    max_l = np.max(logits)
    logsumexp = max_l + np.log(np.sum(np.exp(logits - max_l)))
    logprob = logits[target] - logsumexp
    best = int(np.argmax(logits))

    llama_nlls[t] = {
        "input": token_ids[t],
        "target": target,
        "nll": float(-logprob),
        "best": best
    }

# 4. Formatar e exibir tabela comparativa
print("\n" + "=" * 105)
print("TABELA COMPARATIVA TOKEN A TOKEN: NATIVA (.natq4) vs. LLAMA.CPP (GGUF)")
print("=" * 105)
print(f"{'Pos':>4} | {'Target ID':>9} | {'Token':^14} | {'NLL Nativa':>11} | {'NLL llama':>10} | {'Delta (N-L)':>11} | {'Diag':^8}")
print("-" * 105)

def print_row(pos):
    nat = nativa_nlls.get(pos, {})
    llm = llama_nlls.get(pos, {})
    target = nat.get("target", token_ids[pos + 1])
    tok_str = tok.decode([target])
    # Limpar quebras de linha e caracteres de controle
    tok_repr = repr(tok_str)[1:-1]
    if len(tok_repr) > 12:
        tok_repr = tok_repr[:10] + ".."
    
    n_nll = nat.get("nll", 0.0)
    l_nll = llm.get("nll", 0.0)
    delta = n_nll - l_nll
    diag = "OK" if abs(delta) < 0.5 else ("WARN" if abs(delta) < 2.0 else "DIVERG")
    print(f"{pos:4d} | {target:9d} | {tok_repr:<14} | {n_nll:11.4f} | {l_nll:10.4f} | {delta:+11.4f} | {diag:^8}")

print("[FAIXA A: TOKENS INICIAIS (0 a 16) — PARTIDA A FRIO / SEM CONTEXTO]")
for p in range(17):
    print_row(p)

print("\n[FAIXA B: CONTEXTO INTERMEDIÁRIO (17 a 31)]")
for p in range(17, 32):
    print_row(p)

print("\n[FAIXA C: CONTEXTO SATURADO (256 a 272) — EXATAMENTE O CHUNK 1 DO LLAMA-PERPLEXITY]")
for p in range(256, 273):
    print_row(p)

print("=" * 105)

# Médias
c_nat = [nativa_nlls[p]["nll"] for p in range(17)]
c_llm = [llama_nlls[p]["nll"] for p in range(17)]
w_nat = [nativa_nlls[p]["nll"] for p in range(256, min(N_EVAL, 273))]
w_llm = [llama_nlls[p]["nll"] for p in range(256, min(N_EVAL, 273))]

print(f"Média Faixa A (Pos 0..16)   : Nativa={np.mean(c_nat):.4f} (PPL={math.exp(np.mean(c_nat)):.2f}) | llama.cpp={np.mean(c_llm):.4f} (PPL={math.exp(np.mean(c_llm)):.2f}) | Delta={np.mean(c_nat)-np.mean(c_llm):+.4f}")
print(f"Média Faixa C (Pos 256..272): Nativa={np.mean(w_nat):.4f} (PPL={math.exp(np.mean(w_nat)):.2f}) | llama.cpp={np.mean(w_llm):.4f} (PPL={math.exp(np.mean(w_llm)):.2f}) | Delta={np.mean(w_nat)-np.mean(w_llm):+.4f}")
