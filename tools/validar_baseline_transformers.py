#!/usr/bin/env python3
"""
tools/validar_baseline_transformers.py — Validação Rigorosa de Baseline
Compara os logits gerados pela nossa implementação autoregressiva com
a implementação oficial de referência do HuggingFace Transformers.
"""

import os
import sys
import math
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa
from tools.experimento_real_cortex_hipocampo import (
    QwenModelLoaded,
    LayerKVCache,
    forward_step,
    MODEL_DEFAULT_PATH,
    TOKENIZER_DEFAULT_PATH,
    CORPUS_DEFAULT_PATH
)

SNAPSHOT_DIR = os.path.expanduser(
    r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)


def main():
    print("=" * 80)
    print("EXPERIMENTO 1: VALIDAÇÃO DE BASELINE CONTRA HUGGINGFACE TRANSFORMERS OFICIAL")
    print("=" * 80)

    # 1. Carrega Tokenizer e Texto
    tok = AutoTokenizer.from_pretrained(SNAPSHOT_DIR)
    with open(CORPUS_DEFAULT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 200 and not p.strip().startswith("=")]
    target_text = paragraphs[0][:150]
    
    input_ids = tok(target_text, return_tensors="pt")["input_ids"]
    token_list = input_ids[0].tolist()[:10]  # Primeiros 10 tokens
    n_tokens = len(token_list)
    print(f"[*] Tokens de Teste ({n_tokens} tokens):")
    for idx, t in enumerate(token_list):
        print(f"    [{idx}] ID {t:5d} -> {repr(tok.decode([t]))}")
    print()

    # 2. Executa Modelo Oficial HuggingFace (FP32)
    print("[*] Carregando modelo oficial HuggingFace Transformers...")
    hf_model = AutoModelForCausalLM.from_pretrained(SNAPSHOT_DIR, dtype=torch.float32)
    hf_model.eval()
    
    with torch.no_grad():
        hf_in = torch.tensor([token_list], dtype=torch.long)
        hf_out = hf_model(hf_in)
        hf_logits = hf_out.logits[0].cpu().numpy()  # [n_tokens, vocab_size]
    print(f"[OK] HuggingFace inferência concluída. Shape logits: {hf_logits.shape}")

    # 3. Executa Nossa Implementação Nativa / QwenModelLoaded (Passo a Passo)
    print("\n[*] Carregando implementação customizada via QwenModelLoaded (24 camadas)...")
    our_model = QwenModelLoaded(MODEL_DEFAULT_PATH, num_layers=24)
    caches = [LayerKVCache(mode="full", window_size=32, num_sinks=4) for _ in range(24)]

    our_logits_list = []
    for pos, t_id in enumerate(token_list):
        logits = forward_step(
            token_id=t_id,
            pos=pos,
            model=our_model,
            caches=caches,
            surprisal_prev=0.0,
            threshold_surprisal=0.0
        )
        our_logits_list.append(logits)
    
    our_logits = np.stack(our_logits_list, axis=0)  # [n_tokens, vocab_size]
    print(f"[OK] Implementação customizada concluída. Shape logits: {our_logits.shape}")

    # 4. Auditoria Comparativa Numérica
    print("\n" + "=" * 80)
    print(f"{'POS':<4} | {'TOKEN':<12} | {'MAX ABS DIFF':<14} | {'MEAN ABS DIFF':<14} | {'COS SIMILARITY':<16} | {'TOP-1 MATCH'}")
    print("=" * 80)

    top1_matches = 0
    max_overall_diff = 0.0

    for pos in range(n_tokens):
        hf_l = hf_logits[pos]
        our_l = our_logits[pos]

        abs_diff = np.abs(our_l - hf_l)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        if max_diff > max_overall_diff:
            max_overall_diff = max_diff

        norm_h = np.linalg.norm(hf_l)
        norm_o = np.linalg.norm(our_l)
        cos_sim = float(np.dot(hf_l, our_l) / (norm_h * norm_o))

        hf_top1 = int(np.argmax(hf_l))
        our_top1 = int(np.argmax(our_l))
        match = (hf_top1 == our_top1)
        if match:
            top1_matches += 1

        t_str = tok.decode([token_list[pos]])
        match_str = "SIM (100%)" if match else f"DIVERGÊNCIA (HF:{hf_top1} vs OUR:{our_top1})"
        print(f"{pos:<4} | {t_str:<12} | {max_diff:<14.6f} | {mean_diff:<14.6f} | {cos_sim:<16.8f} | {match_str}")

    print("=" * 80)
    print(f"Resumo da Validação Baseline:")
    print(f"- Acurácia de Pareamento Top-1: {top1_matches}/{n_tokens} ({100.0 * top1_matches / n_tokens:.1f}%)")
    print(f"- Maior diferença absoluta global: {max_overall_diff:.6f}")
    if max_overall_diff < 0.05:
        print("[SUCESSO TOTAL] A implementação customizada é numericamente equivalente ao HuggingFace oficial!")
    else:
        print("[ALERTA] Diferença observada acima do limiar de precisão FP32. Inspecionando componentes...")


if __name__ == "__main__":
    main()
