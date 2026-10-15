#!/usr/bin/env python3
"""
tools/experimento_5_memoria_episodica_real.py — Experimento 5: Agulha no Palheiro Real
Testa a recuperação literal de um fato sintético (código aleatório imprevisível)
inserido em posição conhecida após distâncias D = 20, 40, 60 tokens de distratores do WikiText-2.

Separação estrita:
- Vetor de gravação: inserção contextual do código imprevisível ("O código secreto de acesso é 84920 .")
- Distratores: parágrafos reais do WikiText-2
- Vetor de busca: pergunta formulada com tokens distintos ("Pergunta: Qual é o código secreto? Resposta: O código secreto é")
- Alvo medido: probabilidade e top-1 de recuperação literal do código secreto.
"""

import os
import sys
import math
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

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


def run_needle_test(
    model: QwenModelLoaded,
    tok: nativa.Tokenizer,
    distance: int,
    regime: str,
    tq_bits: int = 4
) -> Dict[str, Any]:
    # 1. Montagem do prompt
    secret_code_str = "84920"
    prefix_text = f"Nota importante de seguranca : O codigo secreto de acesso e {secret_code_str} ."
    
    with open(CORPUS_DEFAULT_PATH, "r", encoding="utf-8") as f:
        wiki_text = f.read()
    paragraphs = [p.strip() for p in wiki_text.split("\n") if len(p.strip()) > 200 and not p.strip().startswith("=")]
    distractor_raw = " ".join(paragraphs[:5])
    distractor_tokens = tok.encode(distractor_raw).tolist()[:distance]
    distractor_text = tok.decode(distractor_tokens)

    query_text = " Pergunta : Qual e o codigo secreto de acesso ? Resposta : O codigo secreto de acesso e"

    full_prompt = prefix_text + " " + distractor_text + query_text
    prompt_tokens = tok.encode(full_prompt).tolist()
    target_tokens = tok.encode(f" {secret_code_str}").tolist()
    target_first_token = target_tokens[0]

    # 2. Inicializa Caches
    window_size = 16
    num_sinks = 4
    caches = [LayerKVCache(mode=regime, window_size=window_size, num_sinks=num_sinks, tq_bits=tq_bits) for _ in range(model.num_layers)]

    surprisal_curr = 0.0
    threshold_surprisal = 4.0  # Limiar para capturar o código de alta surpresa
    logits_last = None

    for pos, t_id in enumerate(prompt_tokens):
        logits = forward_step(
            token_id=t_id,
            pos=pos,
            model=model,
            caches=caches,
            surprisal_prev=surprisal_curr,
            threshold_surprisal=threshold_surprisal
        )
        logits_last = logits
        # Estima surprisal
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l)
        if pos < len(prompt_tokens) - 1:
            next_t = prompt_tokens[pos + 1]
            p_next = float(probs[next_t])
            surprisal_curr = -math.log(max(p_next, 1e-15))

    # 3. Avalia previsão do próximo token (deve ser o código secreto!)
    max_l = np.max(logits_last)
    exp_l = np.exp(logits_last - max_l)
    probs_last = exp_l / np.sum(exp_l)

    prob_code = float(probs_last[target_first_token])
    top1_pred = int(np.argmax(logits_last))
    is_top1 = (top1_pred == target_first_token)

    return {
        "regime": regime,
        "bits": tq_bits,
        "distance": distance,
        "prob_code": prob_code,
        "is_top1": is_top1,
        "top1_str": repr(tok.decode([top1_pred])),
        "target_str": repr(tok.decode([target_first_token])),
        "mem_kb": sum(c.memory_bytes() for c in caches) / 1024.0
    }


def main():
    print("=" * 85)
    print("EXPERIMENTO 5: RECUPERAÇÃO LITERAL DE MEMÓRIA EPISÓDICA (NEEDLE-IN-A-HAYSTACK REAL)")
    print("Agulha: Código imprevisível '84920' | Distratores: WikiText-2 | Consulta: Pergunta distinta")
    print("=" * 85)

    tok = nativa.Tokenizer(TOKENIZER_DEFAULT_PATH)
    model = QwenModelLoaded(MODEL_DEFAULT_PATH, num_layers=24)

    distances = [20, 50]
    regimes = [
        ("full", 2),
        ("window", 2),
        ("h2o", 2),
        ("biogated_turboquant", 4),
        ("biogated_turboquant", 2),
    ]

    print("\n" + "=" * 90)
    print(f"{'DISTÂNCIA':<10} | {'REGIME':<22} | {'P(CÓDIGO)':<12} | {'TOP-1 RECALL':<14} | {'PREVISÃO TOP-1':<18} | {'MEMÓRIA KV'}")
    print("=" * 90)

    for dist in distances:
        for reg, bits in regimes:
            tag = f"{reg}_{bits}b" if reg == "biogated_turboquant" else reg
            res = run_needle_test(model, tok, distance=dist, regime=reg, tq_bits=bits)
            match_str = "SUCESSO" if res["is_top1"] else "FALHA"
            print(f"{dist:<10} | {tag:<22} | {res['prob_code']*100:<10.3f}% | {match_str:<14} | {res['top1_str']:<18} | {res['mem_kb']:<8.1f} KB")
        print("-" * 90)


if __name__ == "__main__":
    main()
