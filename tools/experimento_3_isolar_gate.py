#!/usr/bin/env python3
"""
tools/experimento_3_isolar_gate.py — Experimento 3: Isolar a Política de Gating
Mantém a PRECISÃO ESTITAMENTE FIXA (FP16) e o ORÇAMENTO DE TOKENS FIXO:
Sinks = 4, Janela Recente = 16, Tokens Passados Mantidos = 8.
Tamanho do KV Cache idêntico em todos os regimes: exatamente 28 tokens por camada (344.0 KB).

Compara puramente o critério de seleção do gate:
1. Surprisal Gate (Dopamina / RPE: top-8 tokens passados por erro de predição -log P(x))
2. Random Gate (Seleção uniforme aleatória de 8 tokens passados)
3. H2O Gate (Heavy Hitter Oracle: top-8 tokens passados por atenção acumulada)
"""

import os
import sys
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa
from tools.experimento_real_cortex_hipocampo import (
    QwenModelLoaded,
    rmsnorm,
    apply_rope_heads,
    MODEL_DEFAULT_PATH,
    TOKENIZER_DEFAULT_PATH,
    CORPUS_DEFAULT_PATH,
    NUM_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GROUP_SIZE
)


@dataclass
class GateKVEntry:
    pos: int
    k: np.ndarray  # [num_kv_heads, head_dim]
    v: np.ndarray  # [num_kv_heads, head_dim]
    surprisal: float = 0.0
    cum_attn: float = 0.0


class GatedLayerKVCache:
    def __init__(self, mode: str, window_size: int = 16, num_sinks: int = 4, past_budget: int = 8):
        self.mode = mode
        self.window_size = window_size
        self.num_sinks = num_sinks
        self.past_budget = past_budget
        self.entries: List[GateKVEntry] = []

    def append(self, pos: int, k_rot: np.ndarray, v_val: np.ndarray, surprisal: float):
        entry = GateKVEntry(pos=pos, k=k_rot.astype(np.float16), v=v_val.astype(np.float16), surprisal=surprisal)
        self.entries.append(entry)

        curr_len = len(self.entries)
        # Se ultrapassar sinks + past_budget + window_size
        max_total = self.num_sinks + self.past_budget + self.window_size
        if curr_len > max_total:
            # Candidatos para descarte são os tokens entre os sinks e a janela recente
            past_indices = list(range(self.num_sinks, curr_len - self.window_size))
            
            if self.mode == "surprisal":
                # Evict o token de MENOR surprisal (menos saliente / mais previsível)
                min_idx = min(past_indices, key=lambda idx: self.entries[idx].surprisal)
                self.entries.pop(min_idx)

            elif self.mode == "random":
                # Evict aleatório
                drop_idx = int(np.random.choice(past_indices))
                self.entries.pop(drop_idx)

            elif self.mode == "h2o":
                # Evict o token de MENOR atenção acumulada
                min_idx = min(past_indices, key=lambda idx: self.entries[idx].cum_attn)
                self.entries.pop(min_idx)

    def get_keys_and_values(self) -> Tuple[np.ndarray, np.ndarray]:
        K = np.stack([e.k.astype(np.float32) for e in self.entries], axis=0)
        V = np.stack([e.v.astype(np.float32) for e in self.entries], axis=0)
        return K, V

    def memory_bytes(self) -> int:
        dim = NUM_KV_HEADS * HEAD_DIM
        return len(self.entries) * dim * 2 * 2  # FP16 K e V


def evaluate_gate(
    mode: str,
    tokens: List[int],
    model: QwenModelLoaded,
    window_size: int = 16,
    num_sinks: int = 4,
    past_budget: int = 8
) -> Dict[str, Any]:
    print(f"\n---> Avaliando Gate: {mode.upper()} (Sinks: {num_sinks}, Janela: {window_size}, Orçamento Passado: {past_budget})...")
    caches = [GatedLayerKVCache(mode=mode, window_size=window_size, num_sinks=num_sinks, past_budget=past_budget) for _ in range(model.num_layers)]
    
    n_tokens = len(tokens)
    nll_list = []
    correct_count = 0
    t_start = time.perf_counter()
    surprisal_curr = 0.0

    for i in range(n_tokens - 1):
        target_token = tokens[i + 1]
        pos = i
        token_id = tokens[i]

        x = model.embed_tokens[token_id].data.astype(np.float32)

        for l_idx, layer in enumerate(model.layers):
            norm_x = rmsnorm(x, layer.norm1)
            q = (layer.wq.data @ norm_x) + (layer.bq.data if layer.bq is not None else 0.0)
            k = (layer.wk.data @ norm_x) + (layer.bk.data if layer.bk is not None else 0.0)
            v = (layer.wv.data @ norm_x) + (layer.bv.data if layer.bv is not None else 0.0)

            q = q.reshape(NUM_HEADS, HEAD_DIM)
            k = k.reshape(NUM_KV_HEADS, HEAD_DIM)
            v = v.reshape(NUM_KV_HEADS, HEAD_DIM)

            q_rot = apply_rope_heads(q, pos)
            k_rot = apply_rope_heads(k, pos)

            cache = caches[l_idx]
            cache.append(pos, k_rot, v, surprisal=surprisal_curr)

            K_all, V_all = cache.get_keys_and_values()
            K_exp = np.repeat(K_all, GROUP_SIZE, axis=1)
            V_exp = np.repeat(V_all, GROUP_SIZE, axis=1)

            scores = np.einsum('hd,chd->hc', q_rot, K_exp) / math.sqrt(HEAD_DIM)
            max_s = np.max(scores, axis=-1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            attn_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

            # Acumula atenção para o baseline H2O
            avg_attn = np.mean(attn_probs, axis=0)
            for c_idx, e in enumerate(cache.entries):
                e.cum_attn += float(avg_attn[c_idx])

            out_heads = np.einsum('hc,chd->hd', attn_probs, V_exp)
            out_flat = out_heads.reshape(-1)
            x = x + (layer.wo.data @ out_flat)

            norm_x2 = rmsnorm(x, layer.norm2)
            gate = layer.wgate.data @ norm_x2
            up = layer.wup.data @ norm_x2
            gated = (gate / (1.0 + np.exp(-np.clip(gate, -30.0, 30.0)))) * up
            x = x + (layer.wdown.data @ gated)

        final_x = rmsnorm(x, model.final_norm)
        logits = model.embed_tokens.data @ final_x

        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l)
        prob_target = float(probs[target_token])
        nll = -math.log(max(prob_target, 1e-15))
        nll_list.append(nll)
        surprisal_curr = nll

        if int(np.argmax(logits)) == target_token:
            correct_count += 1

        if (i + 1) % 25 == 0 or (i + 1) == (n_tokens - 1):
            curr_ppl = math.exp(np.mean(nll_list))
            print(f"    Passo {i + 1:2d}/{n_tokens - 1} | PPL parcial: {curr_ppl:.2f} | Surprisal: {nll:.2f}")

    t_end = time.perf_counter()
    mean_nll = float(np.mean(nll_list))
    ppl = math.exp(mean_nll)
    acc = 100.0 * correct_count / (n_tokens - 1)
    total_kv_kb = sum(c.memory_bytes() for c in caches) / 1024.0

    return {
        "mode": mode,
        "ppl": ppl,
        "nll": mean_nll,
        "accuracy": acc,
        "kv_kb": total_kv_kb,
        "elapsed_s": t_end - t_start
    }


def main():
    parser = argparse.ArgumentParser(description="Experimento 3: Isolar a Política de Gating")
    parser.add_argument("--tokens", type=int, default=80, help="Número de tokens do WikiText-2")
    parser.add_argument("--layers", type=int, default=24, help="Número de camadas")
    parser.add_argument("--window", type=int, default=16, help="Janela densa recente")
    parser.add_argument("--sinks", type=int, default=4, help="Sinks de atenção")
    parser.add_argument("--past", type=int, default=8, help="Orçamento fixo de tokens passados")
    args = parser.parse_args()

    print("=" * 85)
    print("EXPERIMENTO 3: ISOLAMENTO DO CRITÉRIO DE GATING (PRECISÃO E MEMÓRIA IDÊNTICAS)")
    print(f"Modelo: Qwen2.5-0.5B ({args.layers} camadas) | Dataset: WikiText-2 | Tokens: {args.tokens}")
    print(f"Orçamento Fixo: {args.sinks} Sinks + {args.window} Janela + {args.past} Passados = {args.sinks + args.window + args.past} tokens/camada (FP16)")
    print("=" * 85)

    tok = nativa.Tokenizer(TOKENIZER_DEFAULT_PATH)
    with open(CORPUS_DEFAULT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 200 and not p.strip().startswith("=")]
    target_text = "\n\n".join(paragraphs[:3])
    eval_tokens = tok.encode(target_text).tolist()[:args.tokens]

    model = QwenModelLoaded(MODEL_DEFAULT_PATH, num_layers=args.layers)

    gates = ["surprisal", "random", "h2o"]
    results = []
    for g in gates:
        res = evaluate_gate(g, eval_tokens, model, window_size=args.window, num_sinks=args.sinks, past_budget=args.past)
        results.append(res)

    print("\n" + "=" * 85)
    print(f"{'CRITÉRIO DE GATING':<22} | {'PERPLEXIDADE (PPL)':<18} | {'NLL (nats)':<10} | {'TOP-1 ACC':<10} | {'KV MEMÓRIA'}")
    print("=" * 85)
    for r in results:
        print(f"{r['mode'].upper():<22} | {r['ppl']:<18.2f} | {r['nll']:<10.3f} | {r['accuracy']:<6.1f}%    | {r['kv_kb']:<8.1f} KB")
    print("=" * 85)


if __name__ == "__main__":
    main()
