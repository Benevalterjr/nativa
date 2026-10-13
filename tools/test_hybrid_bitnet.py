#!/usr/bin/env python3
"""
tools/test_hybrid_bitnet.py
Implementacao e Validacao da Quantizacao Hibrida:
BitNet b1.58 (Ternario 5-trits) + Escala por Bloco (Group-wise) + Preservacao de Outliers (1% a 2%)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.modules['jax'] = None
sys.modules['flax'] = None

import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def quantize_hybrid_bitnet(w: torch.Tensor, group_size: int = 64, outlier_ratio: float = 0.01):
    """
    Quantizacao Hibrida:
    1. Identifica os top `outlier_ratio` (ex: 1%) pesos de maior magnitude e mantem em FP32.
    2. Os 99% restantes sao quantizados para ternario {-1, 0, 1} com escala por bloco (group_size).
    """
    orig_shape = w.shape
    M, K = orig_shape
    w_flat = w.clone()

    # 1. Mascara de Outliers (Top 1% maiores em magnitude absoluta)
    k_outliers = int(w.numel() * outlier_ratio)
    if k_outliers > 0:
        flat_abs = torch.abs(w_flat.view(-1))
        thresh = torch.kthvalue(flat_abs, w.numel() - k_outliers).values
        outlier_mask = flat_abs >= thresh
        sparse_outliers = torch.zeros_like(flat_abs)
        sparse_outliers[outlier_mask] = flat_abs[outlier_mask] * torch.sign(w_flat.view(-1)[outlier_mask])
        sparse_outliers = sparse_outliers.view(M, K)
        
        # Zera os outliers da matriz que vai para o BitNet para nao distorcer a escala
        w_dense = w_flat.clone()
        w_dense.view(-1)[outlier_mask] = 0.0
    else:
        sparse_outliers = torch.zeros_like(w)
        w_dense = w_flat

    # 2. Group-wise BitNet nos 99% restantes
    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w_padded = torch.nn.functional.pad(w_dense, (0, pad_k))
    else:
        w_padded = w_dense

    num_groups = w_padded.shape[1] // group_size
    w_grouped = w_padded.view(M, num_groups, group_size)

    # Gamma local por bloco nos dados sem outliers
    gamma = torch.mean(torch.abs(w_grouped), dim=-1, keepdim=True)
    eps = 1e-7
    scale = 1.0 / (gamma + eps)

    # Quantizacao ternaria
    w_trits = torch.clamp(torch.round(w_grouped * scale), -1.0, 1.0)
    w_ternary_dequant = w_trits * gamma

    w_ternary_dequant = w_ternary_dequant.view(M, -1)
    if pad_k > 0:
        w_ternary_dequant = w_ternary_dequant[:, :K]

    # Matriz reconstruida: BitNet Ternario + Outliers Salientes
    w_reconstructed = w_ternary_dequant + sparse_outliers
    return w_reconstructed, k_outliers, w.numel()

def main():
    print("==========================================================================")
    print("  NATIVA v0.3: QUANTIZACAO HIBRIDA (BITNET b1.58 + OUTLIERS + GROUP-WISE)")
    print("  99% dos pesos em {-1, 0, 1} | 1% Outliers em Alta Precisao | Bloco=64")
    print("==========================================================================")

    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt = "A inteligência artificial é"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Testar com diferentes taxas de retencao de outliers: 0.5%, 1.0%, 2.0%
    for outlier_pct in [0.005, 0.01, 0.02]:
        print(f"\n[*] Testando Preservacao de Outliers = {outlier_pct*100:.1f}% com Bloco = 64...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

        total_weights = 0
        total_outliers = 0

        t0 = time.time()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Focar nas camadas densas de maior impacto (MLP e projecoes de atencao)
                if any(k in name for k in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]):
                    with torch.no_grad():
                        w_rec, n_out, n_tot = quantize_hybrid_bitnet(module.weight.data, group_size=64, outlier_ratio=outlier_pct)
                        module.weight.data.copy_(w_rec)
                        total_outliers += n_out
                        total_weights += n_tot

        t_q = time.time() - t0
        ternary_weights = total_weights - total_outliers
        print(f"    - Quantizado em {t_q:.2f}s!")
        print(f"    - Pesos Ternários: {ternary_weights:,} ({ternary_weights*100.0/total_weights:.2f}%)")
        print(f"    - Pesos Outliers:  {total_outliers:,} ({total_outliers*100.0/total_weights:.2f}%)")
        
        # Calcular memoria teorica em bits por peso
        # 1.58 bits para os ternarios + 16 bits para outliers + (16 bits / 64) para os gamas
        bits_per_param = (0.99 * 1.58) + (outlier_pct * 16.0) + (16.0 / 64.0)
        print(f"    - Densidade media de memoria: ~{bits_per_param:.2f} bits/parametro (Reducao de ~{16.0/bits_per_param:.1f}x!)")

        # Geracao de Texto
        t_gen_s = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=25,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t_gen = time.time() - t_gen_s
        text = tokenizer.decode(out[0].tolist())
        print(f"    - Tempo de geracao: {t_gen:.2f}s ({25.0/t_gen:.2f} tok/s)")
        print(f"    - Texto Gerado:\n      \"{text}\"")

if __name__ == "__main__":
    main()
