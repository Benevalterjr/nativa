#!/usr/bin/env python3
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.modules['jax'] = None
sys.modules['flax'] = None

import time
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def quantize_q4_simulated(w: torch.Tensor, group_size: int = 32):
    """Simulates Nativa BlockQ4 (symmetric INT4 with scale per 32 weights)."""
    orig_shape = w.shape
    M, K = orig_shape
    pad_k = (group_size - (K % group_size)) % group_size
    w_padded = torch.nn.functional.pad(w, (0, pad_k)) if pad_k > 0 else w
    num_groups = w_padded.shape[1] // group_size
    w_grouped = w_padded.view(M, num_groups, group_size)

    # Scale per block of 32
    amax = torch.max(torch.abs(w_grouped), dim=-1, keepdim=True).values
    eps = 1e-7
    scale = amax / 7.0 + eps
    w_q = torch.clamp(torch.round(w_grouped / scale), -8.0, 7.0)
    w_dequant = w_q * scale
    w_dequant = w_dequant.view(M, -1)
    if pad_k > 0:
        w_dequant = w_dequant[:, :K]
    return w_dequant

def quantize_canonical_bitnet(w: torch.Tensor):
    """Canonical BitNet b1.58 (global scaling, ternary {-1, 0, 1})."""
    gamma = torch.mean(torch.abs(w))
    eps = 1e-7
    scale = 1.0 / (gamma + eps)
    w_trits = torch.clamp(torch.round(w * scale), -1.0, 1.0)
    return w_trits * gamma

def quantize_hybrid_bitnet(w: torch.Tensor, group_size: int = 64, outlier_ratio: float = 0.01):
    """Hybrid BitNet: 1% largest outliers in FP32, rest ternary with group_size=64."""
    orig_shape = w.shape
    M, K = orig_shape
    w_flat = w.clone()

    k_outliers = int(w.numel() * outlier_ratio)
    if k_outliers > 0:
        flat_abs = torch.abs(w_flat.view(-1))
        thresh = torch.kthvalue(flat_abs, w.numel() - k_outliers).values
        outlier_mask = flat_abs >= thresh
        sparse_outliers = torch.zeros_like(flat_abs)
        sparse_outliers[outlier_mask] = w_flat.view(-1)[outlier_mask]
        sparse_outliers = sparse_outliers.view(M, K)
        w_dense = w_flat.clone()
        w_dense.view(-1)[outlier_mask] = 0.0
    else:
        sparse_outliers = torch.zeros_like(w)
        w_dense = w_flat

    pad_k = (group_size - (K % group_size)) % group_size
    w_padded = torch.nn.functional.pad(w_dense, (0, pad_k)) if pad_k > 0 else w_dense
    num_groups = w_padded.shape[1] // group_size
    w_grouped = w_padded.view(M, num_groups, group_size)

    gamma = torch.mean(torch.abs(w_grouped), dim=-1, keepdim=True)
    eps = 1e-7
    scale = 1.0 / (gamma + eps)
    w_trits = torch.clamp(torch.round(w_grouped * scale), -1.0, 1.0)
    w_ternary_dequant = (w_trits * gamma).view(M, -1)
    if pad_k > 0:
        w_ternary_dequant = w_ternary_dequant[:, :K]

    return w_ternary_dequant + sparse_outliers

def evaluate_scheme(name, quant_fn, model_id, tokenizer, test_prompt, eval_text):
    print(f"\n---> Avaliando: [{name}]")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    if quant_fn is not None:
        t0 = time.time()
        for mod_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(k in mod_name for k in ["gate_proj", "up_proj", "down_proj"]):
                    with torch.no_grad():
                        module.weight.data.copy_(quant_fn(module.weight.data))
        print(f"     Quantizacao aplicada em {time.time() - t0:.2f}s")

    # 1. Perplexidade / Loss em texto de validacao
    inputs = tokenizer(eval_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
        loss = outputs.loss.item()
        ppl = math.exp(loss) if loss < 20.0 else float("inf")

    # 2. Geracao de texto com prompt
    prompt_inputs = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        gen_out = model.generate(
            prompt_inputs["input_ids"],
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(gen_out[0].tolist(), skip_special_tokens=True)

    print(f"     Loss: {loss:.4f} | Perplexidade (PPL): {ppl:.2f}")
    print(f"     Texto Gerado: \"{gen_text.strip()}\"")
    return {"name": name, "loss": loss, "ppl": ppl, "text": gen_text.strip()}

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print("================================================================================")
    print("  ETAPA 3: AVALIACAO DE QUALIDADE E PERPLEXIDADE NO MODELO REAL")
    print(f"  Modelo Alvo: {model_id}")
    print("================================================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    eval_text = (
        "The transformer architecture has become the standard for large language models. "
        "Attention mechanisms allow tokens to attend across long sequence contexts, while "
        "multi-layer perceptrons project hidden states across high dimensions."
    )
    test_prompt = "O Brasil e um pais com grande diversidade"

    results = []

    # 1. FP32 Baseline
    res_fp32 = evaluate_scheme("FP32 Baseline (Referencia)", None, model_id, tokenizer, test_prompt, eval_text)
    results.append(res_fp32)

    # 2. Q4 Nativa
    res_q4 = evaluate_scheme("Nativa Q4x4 (Blocos de 32)", quantize_q4_simulated, model_id, tokenizer, test_prompt, eval_text)
    results.append(res_q4)

    # 3. Hybrid BitNet (1% outliers em FP32 + Ternario Bloco 64)
    res_hybrid = evaluate_scheme("Hybrid BitNet (1% Outliers FP32 + 99% Ternario Bloco 64)",
                                 lambda w: quantize_hybrid_bitnet(w, group_size=64, outlier_ratio=0.01),
                                 model_id, tokenizer, test_prompt, eval_text)
    results.append(res_hybrid)

    # 4. Canonical BitNet b1.58 puro (Zero Outliers)
    res_canon = evaluate_scheme("BitNet b1.58 Puro (Sem Outliers / Global)", quantize_canonical_bitnet, model_id, tokenizer, test_prompt, eval_text)
    results.append(res_canon)

    print("\n================================================================================")
    print("  QUADRO COMPARATIVO CONSOLIDADO (ETAPA 3)")
    print("================================================================================")
    print(f"{'Configuracao':<40} | {'Loss':<8} | {'PPL':<10} | {'Status Geracao'}")
    print("-" * 80)
    for r in results:
        status = "OK (Coerente)" if r["loss"] < res_fp32["loss"] + 0.8 else "DEGRADADO (Colapso)"
        print(f"{r['name']:<40} | {r['loss']:<8.4f} | {r['ppl']:<10.2f} | {status}")
    print("================================================================================")

if __name__ == "__main__":
    main()
