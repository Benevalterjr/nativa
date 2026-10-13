#!/usr/bin/env python3
"""
tools/test_qwen_bitnet_generation.py
Compara a geracao de texto do Qwen2.5-0.5B:
1. Modelo Original (BF16 / FP32)
2. Modelo Quantizado com BitNet b1.58 (AbsMean Scaling canônico da Linguagem Nativa)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Desabilitar jax quebrado no ambiente
sys.modules['jax'] = None
sys.modules['flax'] = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def quantize_tensor_bitnet(w: torch.Tensor):
    """
    Algoritmo canônico BitNet b1.58 implementado no Nativa:
    gamma = mean(|W|)
    W_scaled = clip(round(W / (gamma + eps)), -1, 1)
    W_dequant = W_scaled * gamma
    """
    gamma = torch.mean(torch.abs(w))
    eps = 1e-7
    scale = 1.0 / (gamma + eps)
    w_trits = torch.clamp(torch.round(w * scale), -1.0, 1.0)
    w_dequant = w_trits * gamma
    return w_dequant, gamma

def apply_bitnet_to_model(model, quantize_attn=True, quantize_mlp=True):
    total_weights = 0
    quantized_weights = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Nao quantizar lm_head para preservar distribuicao de vocabulario se desejado
            if "lm_head" in name:
                continue
            
            is_attn = any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
            is_mlp = any(k in name for k in ["gate_proj", "up_proj", "down_proj"])
            
            should_quant = (is_attn and quantize_attn) or (is_mlp and quantize_mlp)
            if should_quant:
                with torch.no_grad():
                    w_quant, gamma = quantize_tensor_bitnet(module.weight.data)
                    module.weight.data.copy_(w_quant)
                    quantized_weights += module.weight.numel()
            total_weights += module.weight.numel()
            
    print(f"[*] Parametros quantizados para BitNet b1.58: {quantized_weights:,} de {total_weights:,} ({quantized_weights*100.0/total_weights:.1f}%)")

def main():
    print("==========================================================================")
    print("  TESTE DE GERACAO DE TEXTO REAL: QWEN2.5-0.5B COM BITNET b1.58")
    print("==========================================================================")

    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"[*] Carregando tokenizador e modelo oficial ({model_id})...")
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Carregar modelo original em FP32
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    print(f"[*] Modelo original carregado em {time.time() - t0:.2f}s!")

    prompt = "A inteligência artificial é"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"\n[*] Prompt de teste: \"{prompt}\"")
    print(f"[*] Tokens de entrada: {input_ids.tolist()[0]}")

    # 1. Geracao com Modelo Original FP32
    print("\n--- [1] GERANDO COM QWEN ORIGINAL (FP32) ---")
    t_start_orig = time.time()
    with torch.no_grad():
        out_orig = model_orig.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,  # greedy para comparacao deterministica
            pad_token_id=tokenizer.eos_token_id
        )
    t_orig = time.time() - t_start_orig
    tokens_orig = out_orig[0].tolist()
    text_orig = tokenizer.decode(tokens_orig)
    print(f"Texto gerado (Original): \"{text_orig}\"")
    print(f"Tempo de geracao: {t_orig:.2f}s ({15.0 / t_orig:.2f} tokens/s)")

    # 2. Aplicar Quantizacao BitNet b1.58
    print("\n--- [2] APLICANDO QUANTIZACAO BITNET b1.58 (NATIVA) ---")
    t_q_start = time.time()
    apply_bitnet_to_model(model_orig, quantize_attn=False, quantize_mlp=True) # Quantizar MLPs primeiro (70% dos pesos)
    print(f"[*] Quantizacao BitNet concluida em {time.time() - t_q_start:.2f}s!")

    # 3. Geracao com BitNet nos MLPs
    print("\n--- [3] GERANDO COM QWEN BITNET b1.58 (MLPs Ternarios) ---")
    t_start_bit = time.time()
    with torch.no_grad():
        out_bit = model_orig.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    t_bit = time.time() - t_start_bit
    tokens_bit = out_bit[0].tolist()
    text_bit = tokenizer.decode(tokens_bit)
    print(f"Texto gerado (BitNet MLP): \"{text_bit}\"")
    print(f"Tempo de geracao: {t_bit:.2f}s ({15.0 / t_bit:.2f} tokens/s)")

    # 4. Agora quantizar tambem a Atencao (100% BitNet total)
    print("\n--- [4] APLICANDO QUANTIZACAO BITNET 100% TOTAL (MLP + ATTENTION) ---")
    apply_bitnet_to_model(model_orig, quantize_attn=True, quantize_mlp=True)
    with torch.no_grad():
        out_total = model_orig.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    text_total = tokenizer.decode(out_total[0].tolist())
    print(f"Texto gerado (100% BitNet Total): \"{text_total}\"")

    print("\n==========================================================================")
    print("  COMPARATIVO FINAL DE GERACAO")
    print("==========================================================================")
    print(f"Prompt:            \"{prompt}\"")
    print(f"Original FP32:     \"{text_orig}\"")
    print(f"BitNet (MLPs 1.58b): \"{text_bit}\"")
    print(f"BitNet (100% 1.58b): \"{text_total}\"")
    print("==========================================================================")

if __name__ == "__main__":
    main()
