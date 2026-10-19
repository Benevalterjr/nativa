#!/usr/bin/env python3
"""
tools/test_qwen_bitnet_generation.py
Compares text generation of Qwen2.5-0.5B:
1. Original Model (BF16 / FP32)
2. Quantized Model with BitNet b1.58 (Canonical AbsMean Scaling of the Nativa Language)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Disable broken jax in the environment
sys.modules['jax'] = None
sys.modules['flax'] = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def quantize_tensor_bitnet(w: torch.Tensor):
    """
    Canonical BitNet b1.58 algorithm implemented in Nativa:
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
            # Do not quantize lm_head to preserve vocabulary distribution if desired
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
            
    print(f"[*] Parameters quantized to BitNet b1.58: {quantized_weights:,} of {total_weights:,} ({quantized_weights*100.0/total_weights:.1f}%)")

def main():
    print("==========================================================================")
    print("  REAL TEXT GENERATION TEST: QWEN2.5-0.5B WITH BITNET b1.58")
    print("==========================================================================")

    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"[*] Loading tokenizer and official model ({model_id})...")
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load original model in FP32
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    print(f"[*] Original model loaded in {time.time() - t0:.2f}s!")

    prompt = "Artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"\n[*] Test prompt: \"{prompt}\"")
    print(f"[*] Input tokens: {input_ids.tolist()[0]}")

    # 1. Generation with Original FP32 Model
    print("\n--- [1] GENERATING WITH ORIGINAL QWEN (FP32) ---")
    t_start_orig = time.time()
    with torch.no_grad():
        out_orig = model_orig.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,  # greedy for deterministic comparison
            pad_token_id=tokenizer.eos_token_id
        )
    t_orig = time.time() - t_start_orig
    tokens_orig = out_orig[0].tolist()
    text_orig = tokenizer.decode(tokens_orig)
    print(f"Generated text (Original): \"{text_orig}\"")
    print(f"Generation time: {t_orig:.2f}s ({15.0 / t_orig:.2f} tokens/s)")

    # 2. Apply BitNet b1.58 Quantization
    print("\n--- [2] APPLYING BITNET b1.58 QUANTIZATION (NATIVA) ---")
    t_q_start = time.time()
    apply_bitnet_to_model(model_orig, quantize_attn=False, quantize_mlp=True) # Quantize MLPs first (70% of weights)
    print(f"[*] BitNet quantization completed in {time.time() - t_q_start:.2f}s!")

    # 3. Generation with BitNet on MLPs
    print("\n--- [3] GENERATING WITH QWEN BITNET b1.58 (Ternary MLPs) ---")
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
    print(f"Generated text (BitNet MLP): \"{text_bit}\"")
    print(f"Generation time: {t_bit:.2f}s ({15.0 / t_bit:.2f} tokens/s)")

    # 4. Now also quantize Attention (100% total BitNet)
    print("\n--- [4] APPLYING 100% TOTAL BITNET QUANTIZATION (MLP + ATTENTION) ---")
    apply_bitnet_to_model(model_orig, quantize_attn=True, quantize_mlp=True)
    with torch.no_grad():
        out_total = model_orig.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    text_total = tokenizer.decode(out_total[0].tolist())
    print(f"Generated text (100% Total BitNet): \"{text_total}\"")

    print("\n==========================================================================")
    print("  FINAL GENERATION COMPARISON")
    print("==========================================================================")
    print(f"Prompt:              \"{prompt}\"")
    print(f"Original FP32:       \"{text_orig}\"")
    print(f"BitNet (MLPs 1.58b): \"{text_bit}\"")
    print(f"BitNet (100% 1.58b): \"{text_total}\"")
    print("==========================================================================")

if __name__ == "__main__":
    main()
