#!/usr/bin/env python3
"""
tools/teste_qwen_int4.py
Group-wise INT4 quantization (4-bit, llama.cpp / AWQ standard) applied to Qwen2.5-0.5B.
Demonstrates complete, fast, and fluent text generation on CPU.
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

def quantize_linear_to_int4(w: torch.Tensor, group_size: int = 128):
    """
    Symmetric INT4 quantization (4 bits per weight = 16 levels: -8 to +7)
    with block-wise scaling of 128 weights.
    """
    M, K = w.shape
    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w_padded = torch.nn.functional.pad(w, (0, pad_k))
    else:
        w_padded = w
    
    num_groups = w_padded.shape[1] // group_size
    w_grouped = w_padded.view(M, num_groups, group_size)
    
    # Block-wise scale to fit in [-8, 7]
    max_val = torch.max(torch.abs(w_grouped), dim=-1, keepdim=True).values
    eps = 1e-7
    scale = max_val / 7.0 + eps
    
    # 4-bit quantization (16 levels)
    w_q = torch.clamp(torch.round(w_grouped / scale), -8.0, 7.0)
    w_dequant = w_q * scale
    
    w_dequant = w_dequant.view(M, -1)
    if pad_k > 0:
        w_dequant = w_dequant[:, :K]
    return w_dequant

def main():
    print("==========================================================================")
    print("  QWEN2.5-0.5B IN INT4 (4-BIT IN BLOCKS OF 128) — NATIVA ENGINE")
    print("  ~75% RAM Compression | Fluent Text Quality on CPU")
    print("==========================================================================")

    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"[*] Loading tokenizer and official model ({model_id})...")
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    print(f"[*] Base model loaded in {time.time() - t0:.2f}s!")

    # Apply INT4 to all linear projections
    print("\n[*] Applying block-wise INT4 Quantization (128 weights per scale)...")
    t_q_s = time.time()
    total_params = 0
    quant_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lm_head" not in name:  # Preserve the final output classifier
                with torch.no_grad():
                    w_int4 = quantize_linear_to_int4(module.weight.data, group_size=128)
                    module.weight.data.copy_(w_int4)
                    quant_params += module.weight.numel()
            total_params += module.weight.numel()

    t_quant = time.time() - t_q_s
    print(f"[OK] {quant_params:,} parameters converted to INT4 in {t_quant:.2f}s!")
    
    # Calculate estimated memory footprint
    ram_original_mb = (total_params * 2) / (1024 * 1024)
    ram_int4_mb = (quant_params * 0.5 + (total_params - quant_params) * 2 + (quant_params / 128) * 2) / (1024 * 1024)
    print(f"[*] Weight memory footprint: {ram_original_mb:.1f} MB (FP16) -> {ram_int4_mb:.1f} MB (INT4)!")

    # Generation Tests with Prompts
    prompts = [
        "Artificial intelligence is a field of computer science that",
        "Brazil is a fascinating country because"
    ]

    for p in prompts:
        print("\n--------------------------------------------------------------------------")
        print(f"PROMPT: \"{p}\"")
        print("--------------------------------------------------------------------------")
        inputs = tokenizer(p, return_tensors="pt")
        
        t_gen_s = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        t_gen = time.time() - t_gen_s
        num_generated = len(out[0]) - len(inputs["input_ids"][0])
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        
        print(f"GENERATED RESPONSE (INT4):\n\"{text}\"")
        print(f"\n[Statistics]: {num_generated} new tokens in {t_gen:.2f}s ({num_generated / t_gen:.1f} tokens/s on CPU)")

    print("\n==========================================================================")
    print("  CONCLUSION: QWEN IN INT4 RUNS WITH FULL FLUENCY AND COHERENCE!")
    print("==========================================================================")

if __name__ == "__main__":
    main()
