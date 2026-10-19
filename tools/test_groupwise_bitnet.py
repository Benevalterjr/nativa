#!/usr/bin/env python3
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

def quantize_tensor_groupwise(w: torch.Tensor, group_size: int = 64):
    """
    BitNet b1.58 quantization with block-wise scaling (Group-wise gamma).
    Splits the matrix into groups of 'group_size' elements along the input dimension (K).
    """
    orig_shape = w.shape
    M, K = orig_shape
    # Pad K if needed to be a multiple of group_size
    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w_padded = torch.nn.functional.pad(w, (0, pad_k))
    else:
        w_padded = w
    
    # Reshape to [M, num_groups, group_size]
    num_groups = w_padded.shape[1] // group_size
    w_grouped = w_padded.view(M, num_groups, group_size)
    
    # Local gamma per group
    gamma = torch.mean(torch.abs(w_grouped), dim=-1, keepdim=True)
    eps = 1e-7
    scale = 1.0 / (gamma + eps)
    
    # Local ternary quantization
    w_trits = torch.clamp(torch.round(w_grouped * scale), -1.0, 1.0)
    w_dequant = w_trits * gamma
    
    # Restore original shape
    w_dequant = w_dequant.view(M, -1)
    if pad_k > 0:
        w_dequant = w_dequant[:, :K]
    return w_dequant

def main():
    print("==========================================================================")
    print("  EXPERIMENT: BITNET b1.58 WITH BLOCK-WISE SCALING (GROUP-WISE)")
    print("==========================================================================")
    
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for group_size in [128, 64, 32]:
        print(f"\n[*] Testing Group Size = {group_size}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        
        # Quantize all MLPs with Group-wise BitNet
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
                    with torch.no_grad():
                        w_q = quantize_tensor_groupwise(module.weight.data, group_size=group_size)
                        module.weight.data.copy_(w_q)
                        
        prompt = "Artificial intelligence is"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(inputs["input_ids"], max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0].tolist())
        print(f"    Result (Group={group_size}): \"{text}\"")

if __name__ == "__main__":
    main()
