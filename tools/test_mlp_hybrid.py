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

def quantize_hybrid_bitnet(w: torch.Tensor, group_size: int = 64, outlier_ratio: float = 0.02):
    orig_shape = w.shape
    M, K = orig_shape
    w_flat = w.clone()

    k_outliers = int(w.numel() * outlier_ratio)
    if k_outliers > 0:
        flat_abs = torch.abs(w_flat.view(-1))
        thresh = torch.kthvalue(flat_abs, w.numel() - k_outliers).values
        outlier_mask = flat_abs >= thresh
        sparse_outliers = torch.zeros_like(flat_abs)
        sparse_outliers[outlier_mask] = flat_abs[outlier_mask] * torch.sign(w_flat.view(-1)[outlier_mask])
        sparse_outliers = sparse_outliers.view(M, K)
        
        w_dense = w_flat.clone()
        w_dense.view(-1)[outlier_mask] = 0.0
    else:
        sparse_outliers = torch.zeros_like(w)
        w_dense = w_flat

    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w_padded = torch.nn.functional.pad(w_dense, (0, pad_k))
    else:
        w_padded = w_dense

    num_groups = w_padded.shape[1] // group_size
    w_grouped = w_padded.view(M, num_groups, group_size)

    gamma = torch.mean(torch.abs(w_grouped), dim=-1, keepdim=True)
    eps = 1e-7
    scale = 1.0 / (gamma + eps)

    w_trits = torch.clamp(torch.round(w_grouped * scale), -1.0, 1.0)
    w_ternary_dequant = w_trits * gamma

    w_ternary_dequant = w_ternary_dequant.view(M, -1)
    if pad_k > 0:
        w_ternary_dequant = w_ternary_dequant[:, :K]

    w_reconstructed = w_ternary_dequant + sparse_outliers
    return w_reconstructed

def main():
    print("==========================================================================")
    print("  HYBRID TEST: ATTENTION IN FP32 + MLPs IN HYBRID BITNET b1.58")
    print("==========================================================================")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt = "Artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    for outlier_pct in [0.01, 0.03, 0.05]:
        print(f"\n[*] Testing MLPs with Outliers = {outlier_pct*100:.1f}% and Block = 64...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # MLPs only (70% of total model weights!)
                if any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
                    with torch.no_grad():
                        w_rec = quantize_hybrid_bitnet(module.weight.data, group_size=64, outlier_ratio=outlier_pct)
                        module.weight.data.copy_(w_rec)

        with torch.no_grad():
            out = model.generate(inputs["input_ids"], max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0].tolist())
        print(f"    Generated Text:\n    \"{text}\"")

if __name__ == "__main__":
    main()
