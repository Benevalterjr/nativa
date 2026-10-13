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

def main():
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Canal-hibrido: proteger as primeiras/ultimas projecoes e quantizar gate/up com 10% outliers
    # Vamos testar quantizacao nos MLPs protegendo os top 10% pesos de maior impacto
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "down_proj" in name or "gate_proj" in name:
                w = module.weight.data
                M, K = w.shape
                # 10% outliers
                k_out = int(w.numel() * 0.10)
                flat_abs = torch.abs(w.view(-1))
                thresh = torch.kthvalue(flat_abs, w.numel() - k_out).values
                mask = flat_abs >= thresh
                
                sparse = torch.zeros_like(flat_abs)
                sparse[mask] = w.view(-1)[mask]
                
                dense = w.view(-1).clone()
                dense[mask] = 0.0
                dense = dense.view(M, K)
                
                gamma = torch.mean(torch.abs(dense)) + 1e-7
                w_trits = torch.clamp(torch.round(dense / gamma), -1.0, 1.0)
                w_rec = (w_trits * gamma) + sparse.view(M, K)
                module.weight.data.copy_(w_rec)

    prompt = "A inteligência artificial é"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print("Resultado 10% outliers nos MLPs:", tokenizer.decode(out[0].tolist()))

if __name__ == "__main__":
    main()
