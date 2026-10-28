# tools/train_meu_llm12.py — Treina LLM 12L e exporta para o runtime Nativa
import os
import sys
import json
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

# Adiciona a raiz para importar nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import nativa

D, HEADS, DH, FF, NL = 128, 4, 32, 512, 12
VOCAB, SEQ, STEPS = 512, 64, 400
ROPE_BASE, EPS = 10000.0, 1e-6   # Casa perfeitamente com apply_rope e rmsnorm do .nat

# ---------- Tokenizer BPE byte-level (vocab 512) ----------
print("[1/3] Treinando Tokenizer BPE byte-level...")
text = open('data/corpus.txt', encoding='utf-8').read()
b2u = nativa._bytes_to_unicode()
ids = list(text.encode('utf-8'))
toks = {i: b2u[i] for i in range(256)}
merges = []

def bpe_apply(ids, a, b, nid):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i+1] == b:
            out.append(nid); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

while len(toks) < VOCAB:
    cnt = collections.Counter(zip(ids, ids[1:]))
    if not cnt: break
    (a, b) = cnt.most_common(1)[0][0]
    nid = len(toks)
    toks[nid] = toks[a] + toks[b]
    merges.append(f"{toks[a]} {toks[b]}")
    ids = bpe_apply(ids, a, b, nid)

def bpe_encode(s):
    out = list(s.encode('utf-8'))
    # Aplica merges sequencialmente
    # Para consistência usamos o próprio Tokenizer do Nativa
    return out

vocab = {tok_str: tid for tid, tok_str in toks.items()}
os.makedirs("data", exist_ok=True)
json.dump({'vocab': vocab, 'merges': merges}, open('data/meu_tok.json', 'w', encoding='utf-8'))

# Valida com nativa.Tokenizer
native_tok = nativa.Tokenizer('data/meu_tok.json')
encoded_corpus = native_tok.encode(text)
ids = encoded_corpus.tolist() if isinstance(encoded_corpus, nativa.Tensor) else list(encoded_corpus)
print(f"Corpus tokenizado: {len(ids)} tokens | vocab: {len(vocab)}")

# ---------- Modelo (Teacher Forcing Paralelo) ----------
print("\n[2/3] Treinando Mini-LLM 12 Camadas em PyTorch...")

class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.g

def rope(x):  # [B, S, HEADS, DH] com contiguous halves casando 100% com apply_rope do Nativa
    half = DH // 2
    inv = 1.0 / (ROPE_BASE ** (torch.arange(0, half).float() / half))
    ang = torch.arange(x.shape[1]).float()[:, None] * inv[None, :]
    c, s = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMS(D), RMS(D)
        self.wq = nn.Linear(D, D, bias=False); self.wk = nn.Linear(D, D, bias=False)
        self.wv = nn.Linear(D, D, bias=False); self.wo = nn.Linear(D, D, bias=False)
        self.gate = nn.Linear(D, FF, bias=False)
        self.up   = nn.Linear(D, FF, bias=False)
        self.down = nn.Linear(FF, D, bias=False)
    def forward(self, x, mask):
        B, S = x.shape[0], x.shape[1]
        h = self.n1(x)
        q = rope(self.wq(h).view(B, S, HEADS, DH)).transpose(1, 2)
        k = rope(self.wk(h).view(B, S, HEADS, DH)).transpose(1, 2)
        v = self.wv(h).view(B, S, HEADS, DH).transpose(1, 2)
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(DH)).masked_fill(mask, float('-inf')).softmax(-1)
        x = x + self.wo((att @ v).transpose(1, 2).reshape(B, S, D))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.norm = RMS(D)
        self.lm = nn.Linear(D, VOCAB, bias=False)
    def forward(self, t):
        S = t.shape[1]
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)[None, None]
        x = self.emb(t)
        for b in self.blocks: x = b(x, mask)
        return self.lm(self.norm(x))

torch.manual_seed(42)
model = LLM()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
data = torch.tensor(ids[: (len(ids)//SEQ)*SEQ]).view(-1, SEQ)

for step in range(STEPS):
    idx = torch.randint(0, data.shape[0], (8,))
    x, y = data[idx][:, :-1], data[idx][:, 1:]
    logits = model(x)
    loss = F.cross_entropy(logits.transpose(1, 2), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"step {step:4d}/{STEPS} | loss {loss.item():.4f}")

# ---------- Exporta com nomes que o .nat carrega ----------
print("\n[3/3] Exportando pesos SafeTensors formatados para Nativa...")
sd = {
    'emb.weight': model.emb.weight.data.float(),
    'norm.weight': model.norm.g.data.float(),
    'lm_head.weight': model.lm.weight.data.T.contiguous().float()   # x @ W no .nat requer W^T
}
for i, b in enumerate(model.blocks):
    sd[f'layers.{i}.n1'] = b.n1.g.data.float()
    sd[f'layers.{i}.n2'] = b.n2.g.data.float()
    for n in ['wq', 'wk', 'wv', 'wo', 'gate', 'up', 'down']:
        sd[f'layers.{i}.{n}.weight'] = getattr(b, n).weight.data.T.contiguous().float()

save_file(sd, 'data/meu_llm12.safetensors')
print("OK -> data/meu_llm12.safetensors + data/meu_tok.json gerados com sucesso!")
