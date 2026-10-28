# tools/gerar_modelo_nat.py — Emite models/meu_llm12.nat (12 camadas desenroladas)
import os

NL, D, HEADS, DH, FF = 12, 128, 4, 32, 512

def layer_block(i, pos, ind):
    p = ' ' * ind
    heads = '\n'.join(
        f'{p}q{h} = apply_rope(q_all[{h*DH}..{(h+1)*DH}], {pos})\n'
        f'{p}k{h} = apply_rope(k_all[{h*DH}..{(h+1)*DH}], {pos})\n'
        f'{p}v{h} = v_all[{h*DH}..{(h+1)*DH}]\n'
        f'{p}c{i}h{h}.update([k{h}], [v{h}])\n'
        f'{p}a{h} = sdpa([q{h}], c{i}h{h}.k, c{i}h{h}.v)'
        for h in range(HEADS))
    return (f'{p}# Camada {i}\n'
            f'{p}xn = rmsnorm(x, n1_{i})\n'
            f'{p}q_all = xn @ wq_{i}\n{p}k_all = xn @ wk_{i}\n{p}v_all = xn @ wv_{i}\n'
            f'{heads}\n'
            f'{p}att = concat([a0[0], a1[0], a2[0], a3[0]])\n'
            f'{p}x = x + (att @ wo_{i})\n'
            f'{p}xn = rmsnorm(x, n2_{i})\n'
            f'{p}x = x + ((silu(xn @ wg_{i}) * (xn @ wu_{i})) @ wd_{i})')

W = [f'''    n1_{i} = load_safetensors(arquivo, "layers.{i}.n1")
    n2_{i} = load_safetensors(arquivo, "layers.{i}.n2")
    wq_{i} = load_safetensors(arquivo, "layers.{i}.wq.weight")
    wk_{i} = load_safetensors(arquivo, "layers.{i}.wk.weight")
    wv_{i} = load_safetensors(arquivo, "layers.{i}.wv.weight")
    wo_{i} = load_safetensors(arquivo, "layers.{i}.wo.weight")
    wg_{i} = load_safetensors(arquivo, "layers.{i}.gate.weight")
    wu_{i} = load_safetensors(arquivo, "layers.{i}.up.weight")
    wd_{i} = load_safetensors(arquivo, "layers.{i}.down.weight")''' for i in range(NL)]

C = [f'    c{i}h{h} = KVCache(max_seq, {DH})' for i in range(NL) for h in range(HEADS)]

body_pre = '\n'.join(layer_block(i, 't', 8) for i in range(NL))
body_gen = '\n'.join(layer_block(i, 'pos', 8) for i in range(NL))

nat = f'''# models/meu_llm12.nat — GERADO por tools/gerar_modelo_nat.py (nao edite a mao)
# LLM 12 camadas treinado em PyTorch | 3.7M params | MHA + RoPE + SwiGLU + RMSNorm
fn main():
    print("==========================================================================")
    print("  EXECUTANDO MODELO NATIVA 12 CAMADAS (3.7M PARAMS)")
    print("  MHA 4 Cabecas | RoPE | SwiGLU | RMSNorm | SafeTensors")
    print("==========================================================================")
    arquivo = "data/meu_llm12.safetensors"
    max_seq = 512
    tok = Tokenizer("data/meu_tok.json")
    prompt = "Nativa e uma linguagem para IA"
    tokens = encode(tok, prompt)
    prompt_len = len(tokens)
    n_new = 25

    print(f"Prompt: '{{prompt}}' ({{prompt_len}} tokens)")
    print(f"Tokens do prompt: {{tokens}}")

    print("[*] Carregando pesos SafeTensors...")
    emb     = load_safetensors(arquivo, "emb.weight")
    norm_f  = load_safetensors(arquivo, "norm.weight")
    lm_head = load_safetensors(arquivo, "lm_head.weight")

{chr(10).join(W)}

    print("[*] Inicializando 48 KV Caches (12 camadas x 4 cabecas)...")
{chr(10).join(C)}

    print("[*] Executando Prefill do Contexto...")
    t0 = clock()
    x = zeros([128])
    # ---- PREFILL ----
    for t in 0..prompt_len:
        tid = tokens[t]
        x = emb[tid]
{body_pre}
    x = rmsnorm(x, norm_f)
    t_prefill = clock() - t0
    print(f"    Prefill concluido em {{t_prefill * 1000.0:.2f}} ms")

    print(f"[*] Gerando {{n_new}} novos tokens...")
    t_gen = clock()
    # ---- GERACAO (sampling T=0.8, top-p=0.9) ----
    novos = []
    for step in 0..n_new:
        pos = prompt_len + step
        logits = x @ lm_head
        nid = sample(logits, 0.8, 0.9)
        novos.append(nid)
        x = emb[nid]
{body_gen}
        x = rmsnorm(x, norm_f)

    total_time = clock() - t_gen
    speed = 0.0
    if total_time > 0.0:
        speed = n_new / total_time
    print(f"    Geracao concluida em {{total_time * 1000.0:.2f}} ms ({{speed:.2f}} tok/s)")
    print(f"Tokens gerados: {{novos}}")
    texto_gerado = decode(tok, novos)
    print(f"Texto gerado: '{{texto_gerado}}'")
'''

os.makedirs('models', exist_ok=True)
with open('models/meu_llm12.nat', 'w', encoding='utf-8') as f:
    f.write(nat)
print(f'OK -> models/meu_llm12.nat ({len(nat.splitlines())} linhas, {NL} camadas)')
