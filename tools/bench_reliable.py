"""bench_reliable.py - Benchmark confiavel do motor Qwen 1.5B INT4.

- Gera exatamente N tokens (greedy, ignora EOS) por execucao.
- Repete R vezes por configuracao e reporta MEDIANA, minimo e maximo.
- Verifica o BENCH_HASH: se mudar entre execucoes/versoes, a saida mudou.

Uso: python tools/bench_reliable.py [N_TOKENS] [REPETICOES] [configs...]
     configs no formato THREADS:POLICY, ex: 4:PASSIVE 4:ACTIVE
"""
import os
import re
import statistics
import subprocess
import sys

EXE = os.path.join("tools", "qwen_1.5b_q4_native.exe")

n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cfg_args = sys.argv[3:] or ["4:PASSIVE", "4:ACTIVE", "2:PASSIVE"]
configs = [(int(c.split(":")[0]), c.split(":")[1].upper()) for c in cfg_args]

def run_once(threads, policy):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_WAIT_POLICY"] = policy
    out = subprocess.run([EXE, "--bench", str(n_tokens)], capture_output=True,
                         env=env, timeout=900).stdout.decode("utf-8", "replace")
    def grab(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else float("nan")
    h = re.search(r"BENCH_HASH=([0-9a-f]+)", out)
    return {
        "tok_s": grab(r"([0-9.]+)\s*tokens/segundo"),
        "attn": grab(r"cam\)\s*:\s*([0-9.]+)\s*ms"),
        "mlp": grab(r"MLP[^\n]*:\s*([0-9.]+)\s*ms"),
        "lm": grab(r"LM Head[^\n]*:\s*([0-9.]+)\s*ms"),
        "hash": h.group(1) if h else "?",
    }

print(f"Benchmark: {n_tokens} tokens x {reps} repeticoes por configuracao (greedy)\n")
summary = []
for threads, policy in configs:
    runs = []
    for r in range(reps):
        res = run_once(threads, policy)
        runs.append(res)
        print(f"  [{threads}t {policy:<7}] run {r + 1}/{reps}: {res['tok_s']:.2f} tok/s  hash={res['hash']}")
    speeds = [x["tok_s"] for x in runs]
    hashes = {x["hash"] for x in runs}
    summary.append({
        "cfg": f"{threads} / {policy}",
        "med": statistics.median(speeds), "min": min(speeds), "max": max(speeds),
        "ms": 1000.0 / statistics.median(speeds),
        "attn": statistics.median(x["attn"] for x in runs),
        "mlp": statistics.median(x["mlp"] for x in runs),
        "lm": statistics.median(x["lm"] for x in runs),
        "hash": ",".join(sorted(hashes)),
    })
    print()

print("=" * 96)
print(f"{'Threads/Policy':<16}| {'Mediana':>8} | {'Min':>6} | {'Max':>6} | {'ms/tok':>7} | "
      f"{'Attn':>6} | {'MLP':>6} | {'LM':>6} | Hash")
print("-" * 96)
for s in summary:
    print(f"{s['cfg']:<16}| {s['med']:>8.2f} | {s['min']:>6.2f} | {s['max']:>6.2f} | {s['ms']:>7.1f} | "
          f"{s['attn']:>6.1f} | {s['mlp']:>6.1f} | {s['lm']:>6.1f} | {s['hash']}")
print("=" * 96)
all_hashes = {s["hash"] for s in summary}
print("Saida identica em todas as execucoes: " + ("SIM" if len(all_hashes) == 1 and "," not in next(iter(all_hashes)) else "NAO"))
