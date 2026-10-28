import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
import time
import tempfile
import subprocess
import nativa
from tools.chat_natq4 import write_vocab_file

MODEL_PATH = "data/qwen2.5_1.5b.natq4"
TOKENIZER_PATH = "data/qwen2.5_1.5b.natq4.tokenizer.json"
ENGINE_PATH = "tools/nativa_engine.exe"

tok = nativa.Tokenizer(TOKENIZER_PATH)
v_tmp = tempfile.NamedTemporaryFile(prefix="vocab-", suffix=".bin", delete=False)
v_path = v_tmp.name
v_tmp.close()
write_vocab_file(tok, v_path)

proc = subprocess.Popen(
    [ENGINE_PATH, "--", MODEL_PATH, "--chat", "--vocab", v_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=0
)

while True:
    line = proc.stdout.readline()
    if "[NATIVA_CHAT_READY]" in line:
        break

from qutrit_test.test_branch_dev_decisions import TEST_CASES

def norm(s):
    return re.sub(r"[\s_\-]+", "", s.lower())

for c in TEST_CASES[:3]:
    # RESET engine context for clean single-shot evaluation
    proc.stdin.write("RESET\n")
    proc.stdin.flush()

    prompt_user = (
        f"Você é um motor de decisão if-statement determinístico.\n"
        f"Estado: {json.dumps(c['state'], ensure_ascii=False)}\n"
        f"Tarefa: {c['task']}\n"
        f"Opções permitidas: [{', '.join(c['choices'].keys())}]\n\n"
        f"Responda APENAS com a opção escolhida exata entre colchetes, por exemplo: [{list(c['choices'].keys())[0]}]. Não use markdown nem explique nada."
    )

    full_prompt = (
        "<|im_start|>system\n"
        "Você é um motor de decisão if-statement determinístico. Responda apenas com a opção escolhida entre colchetes.<|im_end|>\n"
        f"<|im_start|>user\n{prompt_user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    tokens = tok.encode(full_prompt).tolist()
    t0 = time.perf_counter()
    proc.stdin.write(f"INPUT {len(tokens)} 15 " + " ".join(map(str, tokens)) + "\n")
    proc.stdin.flush()

    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        buf += ch
        if "[TURN_DONE" in buf and "]" in buf[buf.find("[TURN_DONE"): ]:
            break
    dt = (time.perf_counter() - t0) * 1000.0

    raw = buf[:buf.find("[TURN_DONE")].strip()
    raw_norm = norm(raw)

    winner = "indefinido"
    for k in c["choices"].keys():
        if norm(k) in raw_norm:
            winner = k
            break

    print(f"[{c['id']}] Expected: {c['expected']} | Winner: {winner} | Match: {winner == c['expected']} | Time: {dt:.1f}ms | Raw: {raw!r}")

proc.stdin.write("QUIT\n")
proc.stdin.flush()
proc.kill()
os.unlink(v_path)
