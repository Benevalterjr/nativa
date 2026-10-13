import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa

TOKENIZER_PATH = os.path.join("data", "tokenizer_qwen.json")
tokenizer = nativa.Tokenizer(TOKENIZER_PATH)

questions = [
    "Você conhece os zeros de Riemann?",
    "O que é a teoria da relatividade geral?",
    "Escreva uma breve poesia sobre o céu estrelado."
]

env = os.environ.copy()
env["OMP_WAIT_POLICY"] = "ACTIVE"

for q in questions:
    prompt_text = (
        "<|im_start|>system\n"
        "Você é o modelo Qwen2.5-1.5B-Instruct, um assistente inteligente. Responda em português de forma clara e precisa.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    tokens = tokenizer.encode(prompt_text).tolist()
    
    print("\n" + "=" * 70)
    print(f"PERGUNTA: {q}")
    print("=" * 70)
    
    cmd = ["tools/qwen_1.5b_q4_native.exe", "--tokens", str(len(tokens))] + [str(t) for t in tokens] + ["80"]
    subprocess.run(cmd, env=env)
