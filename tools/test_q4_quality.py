import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa

TOKENIZER_PATH = os.path.join("data", "tokenizer_qwen.json")
tokenizer = nativa.Tokenizer(TOKENIZER_PATH)

questions = [
    "Are you familiar with the Riemann zeros?",
    "What is the theory of general relativity?",
    "Write a short poem about the starry sky."
]

env = os.environ.copy()
env["OMP_WAIT_POLICY"] = "ACTIVE"

for q in questions:
    prompt_text = (
        "<|im_start|>system\n"
        "You are Qwen2.5-1.5B-Instruct, an intelligent assistant. Respond clearly and accurately.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    tokens = tokenizer.encode(prompt_text).tolist()
    
    print("\n" + "=" * 70)
    print(f"QUESTION: {q}")
    print("=" * 70)
    
    cmd = ["tools/qwen_1.5b_q4_native.exe", "--tokens", str(len(tokens))] + [str(t) for t in tokens] + ["80"]
    subprocess.run(cmd, env=env)
