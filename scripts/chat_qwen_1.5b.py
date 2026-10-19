#!/usr/bin/env python3
"""chat_qwen_1.5b.py - Interactive chat with Qwen2.5-1.5B-Instruct via Nativa's C99 Native Engine."""

import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import nativa

EXE_PATH = os.path.join(ROOT_DIR, "tools", "qwen_1.5b_q4_native.exe")
if not os.path.exists(EXE_PATH):
    EXE_PATH = os.path.join(ROOT_DIR, "tools", "qwen_1.5b_native.exe")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "data", "tokenizer_qwen.json")


def main():
    if not os.path.exists(EXE_PATH):
        print(f"[ERROR] 1.5B native executable not found: {EXE_PATH}")
        return 1

    print("=" * 70)
    print("  NATIVA AI - CHAT WITH QWEN 2.5 (1.5B INSTRUCT) INT4 (Q4_0) ENGINE")
    print("=" * 70)
    print("• Model:        Qwen2.5-1.5B-Instruct (28 Layers, GQA 12/2 heads)")
    print("• Architecture: C99 INT4 SSSE3 Fused Transformer with OpenMP")
    print("• Memory:       Zero-Copy Win32 mmap (Only ~828 MB in RAM!)")
    print("• Speed:        Measured in real time for each generated response")
    print("• Capability:   Advanced reasoning, mathematics, physics, and fluent language")
    print("• Commands:     Type 'exit' or 'quit' to end the conversation.")
    print("=" * 70)
    print("\n[*] Mapping 828 MB of INT4 weights into virtual memory...")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    tokenizer = nativa.Tokenizer(TOKENIZER_PATH)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OMP_WAIT_POLICY"] = "PASSIVE"

    proc = subprocess.Popen(
        [EXE_PATH, "--interactive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=env,
    )

    # Wait for the C engine to become ready
    while True:
        line = proc.stdout.readline()
        if not line:
            print("[ERROR] Native engine terminated unexpectedly.")
            return 1
        if "[NATIVA_CHAT_READY]" in line:
            break

    print("[*] Qwen 1.5B ready! You can now chat with it.\n")

    is_first_turn = True

    try:
        while True:
            try:
                user_msg = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_msg:
                continue
            if user_msg.lower() in ("sair", "exit", "quit", ":q"):
                print("\nEnding conversation...")
                break

            if is_first_turn:
                prompt_text = (
                    "<|im_start|>system\n"
                    "You are the Qwen2.5-1.5B-Instruct model, an intelligent, helpful assistant with sharp reasoning. Answer clearly and accurately.<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                is_first_turn = False
            else:
                prompt_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

            tokens = tokenizer.encode(prompt_text).tolist()
            if not tokens:
                continue

            max_gen = 1000
            cmd = f"INPUT {len(tokens)} {max_gen} " + " ".join(map(str, tokens)) + "\n"
            proc.stdin.write(cmd)
            proc.stdin.flush()

            sys.stdout.write("Qwen 1.5B > ")
            sys.stdout.flush()

            buf = ""
            printed_idx = 0
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                buf += ch

                if "[TURN_DONE" in buf and "]" in buf[buf.find("[TURN_DONE"): ]:
                    break

                # If [TURN_DONE is forming at the end of the buffer, hold temporarily
                td_pos = buf.find("[TURN_DONE", printed_idx)
                if td_pos != -1:
                    chunk = buf[printed_idx:td_pos]
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        printed_idx = td_pos
                else:
                    # If ending with a potential "[TURN" tag start, hold the last few characters
                    safe_len = len(buf)
                    if "[" in buf[printed_idx:]:
                        last_bracket = buf.rfind("[")
                        if last_bracket >= printed_idx and len(buf) - last_bracket < 12:
                            safe_len = last_bracket
                    if safe_len > printed_idx:
                        chunk = buf[printed_idx:safe_len]
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        printed_idx = safe_len

            stats_str = ""
            if "[TURN_DONE" in buf:
                turn_idx = buf.find("[TURN_DONE")
                # Print any leftover text held before the tag
                if turn_idx > printed_idx:
                    sys.stdout.write(buf[printed_idx:turn_idx].rstrip())
                    sys.stdout.flush()
                meta_part = buf[turn_idx:]
                import re
                m = re.search(r"tokens=(\d+)\s+time=([0-9.]+)\s+speed=([0-9.]+)", meta_part)
                if m:
                    tok_count = int(m.group(1))
                    elap_sec = float(m.group(2))
                    speed_val = float(m.group(3))
                    stats_str = f"[{speed_val:.2f} tokens/s | {tok_count} tokens in {elap_sec:.2f}s]"

            print()
            if stats_str:
                print(f"   \x1b[90m{stats_str}\x1b[0m")
            print()

    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    print("\nSession finished. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
