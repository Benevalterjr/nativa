#!/usr/bin/env python3
"""chat_qwen.py - Interactive chat with Qwen2.5-0.5B via Nativa's C99 Native Engine."""

import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import nativa

EXE_PATH = os.path.join(ROOT_DIR, "tools", "qwen_native.exe")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "data", "tokenizer_qwen.json")


def main():
    if not os.path.exists(EXE_PATH):
        print(f"[ERROR] Native executable not found: {EXE_PATH}")
        return 1

    print("=" * 70)
    print("  NATIVA AI - INTERACTIVE CHAT WITH QWEN 2.5 (C99 NATIVE ENGINE)")
    print("=" * 70)
    print("• Architecture: Qwen2.5-0.5B Transformer (24 Layers, GQA)")
    print("• Runtime:      Compiled C99 binary with OpenMP and AVX SIMD")
    print("• Memory:       Zero-Copy Win32 mmap (instantaneous)")
    print("• Speed:        ~10 tokens/second on CPU")
    print("• Commands:     Type 'exit' or 'quit' to end the conversation.")
    print("=" * 70)
    print("\n[*] Initializing C native engine and tokenizer...")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    tokenizer = nativa.Tokenizer(TOKENIZER_PATH)

    proc = subprocess.Popen(
        [EXE_PATH, "--interactive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )

    # Wait for the C engine to become ready
    while True:
        line = proc.stdout.readline()
        if not line:
            print("[ERROR] Native engine terminated unexpectedly.")
            return 1
        if "[NATIVA_CHAT_READY]" in line:
            break

    print("[*] C engine ready! You can now chat with Qwen.\n")

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
                    "Conversation with a helpful assistant:\n"
                    f"User: {user_msg}\n"
                    "Assistant:"
                )
                is_first_turn = False
            else:
                prompt_text = f"\nUser: {user_msg}\nAssistant:"

            tokens = tokenizer.encode(prompt_text).tolist()
            if not tokens:
                continue

            max_gen = 50
            cmd = f"INPUT {len(tokens)} {max_gen} " + " ".join(map(str, tokens)) + "\n"
            proc.stdin.write(cmd)
            proc.stdin.flush()

            sys.stdout.write("Qwen > ")
            sys.stdout.flush()

            buf = ""
            printed_idx = 0
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                buf += ch

                if "[TURN_DONE]" in buf:
                    break

                if "\nUsuario:" in buf or "\nHumano:" in buf or "\nUser:" in buf or "\nHuman:" in buf:
                    while "[TURN_DONE]" not in buf:
                        extra = proc.stdout.read(1)
                        if not extra:
                            break
                        buf += extra
                    break

                # Print while keeping a safe 10-character margin to check for markers
                if len(buf) - printed_idx > 10:
                    chunk = buf[printed_idx:len(buf) - 10]
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    printed_idx += len(chunk)

            # Print clean tail
            tail = buf[printed_idx:]
            for marker in ["[TURN_DONE]", "\nUsuario:", "\nHumano:", "\nUser:", "\nHuman:"]:
                if marker in tail:
                    tail = tail.split(marker)[0]
            if tail:
                sys.stdout.write(tail.strip())
                sys.stdout.flush()

            print()

    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    print("\nChat session finished. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
