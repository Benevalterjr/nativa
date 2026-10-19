#!/usr/bin/env python3
"""Interactive chat for Qwen2/Qwen3 models compatible with NATQ4X4."""

import os
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nativa

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = os.path.join(ROOT_DIR, "tools", "nativa_engine.exe")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "data", "tokenizer_qwen.json")


def write_vocab_file(tokenizer, path):
    vocab_size = max(tokenizer.inv_vocab, default=-1) + 1
    with open(path, "wb") as f:
        f.write(struct.pack("<I", vocab_size))
        for token_id in range(vocab_size):
            piece = tokenizer.inv_vocab.get(token_id, "")
            if piece.startswith("<0x") and len(piece) == 6 and piece.endswith(">"):
                try:
                    token = bytes([int(piece[3:5], 16)])
                except ValueError:
                    token = piece.encode("utf-8", errors="replace")
            elif piece and all(c in tokenizer.byte_decoder for c in piece):
                token = bytes(tokenizer.byte_decoder[c] for c in piece)
            else:
                token = piece.encode("utf-8", errors="replace")
            if len(token) > 65535:
                raise ValueError(f"Token {token_id} exceeds the runtime limit of 65535 bytes.")
            f.write(struct.pack("<H", len(token)))
            f.write(token)


def main():
    if len(sys.argv) < 2:
        model_path = os.path.join(ROOT_DIR, "data", "qwen2.5_3b.natq4")
    else:
        raw_path = sys.argv[1].strip()
        if not re.fullmatch(r"^[a-zA-Z0-9_\-./\\:]+$", raw_path) or raw_path.startswith("-"):
            print(f"[ERROR] Invalid argument or disallowed character in model path: {raw_path}")
            return 1
        model_path = raw_path

    clean_model_path = os.path.realpath(os.path.abspath(model_path))
    if not os.path.isfile(clean_model_path):
        print(f"[ERROR] Model file not found: {clean_model_path}")
        return 1
    if not clean_model_path.endswith((".natq4", ".bin")):
        print(f"[ERROR] Model file must be .natq4 or .bin: {clean_model_path}")
        return 1
    model_path = clean_model_path

    engine_abs = os.path.abspath(ENGINE_PATH)
    if not os.path.isfile(engine_abs):
        print(f"[ERROR] Universal native engine not found: {engine_abs}")
        return 1

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    companion_tokenizer = model_path + ".tokenizer.json"
    model_tokenizer = os.path.join(os.path.dirname(os.path.abspath(model_path)), "tokenizer.json")
    tokenizer_path = next((p for p in (companion_tokenizer, model_tokenizer, TOKENIZER_PATH) if os.path.isfile(p)), None)
    if not tokenizer_path:
        print("[ERROR] Could not find tokenizer.json next to the model or in the data directory.")
        return 1
    tokenizer = nativa.Tokenizer(tokenizer_path)
    vocab_tmp = tempfile.NamedTemporaryFile(prefix="nativa-vocab-", suffix=".bin", delete=False)
    vocab_path = vocab_tmp.name
    vocab_tmp.close()
    try:
        write_vocab_file(tokenizer, vocab_path)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] Failed to prepare the native vocabulary: {exc}")
        try:
            os.unlink(vocab_path)
        except OSError:
            pass
        return 1

    env = os.environ.copy()
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    proc = subprocess.Popen(
        [engine_abs, "--", model_path, "--chat", "--vocab", os.path.abspath(vocab_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=env,
    )

    # Wait for the C engine to be ready
    while True:
        line = proc.stdout.readline()
        if not line:
            print("[ERROR] Native engine terminated unexpectedly.")
            try:
                os.unlink(vocab_path)
            except OSError:
                pass
            return 1
        if "[NATIVA_CHAT_READY]" in line:
            break

    model_name = os.path.basename(model_path).replace(".natq4", "").upper()
    print(f"[*] Nativa Engine NATQ4 ready for [{model_name}]!\n")

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
                    "You are an intelligent, helpful assistant with sharp reasoning. Answer clearly and accurately.<|im_end|>\n"
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

            sys.stdout.write(f"Nativa [{model_name}] > ")
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

                td_pos = buf.find("[TURN_DONE", printed_idx)
                if td_pos != -1:
                    chunk = buf[printed_idx:td_pos]
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        printed_idx = td_pos
                else:
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
                if turn_idx > printed_idx:
                    sys.stdout.write(buf[printed_idx:turn_idx].rstrip())
                    sys.stdout.flush()
                meta_part = buf[turn_idx:]
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
        try:
            os.unlink(vocab_path)
        except OSError:
            pass

    print("\nSession ended. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
