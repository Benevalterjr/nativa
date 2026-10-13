#!/usr/bin/env python3
"""chat_qwen.py - Chat interativo com Qwen2.5-0.5B via Motor Nativo C99 do Nativa."""

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
        print(f"[ERRO] Executável nativo não encontrado: {EXE_PATH}")
        return 1

    print("=" * 70)
    print("  NATIVA AI - CHAT INTERATIVO COM QWEN 2.5 (MOTOR NATIVO C99)")
    print("=" * 70)
    print("• Arquitetura: Transformer Qwen2.5-0.5B (24 Camadas, GQA)")
    print("• Runtime:     Binário compilado C99 com OpenMP e AVX SIMD")
    print("• Memória:     Zero-Copy Win32 mmap (instantâneo)")
    print("• Velocidade:  ~10 tokens/segundo na CPU")
    print("• Comandos:    Digite 'sair' para encerrar a conversa.")
    print("=" * 70)
    print("\n[*] Inicializando motor nativo C e tokenizador...")

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

    # Aguarda o motor C ficar pronto
    while True:
        line = proc.stdout.readline()
        if not line:
            print("[ERRO] Motor nativo encerrou inesperadamente.")
            return 1
        if "[NATIVA_CHAT_READY]" in line:
            break

    print("[*] Motor C pronto! Você já pode conversar com o Qwen.\n")

    is_first_turn = True

    try:
        while True:
            try:
                user_msg = input("Você > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_msg:
                continue
            if user_msg.lower() in ("sair", "exit", "quit", ":q"):
                print("\nEncerrando conversa...")
                break

            if is_first_turn:
                prompt_text = (
                    "Conversa com um assistente prestativo:\n"
                    f"Usuario: {user_msg}\n"
                    "Assistente:"
                )
                is_first_turn = False
            else:
                prompt_text = f"\nUsuario: {user_msg}\nAssistente:"

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

                if "\nUsuario:" in buf or "\nHumano:" in buf:
                    while "[TURN_DONE]" not in buf:
                        extra = proc.stdout.read(1)
                        if not extra:
                            break
                        buf += extra
                    break

                # Imprime mantendo margem segura de 10 caracteres para verificar marcadores
                if len(buf) - printed_idx > 10:
                    chunk = buf[printed_idx:len(buf) - 10]
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    printed_idx += len(chunk)

            # Imprime cauda limpa
            tail = buf[printed_idx:]
            for marker in ["[TURN_DONE]", "\nUsuario:", "\nHumano:"]:
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

    print("\nSessão de chat finalizada. Até logo, Bene!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
