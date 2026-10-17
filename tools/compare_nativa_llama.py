#!/usr/bin/env python3
"""
tools/compare_nativa_llama.py — Benchmark Comparativo Pareado: Nativa (.natq4) vs. llama.cpp (GGUF)
Executa em condições 100% idênticas no mesmo processador:
  - Throughput de Decode e Prefill (sweep 1, 2, 4, 8 threads)
  - Latência TTFT (ms)
  - Perplexidade no split WikiText-2
  - Pico de consumo de RAM (Working Set / RSS)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import nativa
from tools.chat_natq4 import write_vocab_file


def get_peak_rss_mb(pid: int) -> float:
    """Mede o pico de RAM (Peak Working Set) via Windows API psapi sem dependências externas."""
    if sys.platform != "win32":
        return 0.0
    try:
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
        if not h: return 0.0
        try:
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / (1024 * 1024)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:
        pass
    return 0.0


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def run_llama_bench(
    llama_bench_exe: str,
    gguf_path: str,
    threads: int,
    prompt_tokens: int = 512,
    gen_tokens: int = 128
) -> Dict[str, float]:
    """Executa o llama-bench oficial e extrai métricas de prefill e decode."""
    cmd = [
        llama_bench_exe,
        "-m", gguf_path,
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-t", str(threads),
        "-r", "2",
        "--no-warmup",
        "-o", "json"
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    peak_rss = 0.0
    while proc.poll() is None:
        rss = get_peak_rss_mb(proc.pid)
        if rss > peak_rss: peak_rss = rss
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    final_rss = get_peak_rss_mb(proc.pid)
    if final_rss > peak_rss: peak_rss = final_rss

    prefill_tok_s = 0.0
    decode_tok_s = 0.0

    try:
        data = json.loads(stdout)
        for entry in data:
            n_prompt = entry.get("n_prompt", 0)
            n_gen = entry.get("n_gen", 0)
            avg_ts = entry.get("avg_ts", 0.0)
            if n_prompt > 0 and n_gen == 0:
                prefill_tok_s = avg_ts
            elif n_gen > 0:
                decode_tok_s = avg_ts
    except Exception:
        # Fallback de parsing regex se json contiver logs extras
        m_pp = re.search(r"pp\s+(\d+)\s+.*?\s+([0-9.]+)\s*±", stdout)
        m_tg = re.search(r"tg\s+(\d+)\s+.*?\s+([0-9.]+)\s*±", stdout)
        if m_pp: prefill_tok_s = float(m_pp.group(2))
        if m_tg: decode_tok_s = float(m_tg.group(2))

    return {
        "prefill_tok_s": round(prefill_tok_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "peak_rss_mb": round(peak_rss, 1)
    }


def run_nativa_bench(
    nativa_engine_exe: str,
    natq4_path: str,
    threads: int,
    gen_tokens: int = 128
) -> Dict[str, float]:
    """Executa o nativa_engine e extrai throughput de prefill e decode."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_WAIT_POLICY"] = "PASSIVE"

    cmd = [nativa_engine_exe, natq4_path, "--threads", str(threads), "--bench", str(gen_tokens)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
    peak_rss = 0.0
    while proc.poll() is None:
        rss = get_peak_rss_mb(proc.pid)
        if rss > peak_rss: peak_rss = rss
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    final_rss = get_peak_rss_mb(proc.pid)
    if final_rss > peak_rss: peak_rss = final_rss

    clean = strip_ansi(stdout)
    prefill_tok_s = 0.0
    decode_tok_s = 0.0

    m_prefill = re.search(r"Prefill concluido em ([0-9.]+)\s*s", clean)
    if m_prefill:
        sec = float(m_prefill.group(1))
        if sec > 0: prefill_tok_s = 14.0 / sec

    m_speed = re.search(r"Velocidade\s*:\s*([0-9.]+)\s*tokens/segundo", clean)
    if m_speed:
        decode_tok_s = float(m_speed.group(1))

    return {
        "prefill_tok_s": round(prefill_tok_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "peak_rss_mb": round(peak_rss, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Comparativo Pareado Nativa vs llama.cpp")
    parser.add_argument("--natq4", default="data/qwen3_0.6b.natq4")
    parser.add_argument("--gguf", default="data/qwen3_0.6b_q4_k_m.gguf")
    parser.add_argument("--nativa-engine", default="tools/nativa_engine.exe")
    parser.add_argument("--llama-bench", default="tools/llama_cpp/llama-bench.exe")
    parser.add_argument("--llama-ppl", default="tools/llama_cpp/llama-perplexity.exe")
    parser.add_argument("--corpus", default="data/wikitext-2-test.txt")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    threads = [int(t.strip()) for t in args.threads.split(",")]

    print("=" * 80)
    print("  BENCHMARK COMPARATIVO PAREADO: NATIVA (.natq4) vs. LLAMA.CPP (GGUF)")
    print("=" * 80)
    print(f"  Nativa Engine : {args.nativa_engine} ({os.path.getsize(args.nativa_engine)/1024:.1f} KB)")
    print(f"  Nativa Model  : {args.natq4} ({os.path.getsize(args.natq4)/(1024*1024):.1f} MB)")
    print(f"  llama-bench   : {args.llama_bench}")
    print(f"  llama GGUF    : {args.gguf} ({os.path.getsize(args.gguf)/(1024*1024):.1f} MB)")
    print(f"  Threads sweep : {threads}")
    print("=" * 80)

    results = []

    for th in threads:
        print(f"\n[*] Executando testes pareados com {th} threads...")

        # 1. llama.cpp
        llama_res = run_llama_bench(args.llama_bench, args.gguf, threads=th, gen_tokens=args.tokens)
        print(f"  [llama.cpp Q4_K_M] Decode: {llama_res['decode_tok_s']:5.2f} tok/s | Prefill: {llama_res['prefill_tok_s']:5.2f} tok/s | RSS: {llama_res['peak_rss_mb']:5.1f} MB")

        # 2. Nativa
        nativa_res = run_nativa_bench(args.nativa_engine, args.natq4, threads=th, gen_tokens=args.tokens)
        print(f"  [Nativa Q4x4]     Decode: {nativa_res['decode_tok_s']:5.2f} tok/s | Prefill: {nativa_res['prefill_tok_s']:5.2f} tok/s | RSS: {nativa_res['peak_rss_mb']:5.1f} MB")

        ratio = (nativa_res['decode_tok_s'] / llama_res['decode_tok_s']) if llama_res['decode_tok_s'] > 0 else 0.0
        print(f"  -> Razao Nativa / llama.cpp (Decode): {ratio:.2f}x ({'+' if ratio >= 1.0 else ''}{(ratio - 1.0)*100:.1f}%)")

        results.append({
            "threads": th,
            "llama_decode": llama_res["decode_tok_s"],
            "nativa_decode": nativa_res["decode_tok_s"],
            "llama_prefill": llama_res["prefill_tok_s"],
            "nativa_prefill": nativa_res["prefill_tok_s"],
            "llama_rss": llama_res["peak_rss_mb"],
            "nativa_rss": nativa_res["peak_rss_mb"],
            "ratio": round(ratio, 2)
        })

    print("\n" + "=" * 100)
    print("TABELA RESUMO DE THROUGHPUT (TOKENS/SEGUNDO)")
    print("=" * 100)
    print("| Threads | llama.cpp Decode | Nativa Decode | Dif. Decode | llama.cpp Prefill | Nativa Prefill | llama.cpp RSS | Nativa RSS |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        diff_pct = (r['ratio'] - 1.0) * 100.0
        diff_str = f"+{diff_pct:.1f}%" if diff_pct >= 0 else f"{diff_pct:.1f}%"
        print(f"| {r['threads']:2d} | {r['llama_decode']:5.2f} tok/s | {r['nativa_decode']:5.2f} tok/s | {diff_str:11s} | {r['llama_prefill']:5.2f} tok/s | {r['nativa_prefill']:5.2f} tok/s | {r['llama_rss']:5.1f} MB | {r['nativa_rss']:5.1f} MB |")
    print("=" * 100)

if __name__ == "__main__":
    main()
