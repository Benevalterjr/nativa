import os
import subprocess
import sys
import re

configs = [
    (1, "PASSIVE"),
    (1, "ACTIVE"),
    (2, "PASSIVE"),
    (2, "ACTIVE"),
    (4, "PASSIVE"),
    (4, "ACTIVE"),
]

exe_path = os.path.join("tools", "qwen_1.5b_q4_native.exe")

print("=" * 70)
print("  MATRIZ DE BENCHMARK: THREADS (1, 2, 4) x OMP_WAIT_POLICY")
print("=" * 70)

# Hardware info
try:
    cpu_info = subprocess.check_output("wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors", shell=True, text=True)
    print("\n[CPU Info]:\n" + cpu_info.strip())
except Exception as e:
    print(f"CPU info error: {e}")

try:
    mem_info = subprocess.check_output("wmic memorychip get Speed,Capacity,PartNumber", shell=True, text=True)
    print("\n[RAM Info]:\n" + mem_info.strip())
except Exception as e:
    print(f"RAM info error: {e}")

print("\n" + "=" * 70)

results = []

for threads, policy in configs:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_WAIT_POLICY"] = policy

    print(f"\n[*] Executando: Threads={threads} | Policy={policy}...")
    try:
        proc = subprocess.run([exe_path], capture_output=True, text=True, env=env, timeout=120)
        out = proc.stdout
        
        # Parse metrics with linear parsing to prevent ReDoS (python:S5852)
        speed, attn, mlp, lm = 0.0, 0.0, 0.0, 0.0
        for line in out.splitlines():
            line_str = line.strip()
            if "tokens/segundo" in line_str:
                parts = line_str.split()
                if parts:
                    try:
                        speed = float(parts[0])
                    except ValueError:
                        pass
            elif "cam)" in line_str and "ms" in line_str:
                parts = line_str.split(":")
                if len(parts) > 1:
                    try:
                        attn = float(parts[1].replace("ms", "").strip())
                    except ValueError:
                        pass
            elif "MLP" in line_str and "ms" in line_str:
                parts = line_str.split(":")
                if len(parts) > 1:
                    try:
                        mlp = float(parts[1].replace("ms", "").strip())
                    except ValueError:
                        pass
            elif "LM Head" in line_str and "ms" in line_str:
                parts = line_str.split(":")
                if len(parts) > 1:
                    try:
                        lm = float(parts[1].replace("ms", "").strip())
                    except ValueError:
                        pass

        results.append({
            "threads": threads,
            "policy": policy,
            "speed": speed,
            "attn": attn,
            "mlp": mlp,
            "lm": lm
        })
        print(f"    -> {speed:.2f} tok/s | Attn: {attn:.1f} ms | MLP: {mlp:.1f} ms | LM: {lm:.1f} ms")
    except Exception as e:
        print(f"    -> Erro: {e}")

print("\n" + "=" * 70)
print(f"{'Threads':<8} | {'Policy':<8} | {'Tokens/s':<10} | {'Attn (ms)':<10} | {'MLP (ms)':<10} | {'LM Head (ms)':<12}")
print("-" * 70)
for r in results:
    print(f"{r['threads']:<8} | {r['policy']:<8} | {r['speed']:<10.2f} | {r['attn']:<10.1f} | {r['mlp']:<10.1f} | {r['lm']:<12.1f}")
print("=" * 70)
