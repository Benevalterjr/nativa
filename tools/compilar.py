#!/usr/bin/env python3
import os
import sys
import subprocess

VC = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

def compile_c(src, exe, extra_flags=""):
    bat_content = f'@echo off\ncall "{VC}" >nul 2>&1\ncl.exe /O2 /arch:AVX /openmp /nologo {extra_flags} "{src}" /Fe:"{exe}"\n'
    bat_path = "_temp_build.bat"
    with open(bat_path, "w") as f:
        f.write(bat_content)
    try:
        res = subprocess.run([bat_path], capture_output=True, text=True)
        if res.returncode != 0:
            print("[COMPILATION ERROR]")
            print(res.stdout)
            print(res.stderr)
            return False
        else:
            print(f"[OK] Successfully compiled: {exe}")
            return True
    finally:
        if os.path.exists(bat_path):
            os.remove(bat_path)
        base_obj = os.path.splitext(os.path.basename(src))[0] + ".obj"
        if os.path.exists(base_obj):
            try:
                os.remove(base_obj)
            except OSError:
                pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compilar.py <src.c> <out.exe>")
        sys.exit(1)
    src = sys.argv[1]
    exe = sys.argv[2]
    extra = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
    success = compile_c(src, exe, extra)
    sys.exit(0 if success else 1)
