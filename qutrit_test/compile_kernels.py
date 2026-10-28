import os
import sys
import shutil
import subprocess

def compile_dll():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    c_src = os.path.join(this_dir, "qutrit_kernels.c")
    dll_out = os.path.join(this_dir, "qutrit_kernels.dll")

    print(f"[*] Compiling Qutrit Kernels: {c_src} -> {dll_out}")

    # Check MSVC first
    vcvars_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]

    compiled = False
    for vc in vcvars_candidates:
        if os.path.exists(vc):
            bat = os.path.join(this_dir, "_build_qutrit.bat")
            with open(bat, "w") as bf:
                bf.write(f'@echo off\ncall "{vc}" >nul 2>&1\ncl.exe /LD /O2 /openmp /nologo "{c_src}" /Fe:"{dll_out}"\n')
            proc = subprocess.run([bat], capture_output=True, text=True)
            if proc.returncode == 0 and os.path.exists(dll_out):
                print(f"[OK] Compiled successfully with MSVC (cl.exe /O2 /openmp)!")
                compiled = True
                break
            else:
                print(f"[!] MSVC error: {proc.stderr}\n{proc.stdout}")

    if not compiled:
        gcc = shutil.which("gcc") or shutil.which("clang")
        if gcc:
            cmd = [gcc, "-O3", "-shared", "-fPIC", "-fopenmp", "-mavx2", "-mpopcnt", c_src, "-o", dll_out, "-lm"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and os.path.exists(dll_out):
                print(f"[OK] Compiled successfully with GCC/Clang!")
                compiled = True
            else:
                print(f"[!] GCC error: {proc.stderr}\n{proc.stdout}")

    if not compiled:
        raise RuntimeError("Failed to compile qutrit_kernels.dll with both MSVC and GCC!")

    return dll_out

if __name__ == "__main__":
    compile_dll()
