import os
import subprocess

vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
src = "tools/teste_fused_mlp_fast.c"
exe = "tools/teste_fused_mlp_fast.exe"
bat_path = "tools/_build_avx.bat"

bat_content = f'@echo off\ncall "{vc}" >nul 2>&1\ncl.exe /O2 /arch:AVX /openmp /nologo "{src}" /Fe:"{exe}"\n'
with open(bat_path, "w") as f:
    f.write(bat_content)

res = subprocess.run([bat_path], capture_output=True, text=True)
print("Returncode:", res.returncode)
if res.returncode == 0:
    subprocess.run([exe])
else:
    print(res.stderr or res.stdout)

if os.path.exists(bat_path):
    os.remove(bat_path)
