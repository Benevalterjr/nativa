@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl.exe /LD /O2 /openmp /nologo "G:\dyad-apps\dyad-apps\nativa\qutrit_test\qutrit_kernels.c" /Fe:"G:\dyad-apps\dyad-apps\nativa\qutrit_test\qutrit_kernels.dll"
