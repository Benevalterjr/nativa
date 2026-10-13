@echo off
setlocal
cd /d "%~dp0\.."
title Nativa AI - Engine Universal (.natq4)
echo ======================================================================
echo   Nativa AI - Runtime Universal com Deteccao Dinamica de CPU
echo ======================================================================
echo.

set "PY_CMD="
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0..\.venv\Scripts\python.exe"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3"
    ) else (
        set "PY_CMD=python"
    )
)

set OMP_NUM_THREADS=4
set OMP_WAIT_POLICY=PASSIVE
%PY_CMD% tools\chat_natq4.py data\qwen2.5_3b.natq4
echo.
pause
