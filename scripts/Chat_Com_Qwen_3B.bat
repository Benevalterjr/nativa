@echo off
setlocal
cd /d "%~dp0\.."
title Nativa AI - Chat com Qwen2.5-3B-Instruct (INT4 Q4x4 Engine)
echo ======================================================================
echo   Iniciando Chat com Qwen 2.5 (3B Instruct) - Motor INT4 Nativa...
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
%PY_CMD% scripts\chat_qwen_3b.py
echo.
pause
