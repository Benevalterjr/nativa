@echo off
setlocal
cd /d "%~dp0\.."
title Nativa AI - Chat Interativo com Qwen2.5-0.5B
echo ======================================================================
echo   Iniciando Chat com Qwen 2.5 (Motor Nativo C99)...
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

%PY_CMD% scripts\chat_qwen.py
echo.
pause
