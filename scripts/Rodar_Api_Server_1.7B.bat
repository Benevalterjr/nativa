@echo off
setlocal
cd /d "%~dp0\.."
title Nativa AI - OpenAI API Server (Qwen3 1.7B INT4)
echo ======================================================================
echo   Iniciando Servidor OpenAI Local com Qwen 3 (1.7B INT4)...
echo   Porta: 8080 - Modelo: data\qwen3_1.7b.natq4
echo ======================================================================
echo.

set "MODEL_PATH=%~dp0..\data\qwen3_1.7b.natq4"
set "TOKENIZER_PATH=%~dp0..\data\qwen3_1.7b.natq4.tokenizer.json"
set "ENGINE_PATH=%~dp0..\tools\nativa_engine.exe"
set "PORT=8080"
set "HOST=127.0.0.1"
set "OMP_NUM_THREADS=4"
set "OMP_WAIT_POLICY=PASSIVE"

if exist "%~dp0..\.venv\Scripts\python.exe" (
    "%~dp0..\.venv\Scripts\python.exe" tools\api_server.py
) else (
    python tools\api_server.py
)

pause
