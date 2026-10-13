@echo off
cd /d "%~dp0\.."
title Nativa AI - Chat com Qwen2.5-3B-Instruct (INT4 Q4x4 Engine)
echo ======================================================================
echo   Iniciando Chat com Qwen 2.5 (3B Instruct) - Motor INT4 Nativa...
echo ======================================================================
echo.
set OMP_NUM_THREADS=4
set OMP_WAIT_POLICY=PASSIVE
python scripts\chat_qwen_3b.py
echo.
pause
