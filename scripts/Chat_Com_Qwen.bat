@echo off
cd /d "%~dp0\.."
title Nativa AI - Chat Interativo com Qwen2.5-0.5B
echo ======================================================================
echo   Iniciando Chat com Qwen 2.5 (Motor Nativo C99)...
echo ======================================================================
echo.
python scripts\chat_qwen.py
echo.
pause
