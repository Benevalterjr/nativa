@echo off
cd /d "%~dp0\.."
title Nativa AI - Engine Universal (.natq4)
echo ======================================================================
echo   Nativa AI - Runtime Universal com Deteccao Dinamica de CPU
echo ======================================================================
echo.
set OMP_NUM_THREADS=4
set OMP_WAIT_POLICY=PASSIVE
python tools\chat_natq4.py data\qwen2.5_3b.natq4
echo.
pause
