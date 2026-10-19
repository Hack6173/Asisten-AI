@echo off
title Asisten-AI Autonomous Intelligence
echo =======================================================
echo    Asisten-AI Autonomous Intelligence Launcher
echo    Autonomous Research, Self-Learning, Self-Evaluation
echo =======================================================
echo.

cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_asisten.py --mode dashboard
) else (
    python run_asisten.py --mode dashboard
)

pause
