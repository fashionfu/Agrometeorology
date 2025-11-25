@echo off
chcp 65001
cd /d "%~dp0"
python run_full_analysis.py
pause

