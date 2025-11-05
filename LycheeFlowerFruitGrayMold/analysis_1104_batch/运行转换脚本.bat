@echo off
chcp 65001 >nul
cd /d "%~dp0"
python scripts/md_to_word_improved.py "analysis_1104_batch/改善版.md" -o "analysis_1104_batch/改善版.docx"
pause


