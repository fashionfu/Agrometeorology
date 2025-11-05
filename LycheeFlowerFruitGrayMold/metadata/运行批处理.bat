@echo off
chcp 65001 >nul
title 气候数据处理程序
echo.
echo ==========================================
echo    气候数据处理程序
echo ==========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查必要的库
echo 正在检查依赖库...
python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo 正在安装pandas...
    python -m pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
)

python -c "import openpyxl" >nul 2>&1
if errorlevel 1 (
    echo 正在安装openpyxl...
    python -m pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo 开始处理数据...
echo.

REM 运行处理脚本
python run_process.py

echo.
pause

