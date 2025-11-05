@echo off
chcp 65001 >nul
echo ========================================================================
echo GitHub推送脚本 - 荔枝霜疫霉花果预警阈值模型
echo ========================================================================
echo.

echo 步骤1: 检查Git状态
git status
if errorlevel 1 (
    echo 初始化Git仓库...
    git init
)

echo.
echo 步骤2: 添加所有文件
git add .

echo.
echo 步骤3: 提交更改
git commit -m "Add LycheeFlowerFruitGrayMold: 荔枝霜疫霉花果预警阈值模型"

echo.
echo 步骤4: 检查分支
git branch --show-current >nul 2>&1
if errorlevel 1 (
    echo 创建main分支...
    git checkout -b main
)

echo.
echo 步骤5: 设置远程仓库
git remote remove origin 2>nul
git remote add origin https://github.com/fashionfu/Agrometeorology.git

echo.
echo ========================================================================
echo 重要提示：
echo ========================================================================
echo 由于需要将文件推送到主仓库的子目录中，建议使用以下方法：
echo.
echo 方法1：在主仓库中创建子目录（推荐）
echo   1. 克隆主仓库:
echo      git clone https://github.com/fashionfu/Agrometeorology.git
echo      cd Agrometeorology
echo.
echo   2. 创建子目录并复制当前项目的所有文件到 LycheeFlowerFruitGrayMold 目录
echo.
echo   3. 添加并提交:
echo      git add LycheeFlowerFruitGrayMold/
echo      git commit -m "Add LycheeFlowerFruitGrayMold"
echo      git push origin main
echo.
echo 方法2：直接推送（如果主仓库是空的）
echo   git push -u origin main
echo.
echo ========================================================================
pause

