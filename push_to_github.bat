@echo off
chcp 65001 >nul
echo 正在推送到 GitHub 仓库的 LycheeFlowerFruitGrayMold 文件夹...
echo.

cd /d "%~dp0"

REM 检查是否已初始化 git
if not exist ".git" (
    echo 错误: 当前目录不是 git 仓库
    pause
    exit /b 1
)

REM 设置远程仓库
git remote remove origin 2>nul
git remote add origin https://github.com/fashionfu/Agrometeorology.git

REM 拉取远程仓库内容（如果存在）
echo 正在拉取远程仓库内容...
git fetch origin main 2>nul
if %errorlevel% equ 0 (
    echo 远程仓库存在，准备合并...
    git checkout -b temp_branch 2>nul
    git branch -D main 2>nul
    git checkout -b main 2>nul
    git pull origin main --allow-unrelated-histories --no-edit 2>nul
) else (
    echo 远程仓库不存在或无法访问，将创建新分支...
)

REM 创建 LycheeFlowerFruitGrayMold 文件夹结构
echo 正在准备文件结构...
if not exist "temp_staging" mkdir temp_staging
xcopy /E /I /Y *.* temp_staging\ 2>nul

REM 切换到临时目录并重新组织文件
cd temp_staging
if not exist "LycheeFlowerFruitGrayMold" mkdir LycheeFlowerFruitGrayMold

REM 将当前所有文件移动到 LycheeFlowerFruitGrayMold 文件夹
for %%f in (*.*) do (
    if not "%%f"=="push_to_github.bat" (
        move "%%f" "LycheeFlowerFruitGrayMold\" 2>nul
    )
)

for /d %%d in (*) do (
    if not "%%d"=="LycheeFlowerFruitGrayMold" if not "%%d"==".git" (
        move "%%d" "LycheeFlowerFruitGrayMold\" 2>nul
    )
)

REM 返回原目录
cd ..

REM 添加所有更改
git add -A
git commit -m "Update LycheeFlowerFruitGrayMold folder" 2>nul

REM 推送到远程仓库
echo 正在推送到远程仓库...
git push -u origin main --force

if %errorlevel% equ 0 (
    echo.
    echo 成功推送到 GitHub!
) else (
    echo.
    echo 推送失败，请检查网络连接和权限
    echo 您可以稍后手动运行: git push -u origin main
)

REM 清理临时文件
if exist "temp_staging" rmdir /S /Q temp_staging 2>nul

echo.
pause

