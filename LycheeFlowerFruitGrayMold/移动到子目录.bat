@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================================================
echo 将文件移动到LycheeFlowerFruitGrayMold子目录
echo ========================================================================
echo.

set SUBDIR=LycheeFlowerFruitGrayMold

echo 步骤1: 检查当前状态
git status --short
echo.

echo 步骤2: 创建临时目录并移动文件
if exist %SUBDIR% (
    echo 子目录已存在，先清理...
    rmdir /s /q %SUBDIR% 2>nul
)

mkdir %SUBDIR%
echo 已创建子目录: %SUBDIR%
echo.

echo 步骤3: 移动文件到子目录（排除.git和子目录本身）
for %%f in (*.*) do (
    if not "%%f"==".git" (
        if not "%%f"=="%SUBDIR%" (
            move "%%f" "%SUBDIR%\" >nul 2>&1
            echo 移动: %%f
        )
    )
)

echo.
echo 步骤4: 移动目录到子目录（排除.git和子目录本身）
for /d %%d in (*) do (
    if not "%%d"==".git" (
        if not "%%d"=="%SUBDIR%" (
            move "%%d" "%SUBDIR%\" >nul 2>&1
            echo 移动目录: %%d
        )
    )
)

echo.
echo 步骤5: 添加所有更改
git add .
echo.

echo 步骤6: 提交更改
git commit -m "Reorganize: Move all files to LycheeFlowerFruitGrayMold subdirectory"
echo.

echo 步骤7: 推送到远程仓库
git push origin main
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！文件已移动到LycheeFlowerFruitGrayMold子目录并推送
    echo ========================================================================
) else (
    echo.
    echo ========================================================================
    echo 推送失败，请检查网络连接和权限
    echo ========================================================================
)

echo.
pause

