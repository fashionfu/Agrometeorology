@echo off
chcp 65001 >nul
echo ============================================
echo 使用 Git Bash 修复提交信息（推荐方法）
echo ============================================
echo.

REM 检查 Git Bash 是否存在
where bash >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到 Git Bash
    echo 请确保已安装 Git for Windows
    pause
    exit /b 1
)

REM 检查临时仓库是否存在，如果不存在则克隆
if not exist "E:\temp_agrometeo_repo" (
    echo 克隆仓库...
    git clone https://github.com/fashionfu/Agrometeorology.git E:\temp_agrometeo_repo
)

REM 使用 Git Bash 运行修复脚本
bash -c "cd /e/temp_agrometeo_repo && export LANG=zh_CN.UTF-8 && export LC_ALL=zh_CN.UTF-8 && echo '更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)' > commit_msg.txt && git commit --amend -F commit_msg.txt && rm commit_msg.txt && git log -1 --pretty=format:'%%s' && echo. && echo 是否推送到远程仓库？ && read -p '(y/n) ' -n 1 -r && echo && if [[ $REPLY =~ ^[Yy]$ ]]; then git push origin main --force && echo 推送完成！; else echo 请稍后手动推送; fi"

pause

