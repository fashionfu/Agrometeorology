@echo off
chcp 65001 >nul
echo ============================================
echo 使用 Git Bash 修复提交信息乱码问题
echo ============================================
echo.

REM 检查 Git Bash
where bash >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到 Git Bash
    echo 请确保已安装 Git for Windows
    echo 下载地址: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM 切换到临时仓库目录
cd /d E:\temp_agrometeo_repo 2>nul || (
    echo 克隆仓库...
    git clone https://github.com/fashionfu/Agrometeorology.git E:\temp_agrometeo_repo
    cd /d E:\temp_agrometeo_repo
)

echo 正在使用 Git Bash 修复提交信息...
echo.

REM 使用 Git Bash 执行修复命令
bash -c "cd /e/temp_agrometeo_repo && export LANG=zh_CN.UTF-8 && export LC_ALL=zh_CN.UTF-8 && export LESSCHARSET=utf-8 && printf '更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)' > /tmp/commit_msg.txt && git commit --amend -F /tmp/commit_msg.txt && echo '' && echo '提交信息已修改：' && git log -1 --pretty=format:'%%s' && echo '' && echo '' && read -p '是否推送到远程仓库？(y/n) ' -n 1 -r && echo && if [[ $REPLY =~ ^[Yy]$ ]]; then git push origin main --force && echo '推送完成！请在 GitHub 网页上查看提交信息。'; else echo '请稍后手动推送: git push origin main --force'; fi && rm -f /tmp/commit_msg.txt"

echo.
echo ============================================
pause

