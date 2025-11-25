# Git UTF-8 编码配置脚本
# 运行此脚本以配置 Git 使用 UTF-8 编码，解决中文乱码问题

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "UTF-8"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Git UTF-8 编码配置工具" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# 配置 Git 使用 UTF-8
Write-Host "正在配置 Git 编码设置..." -ForegroundColor Yellow

git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global core.quotepath false

Write-Host "✓ Git 编码配置已更新" -ForegroundColor Green
Write-Host ""

# 显示当前配置
Write-Host "当前 Git 编码配置：" -ForegroundColor Cyan
git config --global --list | Select-String -Pattern "i18n|quotepath"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "配置完成！" -ForegroundColor Green
Write-Host ""
Write-Host "提示：" -ForegroundColor Yellow
Write-Host "1. 以后提交时，确保使用 UTF-8 编码" -ForegroundColor White
Write-Host "2. 在 PowerShell 中提交前，可以设置环境变量：" -ForegroundColor White
Write-Host "   `$env:PYTHONIOENCODING='UTF-8'" -ForegroundColor Gray
Write-Host "3. 如果之前的提交信息显示乱码，可以使用 fix_last_commit.ps1 修复" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Cyan

