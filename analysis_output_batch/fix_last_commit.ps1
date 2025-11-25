# 修复最后一次提交的中文编码问题
# 使用方法：在仓库根目录执行此脚本

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "UTF-8"

Write-Host "正在修复最后一次提交的编码问题..." -ForegroundColor Yellow

# 获取当前分支
$branch = git rev-parse --abbrev-ref HEAD
Write-Host "当前分支: $branch" -ForegroundColor Cyan

# 显示最后一次提交信息
Write-Host "`n最后一次提交信息：" -ForegroundColor Cyan
git log -1 --pretty=format:"%s"

# 询问是否修改
$response = Read-Host "`n是否要修改最后一次提交信息？(y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "已取消操作" -ForegroundColor Yellow
    exit
}

# 输入新的提交信息
$newMessage = Read-Host "请输入新的提交信息（或按回车使用默认）"
if ([string]::IsNullOrWhiteSpace($newMessage)) {
    $newMessage = "更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)"
}

# 修改提交信息
git commit --amend -m $newMessage

Write-Host "`n提交信息已修改" -ForegroundColor Green
Write-Host "新的提交信息：" -ForegroundColor Cyan
git log -1 --pretty=format:"%s"

# 询问是否推送
$pushResponse = Read-Host "`n是否要推送到远程仓库？(y/n)"
if ($pushResponse -eq "y" -or $pushResponse -eq "Y") {
    Write-Host "正在推送到远程仓库..." -ForegroundColor Yellow
    git push origin $branch --force
    Write-Host "推送完成！" -ForegroundColor Green
} else {
    Write-Host "请稍后手动推送: git push origin $branch --force" -ForegroundColor Yellow
}

