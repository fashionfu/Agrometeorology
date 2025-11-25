# 使用文件方式修复提交信息，确保 UTF-8 编码
# 这是最可靠的方法

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "UTF-8"
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

$repoPath = "E:\temp_agrometeo_repo"
if (-not (Test-Path $repoPath)) {
    Write-Host "克隆仓库..." -ForegroundColor Yellow
    git clone https://github.com/fashionfu/Agrometeorology.git $repoPath
}

Push-Location $repoPath

# 创建 UTF-8 编码的提交信息文件
$commitMsgFile = Join-Path $repoPath "commit_message.txt"
$commitMessage = "更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)"

# 使用 UTF-8 编码保存文件（无 BOM）
[System.IO.File]::WriteAllText($commitMsgFile, $commitMessage, [System.Text.UTF8Encoding]::new($false))

Write-Host "使用文件方式修改提交信息..." -ForegroundColor Yellow
git commit --amend -F $commitMsgFile

# 显示提交信息
Write-Host "`n提交信息已修改，内容：" -ForegroundColor Green
Get-Content $commitMsgFile -Encoding UTF8

# 清理临时文件
Remove-Item $commitMsgFile -Force

# 询问是否推送
Write-Host "`n是否推送到远程仓库？" -ForegroundColor Yellow
$response = Read-Host "输入 y 推送，其他键跳过"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "正在推送..." -ForegroundColor Yellow
    git push origin main --force
    Write-Host "推送完成！请在 GitHub 网页上查看提交信息。" -ForegroundColor Green
} else {
    Write-Host "请稍后手动推送: git push origin main --force" -ForegroundColor Yellow
}

Pop-Location

