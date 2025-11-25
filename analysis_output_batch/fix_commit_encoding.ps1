# 修复 Git 提交信息中文乱码问题
# 使用方法：在仓库根目录执行此脚本

# 设置 UTF-8 编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "UTF-8"
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# 配置 Git 使用 UTF-8
Write-Host "配置 Git 使用 UTF-8 编码..." -ForegroundColor Yellow
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global core.quotepath false

# 如果是在临时仓库，修复最后一次提交
$repoPath = "E:\temp_agrometeo_repo"
if (Test-Path $repoPath) {
    Write-Host "`n检测到临时仓库，准备修复提交信息..." -ForegroundColor Yellow
    Push-Location $repoPath
    
    # 修改最后一次提交信息
    $commitMessage = "更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)"
    git commit --amend -m $commitMessage
    
    Write-Host "`n提交信息已修改，是否推送到远程仓库？" -ForegroundColor Yellow
    $response = Read-Host "输入 y 推送，其他键跳过"
    if ($response -eq "y" -or $response -eq "Y") {
        git push origin main --force
        Write-Host "已推送到远程仓库" -ForegroundColor Green
    } else {
        Write-Host "请稍后手动推送: git push origin main --force" -ForegroundColor Yellow
    }
    
    Pop-Location
} else {
    Write-Host "`n临时仓库不存在，请先克隆仓库" -ForegroundColor Yellow
    Write-Host "执行命令: git clone https://github.com/fashionfu/Agrometeorology.git E:\temp_agrometeo_repo" -ForegroundColor Cyan
}

Write-Host "`nGit 编码配置已完成！" -ForegroundColor Green

