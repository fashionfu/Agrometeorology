# PowerShell 脚本：推送到 GitHub 仓库的 LycheeFlowerFruitGrayMold 文件夹
# 编码: UTF-8

$ErrorActionPreference = "Stop"

Write-Host "正在推送到 GitHub 仓库的 LycheeFlowerFruitGrayMold 文件夹..." -ForegroundColor Green
Write-Host ""

# 切换到脚本所在目录
Set-Location $PSScriptRoot

# 检查是否已初始化 git
if (-not (Test-Path ".git")) {
    Write-Host "错误: 当前目录不是 git 仓库" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

# 设置远程仓库
Write-Host "配置远程仓库..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin https://github.com/fashionfu/Agrometeorology.git

# 尝试拉取远程内容
Write-Host "尝试拉取远程仓库内容..." -ForegroundColor Yellow
try {
    git fetch origin main 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "远程仓库存在，准备合并..." -ForegroundColor Yellow
        # 创建临时分支并合并
        git checkout -b temp_merge 2>&1 | Out-Null
        git pull origin main --allow-unrelated-histories --no-edit 2>&1 | Out-Null
        git checkout main 2>&1 | Out-Null
        git merge temp_merge --no-edit 2>&1 | Out-Null
        git branch -D temp_merge 2>&1 | Out-Null
    }
} catch {
    Write-Host "无法拉取远程内容，将创建新提交..." -ForegroundColor Yellow
}

# 创建临时目录来重新组织文件
Write-Host "正在准备文件结构..." -ForegroundColor Yellow
$tempDir = Join-Path $PSScriptRoot "temp_staging"
if (Test-Path $tempDir) {
    Remove-Item -Path $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

# 创建目标文件夹
$targetDir = Join-Path $tempDir "LycheeFlowerFruitGrayMold"
New-Item -ItemType Directory -Path $targetDir | Out-Null

# 复制所有文件到目标文件夹（排除 .git 和脚本文件）
Get-ChildItem -Path $PSScriptRoot -Exclude ".git", "temp_staging", "push_to_github.bat", "push_to_github_simple.ps1" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $targetDir -Recurse -Force
}

# 将文件移回根目录并替换
Write-Host "正在更新文件结构..." -ForegroundColor Yellow
Get-ChildItem -Path $targetDir | ForEach-Object {
    $destPath = Join-Path $PSScriptRoot $_.Name
    if (Test-Path $destPath) {
        Remove-Item -Path $destPath -Recurse -Force
    }
    Move-Item -Path $_.FullName -Destination $PSScriptRoot -Force
}

# 创建 LycheeFlowerFruitGrayMold 文件夹并移动所有内容
$lycheeDir = Join-Path $PSScriptRoot "LycheeFlowerFruitGrayMold"
if (-not (Test-Path $lycheeDir)) {
    New-Item -ItemType Directory -Path $lycheeDir | Out-Null
}

# 移动所有文件到 LycheeFlowerFruitGrayMold（排除 .git 和脚本）
Get-ChildItem -Path $PSScriptRoot -Exclude ".git", "temp_staging", "push_to_github.bat", "push_to_github_simple.ps1", "LycheeFlowerFruitGrayMold" | ForEach-Object {
    $destPath = Join-Path $lycheeDir $_.Name
    if (Test-Path $destPath) {
        Remove-Item -Path $destPath -Recurse -Force
    }
    Move-Item -Path $_.FullName -Destination $lycheeDir -Force
}

# 清理临时目录
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

# 添加所有更改
Write-Host "正在添加文件到 git..." -ForegroundColor Yellow
git add -A

# 提交更改
Write-Host "正在提交更改..." -ForegroundColor Yellow
git commit -m "Update LycheeFlowerFruitGrayMold folder" 2>&1 | Out-Null

# 推送到远程仓库
Write-Host "正在推送到远程仓库..." -ForegroundColor Yellow
Write-Host ""
git push -u origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ 成功推送到 GitHub!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "推送失败，请检查网络连接和权限" -ForegroundColor Red
    Write-Host "您可以稍后手动运行: git push -u origin main" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "按 Enter 键退出"

