# Git 中文提交信息乱码解决方案

## 问题原因
在 Windows 系统上，Git 默认可能使用 GBK 编码，导致中文提交信息在 GitHub 上显示为乱码。

## 解决方案

### 1. 配置 Git 使用 UTF-8 编码（已完成）

以下配置已设置，确保以后提交都使用 UTF-8 编码：

```bash
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global core.quotepath false
```

### 2. 修复已推送的提交信息

如果需要修复之前已经推送的提交信息，可以按以下步骤操作：

#### 方法一：修改最后一次提交信息（推荐）

```bash
# 克隆仓库到临时目录
cd E:\
git clone https://github.com/fashionfu/Agrometeorology.git temp_fix_repo

cd temp_fix_repo
cd LycheeFlowerFruitGrayMold

# 修改最后一次提交信息
git commit --amend -m "更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)"

# 强制推送（注意：这会修改历史记录）
git push origin main --force
```

#### 方法二：使用环境变量（适用于 PowerShell）

在提交前设置环境变量：

```powershell
$env:LANG="zh_CN.UTF-8"
$env:LC_ALL="zh_CN.UTF-8"
$env:PYTHONIOENCODING="UTF-8"

# 然后执行提交
git commit -m "更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)"
```

### 3. 验证配置

检查配置是否生效：

```bash
git config --global --list | grep i18n
```

应该看到：
```
i18n.commitencoding=utf-8
i18n.logoutputencoding=utf-8
```

### 4. 以后提交时确保编码正确

在 PowerShell 中提交时，可以设置环境变量：

```powershell
$env:PYTHONIOENCODING="UTF-8"
git commit -m "你的中文提交信息"
```

或者在提交脚本中添加：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
git commit -m "你的中文提交信息"
```

## 注意事项

1. **修改历史记录**：如果使用 `git commit --amend` 或 `git push --force`，会修改 Git 历史记录。如果其他人已经克隆了仓库，需要通知他们重新克隆。

2. **GitHub 显示**：即使本地显示正常，GitHub 上也可能因为之前的提交编码问题而显示乱码。修复后，新的提交应该正常显示。

3. **团队协作**：如果是团队项目，建议统一使用 UTF-8 编码，并在项目文档中说明。

## 快速修复脚本

可以创建一个 PowerShell 脚本来确保正确编码：

```powershell
# fix_git_encoding.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "UTF-8"
$env:LANG = "zh_CN.UTF-8"

# 配置 Git
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global core.quotepath false

Write-Host "Git 编码配置已更新为 UTF-8" -ForegroundColor Green
```

