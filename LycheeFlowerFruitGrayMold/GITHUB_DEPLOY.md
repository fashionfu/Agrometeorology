# GitHub部署说明

## 将项目推送到GitHub仓库

本项目需要推送到主仓库 `https://github.com/fashionfu/Agrometeorology.git` 的 `LycheeFlowerFruitGrayMold/` 子目录中。

## 方法一：在主仓库中创建子目录（推荐）

### 步骤1：克隆主仓库

```bash
git clone https://github.com/fashionfu/Agrometeorology.git
cd Agrometeorology
```

### 步骤2：创建子目录并复制文件

```bash
# 创建子目录
mkdir -p LycheeFlowerFruitGrayMold

# 复制当前项目的所有文件到子目录
# 注意：请将当前项目的所有文件（不包括.git目录）复制到 LycheeFlowerFruitGrayMold 目录
```

### 步骤3：添加并提交

```bash
# 添加所有文件
git add LycheeFlowerFruitGrayMold/

# 提交更改
git commit -m "Add LycheeFlowerFruitGrayMold: 荔枝霜疫霉花果预警阈值模型"

# 推送到远程仓库
git push origin main
```

## 方法二：使用Git子模块（如果主仓库支持）

如果主仓库已经存在其他内容，可以使用子模块方式：

```bash
# 1. 在当前项目目录初始化git
cd /path/to/your/LycheeFlowerFruitGrayMold
git init
git add .
git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型"

# 2. 在主仓库中添加子模块
cd /path/to/Agrometeorology
git submodule add <your-fork-url>/LycheeFlowerFruitGrayMold.git LycheeFlowerFruitGrayMold
git commit -m "Add LycheeFlowerFruitGrayMold submodule"
git push origin main
```

## 方法三：直接推送到子目录（如果仓库为空）

如果主仓库是空的，可以直接推送：

```bash
# 1. 在当前项目目录初始化git
git init
git add .
git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型"

# 2. 添加远程仓库
git remote add origin https://github.com/fashionfu/Agrometeorology.git

# 3. 推送到main分支
git push -u origin main
```

然后需要：
- 在GitHub网页上手动创建 `LycheeFlowerFruitGrayMold` 目录
- 或者使用GitHub的文件上传功能将文件上传到该目录

## .gitignore 建议

建议创建 `.gitignore` 文件，排除不必要的文件：

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# Excel临时文件
~$*.xlsx
~$*.xls

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db

# 输出目录（可选，根据需求决定是否提交）
# analysis_1104/
# analysis_1104_batch/
# analysis_1105/
```

## 注意事项

1. **数据文件**：`metadata/` 目录中的敏感数据文件建议：
   - 如果数据是公开的，可以直接提交
   - 如果数据是敏感的，应该：
     - 添加到 `.gitignore`
     - 或使用Git LFS（Large File Storage）
     - 或提供示例数据文件

2. **大文件**：如果Excel文件较大，考虑使用Git LFS：
   ```bash
   git lfs install
   git lfs track "*.xlsx"
   git lfs track "*.xls"
   git add .gitattributes
   ```

3. **版本控制**：
   - 建议定期提交代码更改
   - 使用有意义的commit message
   - 考虑使用标签（tags）标记重要版本

## 验证部署

推送完成后，访问以下URL验证：
- https://github.com/fashionfu/Agrometeorology/tree/main/LycheeFlowerFruitGrayMold

应该能看到项目文件列表和README.md。

