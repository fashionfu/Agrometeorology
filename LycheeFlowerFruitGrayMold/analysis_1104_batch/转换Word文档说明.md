# 将改善版.md转换为Word文档说明

## 方法一：使用批处理文件（推荐）

直接双击运行：
```
analysis_1104_batch/运行转换脚本.bat
```

## 方法二：使用Python命令

在项目根目录下运行：

```bash
python scripts/md_to_word_improved.py "analysis_1104_batch/改善版.md" -o "analysis_1104_batch/改善版.docx"
```

## 注意事项

1. 确保已安装 `python-docx` 库：
   ```bash
   pip install python-docx
   ```

2. 如果遇到编码问题，请确保：
   - Python文件使用UTF-8编码
   - 终端支持UTF-8编码

3. 生成的Word文档将保存在：
   ```
   analysis_1104_batch/改善版.docx
   ```

## 脚本功能

- ✅ 自动转换Markdown标题为Word标题样式
- ✅ 保留表格格式
- ✅ 处理加粗文本（**文本**）
- ✅ 设置中文字体（宋体、黑体）
- ✅ 设置段落格式（首行缩进、行距）


