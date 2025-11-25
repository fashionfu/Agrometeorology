@echo off
chcp 65001 >nul
echo ========================================
echo 荔枝霜疫霉分析结果文档转换工具
echo ========================================
echo.

REM 设置工作目录
cd /d "%~dp0"

REM 检查pandoc是否安装
where pandoc >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 pandoc，正在尝试使用 Python 转换...
    echo.
    goto :python_convert
) else (
    echo [信息] 检测到 pandoc，使用 pandoc 转换...
    echo.
    goto :pandoc_convert
)

:pandoc_convert
echo 正在使用 pandoc 将 Markdown 转换为 Word 文档...
echo.

pandoc "荔枝霜疫霉分析结果.md" -o "荔枝霜疫霉分析结果.docx" --from markdown --to docx --reference-doc=reference.docx 2>nul

if exist "荔枝霜疫霉分析结果.docx" (
    echo [成功] 文档转换完成！
    echo 输出文件: 荔枝霜疫霉分析结果.docx
    echo.
    goto :end
) else (
    echo [警告] pandoc 转换失败，尝试使用 Python...
    echo.
    goto :python_convert
)

:python_convert
echo 正在使用 Python 将 Markdown 转换为 Word 文档...
echo.

REM 检查 Python 脚本是否存在
if not exist "convert_md_to_docx.py" (
    echo [错误] 找不到 convert_md_to_docx.py 脚本文件
    echo 请确保该文件与 bat 文件在同一目录
    goto :end
)

python convert_md_to_docx.py

if exist "荔枝霜疫霉分析结果.docx" (
    echo.
    echo [成功] 文档转换完成！
    echo 输出文件: 荔枝霜疫霉分析结果.docx
) else (
    echo.
    echo [错误] 转换失败！
    echo 请确保已安装 python-docx 库: pip install python-docx
    echo.
    echo 如果未安装，请运行以下命令：
    echo pip install python-docx
)

:end
echo.
echo ========================================
pause

