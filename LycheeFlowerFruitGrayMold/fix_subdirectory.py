#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将文件移动到LycheeFlowerFruitGrayMold子目录
"""

import os
import subprocess
import shutil
from pathlib import Path

SUBDIR = "LycheeFlowerFruitGrayMold"

def run_command(cmd, cwd=None):
    """运行命令"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行错误: {e}")
        return False

def main():
    print("=" * 70)
    print("将文件移动到LycheeFlowerFruitGrayMold子目录")
    print("=" * 70)
    
    current_dir = Path.cwd()
    subdir_path = current_dir / SUBDIR
    
    # 检查是否已经在子目录中
    if current_dir.name == SUBDIR:
        print("已经在子目录中，无需移动")
        return
    
    # 检查子目录是否已存在
    if subdir_path.exists():
        print(f"子目录 {SUBDIR} 已存在，先清理...")
        shutil.rmtree(subdir_path)
    
    # 创建子目录
    subdir_path.mkdir(exist_ok=True)
    print(f"已创建子目录: {SUBDIR}")
    
    # 移动文件和目录
    moved_count = 0
    for item in current_dir.iterdir():
        # 跳过.git目录和子目录本身
        if item.name in [".git", SUBDIR, "__pycache__"]:
            continue
        
        # 跳过隐藏文件（如.gitignore等需要保留）
        if item.name.startswith(".") and item.name != ".gitignore":
            continue
        
        dest = subdir_path / item.name
        
        try:
            if item.is_dir():
                shutil.move(str(item), str(dest))
                print(f"移动目录: {item.name} -> {SUBDIR}/")
            else:
                shutil.move(str(item), str(dest))
                print(f"移动文件: {item.name} -> {SUBDIR}/")
            moved_count += 1
        except Exception as e:
            print(f"移动 {item.name} 失败: {e}")
    
    print(f"\n共移动 {moved_count} 个项目")
    
    # 添加更改
    print("\n添加更改到Git...")
    run_command("git add .")
    
    # 提交
    print("\n提交更改...")
    run_command('git commit -m "Reorganize: Move all files to LycheeFlowerFruitGrayMold subdirectory"')
    
    # 推送
    print("\n推送到远程仓库...")
    success = run_command("git push origin main")
    
    if success:
        print("\n" + "=" * 70)
        print("成功！文件已移动到LycheeFlowerFruitGrayMold子目录并推送")
        print("=" * 70)
        print(f"\n访问: https://github.com/fashionfu/Agrometeorology/tree/main/{SUBDIR}")
    else:
        print("\n" + "=" * 70)
        print("推送失败，但文件结构已调整")
        print("请手动执行: git push origin main")
        print("=" * 70)

if __name__ == "__main__":
    main()

