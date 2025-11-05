#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将文件移动到LycheeFlowerFruitGrayMold子目录并推送到GitHub
"""

import os
import subprocess
import shutil
from pathlib import Path

SUBDIR = "LycheeFlowerFruitGrayMold"

def run_command(cmd, cwd=None, check=False):
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
        if result.stderr and result.returncode != 0:
            print(f"错误: {result.stderr}")
        if check and result.returncode != 0:
            return False
        return True
    except Exception as e:
        print(f"执行错误: {e}")
        return False

def main():
    print("=" * 70)
    print("将文件移动到LycheeFlowerFruitGrayMold子目录并推送到GitHub")
    print("=" * 70)
    
    current_dir = Path.cwd()
    subdir_path = current_dir / SUBDIR
    
    # 检查是否已经在子目录中
    if current_dir.name == SUBDIR:
        print("已经在子目录中，直接推送")
        os.chdir("..")
        current_dir = Path.cwd()
        subdir_path = current_dir / SUBDIR
    
    # 检查子目录是否存在以及是否已有文件
    if subdir_path.exists() and (subdir_path / "README.md").exists():
        print(f"子目录 {SUBDIR} 已存在且包含文件，直接推送")
    else:
        # 创建子目录
        if not subdir_path.exists():
            subdir_path.mkdir(exist_ok=True)
            print(f"已创建子目录: {SUBDIR}")
        
        # 移动文件和目录
        moved_count = 0
        skip_items = [".git", SUBDIR, "__pycache__", "移动并推送.bat", "移动并推送.py"]
        
        for item in current_dir.iterdir():
            # 跳过特定项目
            if item.name in skip_items:
                continue
            
            # 跳过隐藏文件（保留.gitignore）
            if item.name.startswith(".") and item.name not in [".gitignore", ".gitattributes"]:
                continue
            
            dest = subdir_path / item.name
            
            try:
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
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
    if not run_command("git add .", cwd=current_dir):
        print("Git add 失败")
        return
    
    # 提交
    print("\n提交更改...")
    run_command(
        'git commit -m "Reorganize: Move all files to LycheeFlowerFruitGrayMold subdirectory"',
        cwd=current_dir,
        check=False
    )
    
    # 推送
    print("\n推送到GitHub...")
    success = run_command("git push origin main", cwd=current_dir, check=False)
    
    if success:
        print("\n" + "=" * 70)
        print("成功！文件已移动到子目录并推送到GitHub")
        print("=" * 70)
        print(f"\n访问: https://github.com/fashionfu/Agrometeorology/tree/main/{SUBDIR}")
    else:
        print("\n" + "=" * 70)
        print("推送失败，请检查:")
        print("1. 网络连接")
        print("2. GitHub认证")
        print("3. 仓库权限")
        print("\n可以手动执行: git push origin main")
        print("=" * 70)

if __name__ == "__main__":
    main()

