#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将项目推送到GitHub仓库的LycheeFlowerFruitGrayMold子目录
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """运行命令并返回结果"""
    print(f"\n执行命令: {cmd}")
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
            print(f"错误输出: {result.stderr}")
        if check and result.returncode != 0:
            print(f"命令执行失败，退出码: {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False

def main():
    print("=" * 70)
    print("GitHub部署脚本")
    print("=" * 70)
    
    current_dir = Path.cwd()
    repo_url = "https://github.com/fashionfu/Agrometeorology.git"
    subdir = "LycheeFlowerFruitGrayMold"
    
    print(f"\n当前目录: {current_dir}")
    print(f"目标仓库: {repo_url}")
    print(f"子目录: {subdir}")
    
    # 检查是否已有git仓库
    git_dir = current_dir / ".git"
    if git_dir.exists():
        print("\n检测到已有git仓库")
        # 检查远程仓库
        result = subprocess.run(
            "git remote -v",
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if "Agrometeorology" in result.stdout:
            print("远程仓库已配置")
        else:
            print("添加远程仓库...")
            if not run_command(f'git remote add origin {repo_url}', check=False):
                # 如果已存在，尝试设置URL
                run_command(f'git remote set-url origin {repo_url}')
    else:
        print("\n初始化git仓库...")
        if not run_command("git init"):
            print("初始化失败")
            return
    
    # 添加.gitignore（如果不存在）
    gitignore_file = current_dir / ".gitignore"
    if not gitignore_file.exists():
        print("\n.gitignore文件不存在，已创建")
    
    # 添加所有文件
    print("\n添加文件到git...")
    if not run_command("git add ."):
        print("添加文件失败")
        return
    
    # 检查是否有更改
    result = subprocess.run(
        "git status --porcelain",
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if not result.stdout.strip():
        print("\n没有更改需要提交")
        return
    
    # 提交更改
    print("\n提交更改...")
    commit_msg = "Add LycheeFlowerFruitGrayMold: 荔枝霜疫霉花果预警阈值模型"
    if not run_command(f'git commit -m "{commit_msg}"'):
        print("提交失败")
        return
    
    # 检查分支
    result = subprocess.run(
        "git branch --show-current",
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    branch = result.stdout.strip() or "main"
    
    # 创建main分支（如果不存在）
    if branch != "main":
        print(f"\n当前分支: {branch}，切换到main分支...")
        run_command("git checkout -b main", check=False)
        branch = "main"
    
    # 询问是否推送
    print("\n" + "=" * 70)
    print("准备推送到GitHub")
    print("=" * 70)
    print(f"\n将推送到: {repo_url}")
    print(f"分支: {branch}")
    print("\n注意：")
    print("1. 如果主仓库需要创建子目录，请在GitHub网页上手动创建")
    print("2. 或者使用以下命令克隆主仓库后合并：")
    print(f"   git clone {repo_url}")
    print(f"   cd Agrometeorology")
    print(f"   mkdir -p {subdir}")
    print(f"   # 复制当前项目的文件到 {subdir} 目录")
    print(f"   git add {subdir}/")
    print(f"   git commit -m \"Add {subdir}\"")
    print(f"   git push origin main")
    print("\n是否现在推送？(y/n): ", end="")
    
    # 由于无法交互，我们提供说明
    print("\n\n由于无法交互输入，请手动执行以下命令：")
    print("\n方法1：直接推送（如果主仓库是空的或允许）")
    print(f"  git remote add origin {repo_url}  # 如果还没有添加")
    print(f"  git push -u origin {branch}")
    
    print("\n方法2：在主仓库中创建子目录（推荐）")
    print(f"  # 1. 克隆主仓库")
    print(f"  git clone {repo_url}")
    print(f"  cd Agrometeorology")
    print(f"  # 2. 创建子目录并复制文件")
    print(f"  mkdir -p {subdir}")
    print(f"  # 将当前目录的所有文件复制到 {subdir} 目录")
    print(f"  # 3. 添加并提交")
    print(f"  git add {subdir}/")
    print(f"  git commit -m \"Add {subdir}: 荔枝霜疫霉花果预警阈值模型\"")
    print(f"  git push origin main")
    
    # 尝试设置远程并推送（如果用户已经配置好）
    print("\n\n尝试设置远程仓库...")
    run_command(f'git remote add origin {repo_url}', check=False)
    run_command(f'git remote set-url origin {repo_url}', check=False)
    
    print("\n" + "=" * 70)
    print("部署脚本执行完成")
    print("=" * 70)
    print("\n请根据上述说明手动完成GitHub推送操作。")

if __name__ == "__main__":
    main()

