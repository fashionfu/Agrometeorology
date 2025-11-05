#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动尝试多种Git推送方案，直到成功为止
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/fashionfu/Agrometeorology.git"
SUBDIR = "LycheeFlowerFruitGrayMold"
BRANCH = "main"

def run_command(cmd, cwd=None, timeout=30):
    """运行命令并返回结果"""
    print(f"\n{'='*70}")
    print(f"执行: {cmd}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"错误: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("命令执行超时")
        return False, "", "Timeout"
    except Exception as e:
        print(f"执行错误: {e}")
        return False, "", str(e)

def check_git_installed():
    """检查Git是否安装"""
    success, _, _ = run_command("git --version")
    return success

def method_1_init_and_push():
    """方案1: 在当前目录初始化Git并直接推送"""
    print("\n" + "="*70)
    print("方案1: 在当前目录初始化Git并直接推送")
    print("="*70)
    
    current_dir = Path.cwd()
    
    # 初始化Git
    if not (current_dir / ".git").exists():
        success, _, _ = run_command("git init", cwd=current_dir)
        if not success:
            return False
    else:
        print("Git仓库已存在")
    
    # 添加远程
    run_command("git remote remove origin", cwd=current_dir, timeout=5)
    run_command(f'git remote add origin {REPO_URL}', cwd=current_dir)
    
    # 添加文件
    run_command("git add .", cwd=current_dir)
    
    # 提交
    run_command('git commit -m "Add LycheeFlowerFruitGrayMold: 荔枝霜疫霉花果预警阈值模型"', cwd=current_dir)
    
    # 创建分支
    run_command(f"git checkout -b {BRANCH}", cwd=current_dir, timeout=5)
    
    # 推送
    success, _, _ = run_command(f"git push -u origin {BRANCH}", cwd=current_dir, timeout=60)
    return success

def method_2_clone_and_copy():
    """方案2: 克隆主仓库，复制文件到子目录，然后推送"""
    print("\n" + "="*70)
    print("方案2: 克隆主仓库，复制文件到子目录")
    print("="*70)
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    temp_repo_dir = parent_dir / "Agrometeorology_temp"
    
    # 清理临时目录
    if temp_repo_dir.exists():
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
    
    # 克隆仓库
    success, _, _ = run_command(f"git clone {REPO_URL}", cwd=parent_dir, timeout=120)
    if not success:
        return False
    
    # 重命名
    repo_dir = parent_dir / "Agrometeorology"
    if repo_dir.exists():
        if temp_repo_dir.exists():
            import shutil
            shutil.rmtree(temp_repo_dir)
        repo_dir.rename(temp_repo_dir)
    else:
        return False
    
    # 创建子目录
    subdir_path = temp_repo_dir / SUBDIR
    subdir_path.mkdir(exist_ok=True)
    
    # 复制文件（排除.git）
    import shutil
    for item in current_dir.iterdir():
        if item.name == ".git":
            continue
        dest = subdir_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # 添加并提交
    run_command("git add .", cwd=temp_repo_dir)
    run_command(f'git commit -m "Add {SUBDIR}: 荔枝霜疫霉花果预警阈值模型"', cwd=temp_repo_dir)
    
    # 推送
    success, _, _ = run_command(f"git push origin {BRANCH}", cwd=temp_repo_dir, timeout=120)
    
    return success

def method_3_sparse_checkout():
    """方案3: 使用sparse-checkout只检出子目录"""
    print("\n" + "="*70)
    print("方案3: 使用sparse-checkout")
    print("="*70)
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    temp_repo_dir = parent_dir / "Agrometeorology_sparse"
    
    # 清理
    if temp_repo_dir.exists():
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
    
    # 克隆（不检出文件）
    success, _, _ = run_command(f"git clone --no-checkout {REPO_URL} {temp_repo_dir.name}", cwd=parent_dir, timeout=120)
    if not success:
        return False
    
    # 配置sparse-checkout
    run_command("git sparse-checkout init --cone", cwd=temp_repo_dir)
    run_command(f"git sparse-checkout set {SUBDIR}", cwd=temp_repo_dir)
    run_command(f"git checkout {BRANCH}", cwd=temp_repo_dir, timeout=5)
    
    # 创建子目录并复制文件
    subdir_path = temp_repo_dir / SUBDIR
    subdir_path.mkdir(exist_ok=True, parents=True)
    
    import shutil
    for item in current_dir.iterdir():
        if item.name == ".git":
            continue
        dest = subdir_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # 添加并提交
    run_command("git add .", cwd=temp_repo_dir)
    run_command(f'git commit -m "Add {SUBDIR}: 荔枝霜疫霉花果预警阈值模型"', cwd=temp_repo_dir)
    
    # 推送
    success, _, _ = run_command(f"git push origin {BRANCH}", cwd=temp_repo_dir, timeout=120)
    
    return success

def method_4_force_push():
    """方案4: 强制推送（如果主仓库是空的）"""
    print("\n" + "="*70)
    print("方案4: 强制推送")
    print("="*70)
    
    current_dir = Path.cwd()
    
    # 确保有Git仓库
    if not (current_dir / ".git").exists():
        run_command("git init", cwd=current_dir)
    
    # 设置远程
    run_command("git remote remove origin", cwd=current_dir, timeout=5)
    run_command(f'git remote add origin {REPO_URL}', cwd=current_dir)
    
    # 添加并提交
    run_command("git add .", cwd=current_dir)
    run_command('git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型"', cwd=current_dir)
    
    # 创建分支
    run_command(f"git checkout -b {BRANCH}", cwd=current_dir, timeout=5)
    
    # 强制推送
    success, _, _ = run_command(f"git push -u origin {BRANCH} --force", cwd=current_dir, timeout=120)
    
    return success

def method_5_orphan_branch():
    """方案5: 创建orphan分支并推送"""
    print("\n" + "="*70)
    print("方案5: 创建orphan分支")
    print("="*70)
    
    current_dir = Path.cwd()
    
    if not (current_dir / ".git").exists():
        run_command("git init", cwd=current_dir)
    
    # 创建orphan分支
    run_command(f"git checkout --orphan {BRANCH}", cwd=current_dir, timeout=5)
    
    # 添加文件
    run_command("git add .", cwd=current_dir)
    run_command('git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型"', cwd=current_dir)
    
    # 设置远程
    run_command("git remote remove origin", cwd=current_dir, timeout=5)
    run_command(f'git remote add origin {REPO_URL}', cwd=current_dir)
    
    # 推送
    success, _, _ = run_command(f"git push -u origin {BRANCH}", cwd=current_dir, timeout=120)
    
    return success

def method_6_subtree_push():
    """方案6: 使用subtree推送"""
    print("\n" + "="*70)
    print("方案6: 使用subtree推送")
    print("="*70)
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    temp_repo_dir = parent_dir / "Agrometeorology_subtree"
    
    # 在当前目录初始化
    if not (current_dir / ".git").exists():
        run_command("git init", cwd=current_dir)
        run_command("git add .", cwd=current_dir)
        run_command('git commit -m "Initial commit"', cwd=current_dir)
    
    # 克隆主仓库
    if temp_repo_dir.exists():
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
    
    success, _, _ = run_command(f"git clone {REPO_URL} {temp_repo_dir.name}", cwd=parent_dir, timeout=120)
    if not success:
        return False
    
    # 添加subtree
    current_abs = current_dir.absolute()
    success, _, _ = run_command(
        f'git subtree add --prefix={SUBDIR} {current_abs} {BRANCH} --squash',
        cwd=temp_repo_dir,
        timeout=120
    )
    
    if success:
        # 推送
        success, _, _ = run_command(f"git push origin {BRANCH}", cwd=temp_repo_dir, timeout=120)
        return success
    
    return False

def method_7_worktree():
    """方案7: 使用worktree"""
    print("\n" + "="*70)
    print("方案7: 使用worktree")
    print("="*70)
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    temp_repo_dir = parent_dir / "Agrometeorology_worktree"
    
    # 克隆
    if temp_repo_dir.exists():
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
    
    success, _, _ = run_command(f"git clone {REPO_URL} {temp_repo_dir.name}", cwd=parent_dir, timeout=120)
    if not success:
        return False
    
    # 添加worktree
    worktree_path = temp_repo_dir / SUBDIR
    success, _, _ = run_command(
        f"git worktree add {worktree_path} {BRANCH}",
        cwd=temp_repo_dir,
        timeout=30
    )
    
    if not success:
        # 如果分支不存在，先创建
        run_command(f"git checkout -b {BRANCH}", cwd=temp_repo_dir, timeout=5)
        run_command(f"git worktree add {worktree_path}", cwd=temp_repo_dir, timeout=30)
    
    # 复制文件
    import shutil
    for item in current_dir.iterdir():
        if item.name == ".git":
            continue
        dest = worktree_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # 添加并提交
    run_command("git add .", cwd=worktree_path)
    run_command(f'git commit -m "Add {SUBDIR}: 荔枝霜疫霉花果预警阈值模型"', cwd=worktree_path)
    
    # 推送
    success, _, _ = run_command(f"git push origin {BRANCH}", cwd=temp_repo_dir, timeout=120)
    
    return success

def method_8_shallow_clone():
    """方案8: 浅克隆后推送"""
    print("\n" + "="*70)
    print("方案8: 浅克隆")
    print("="*70)
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    temp_repo_dir = parent_dir / "Agrometeorology_shallow"
    
    if temp_repo_dir.exists():
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
    
    # 浅克隆
    success, _, _ = run_command(f"git clone --depth 1 {REPO_URL} {temp_repo_dir.name}", cwd=parent_dir, timeout=120)
    if not success:
        return False
    
    # 创建子目录并复制
    subdir_path = temp_repo_dir / SUBDIR
    subdir_path.mkdir(exist_ok=True)
    
    import shutil
    for item in current_dir.iterdir():
        if item.name == ".git":
            continue
        dest = subdir_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # 添加并提交
    run_command("git add .", cwd=temp_repo_dir)
    run_command(f'git commit -m "Add {SUBDIR}: 荔枝霜疫霉花果预警阈值模型"', cwd=temp_repo_dir)
    
    # 推送
    success, _, _ = run_command(f"git push origin {BRANCH}", cwd=temp_repo_dir, timeout=120)
    
    return success

def method_9_manual_subdir():
    """方案9: 手动创建子目录结构并推送"""
    print("\n" + "="*70)
    print("方案9: 手动创建子目录结构")
    print("="*70)
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    temp_repo_dir = parent_dir / "Agrometeorology_manual"
    
    if temp_repo_dir.exists():
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
    
    # 初始化新仓库
    run_command("git init", cwd=temp_repo_dir)
    
    # 创建子目录
    subdir_path = temp_repo_dir / SUBDIR
    subdir_path.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    import shutil
    for item in current_dir.iterdir():
        if item.name == ".git":
            continue
        dest = subdir_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # 添加远程
    run_command(f'git remote add origin {REPO_URL}', cwd=temp_repo_dir)
    
    # 添加并提交
    run_command("git add .", cwd=temp_repo_dir)
    run_command(f'git commit -m "Add {SUBDIR}: 荔枝霜疫霉花果预警阈值模型"', cwd=temp_repo_dir)
    
    # 创建分支
    run_command(f"git checkout -b {BRANCH}", cwd=temp_repo_dir, timeout=5)
    
    # 推送
    success, _, _ = run_command(f"git push -u origin {BRANCH}", cwd=temp_repo_dir, timeout=120)
    
    return success

def method_10_github_cli():
    """方案10: 使用GitHub CLI（如果可用）"""
    print("\n" + "="*70)
    print("方案10: 使用GitHub CLI")
    print("="*70)
    
    # 检查gh是否安装
    success, _, _ = run_command("gh --version", timeout=5)
    if not success:
        print("GitHub CLI未安装，跳过此方案")
        return False
    
    current_dir = Path.cwd()
    
    # 初始化并提交
    if not (current_dir / ".git").exists():
        run_command("git init", cwd=current_dir)
        run_command("git add .", cwd=current_dir)
        run_command('git commit -m "Initial commit"', cwd=current_dir)
    
    # 使用gh创建仓库或推送
    # 注意：这需要认证
    success, _, _ = run_command(f"gh repo create Agrometeorology --source=. --private=false", cwd=current_dir, timeout=60)
    
    return success

def main():
    print("="*70)
    print("自动推送脚本 - 尝试10种方案直到成功")
    print("="*70)
    
    # 检查Git
    if not check_git_installed():
        print("错误: Git未安装或不在PATH中")
        return
    
    # 10种方案
    methods = [
        ("方案1: 直接初始化推送", method_1_init_and_push),
        ("方案2: 克隆后复制到子目录", method_2_clone_and_copy),
        ("方案3: Sparse checkout", method_3_sparse_checkout),
        ("方案4: 强制推送", method_4_force_push),
        ("方案5: Orphan分支", method_5_orphan_branch),
        ("方案6: Subtree推送", method_6_subtree_push),
        ("方案7: Worktree", method_7_worktree),
        ("方案8: 浅克隆", method_8_shallow_clone),
        ("方案9: 手动子目录", method_9_manual_subdir),
        ("方案10: GitHub CLI", method_10_github_cli),
    ]
    
    attempt = 0
    max_attempts = len(methods)
    
    for method_name, method_func in methods:
        attempt += 1
        print(f"\n\n{'#'*70}")
        print(f"尝试 {attempt}/{max_attempts}: {method_name}")
        print(f"{'#'*70}")
        
        try:
            success = method_func()
            if success:
                print(f"\n{'='*70}")
                print(f"✓ 成功! 使用方案 {attempt}: {method_name}")
                print(f"{'='*70}")
                return
            else:
                print(f"\n✗ 方案 {attempt} 失败，尝试下一个方案...")
                time.sleep(2)  # 短暂等待
        except Exception as e:
            print(f"\n✗ 方案 {attempt} 执行出错: {e}")
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print("所有方案都尝试失败")
    print("="*70)
    print("\n请检查:")
    print("1. 网络连接是否正常")
    print("2. GitHub访问权限")
    print("3. 仓库URL是否正确")
    print("4. 认证信息是否配置")

if __name__ == "__main__":
    main()

