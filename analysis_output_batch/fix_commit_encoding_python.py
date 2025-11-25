#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Python 修复 Git 提交信息的中文乱码问题
这是最可靠的方法
"""
import os
import subprocess
import sys

def fix_commit_message():
    """修复提交信息"""
    repo_path = r"E:\temp_agrometeo_repo"
    
    # 如果仓库不存在，先克隆
    if not os.path.exists(repo_path):
        print("克隆仓库...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/fashionfu/Agrometeorology.git",
            repo_path
        ], check=True)
    
    os.chdir(repo_path)
    
    # 创建 UTF-8 编码的提交信息文件（无 BOM）
    commit_message = "更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)"
    commit_msg_file = os.path.join(repo_path, "commit_msg_utf8.txt")
    
    # 使用 UTF-8 编码保存（无 BOM）
    with open(commit_msg_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(commit_message)
    
    print(f"提交信息文件已创建: {commit_msg_file}")
    print(f"提交信息内容: {commit_message}")
    print("")
    
    # 修改提交信息
    print("正在修改提交信息...")
    try:
        subprocess.run([
            "git", "commit", "--amend", "-F", commit_msg_file
        ], check=True, encoding='utf-8')
        
        # 验证提交信息
        print("\n验证提交信息...")
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        
        print(f"当前提交信息: {result.stdout}")
        print("")
        
        # 自动推送（如果需要交互，可以取消注释下面的代码）
        print("\n正在推送...")
        subprocess.run([
            "git", "push", "origin", "main", "--force"
        ], check=True, encoding='utf-8')
        print("\n✅ 推送完成！")
        print("请在 GitHub 网页上查看提交信息：")
        print("https://github.com/fashionfu/Agrometeorology/commits/main")
        
        # 如果需要交互式，取消注释下面的代码，并注释上面的自动推送代码
        # response = input("是否推送到远程仓库？(y/n): ").strip().lower()
        # if response == 'y':
        #     print("\n正在推送...")
        #     subprocess.run([
        #         "git", "push", "origin", "main", "--force"
        #     ], check=True, encoding='utf-8')
        #     print("\n✅ 推送完成！")
        #     print("请在 GitHub 网页上查看提交信息：")
        #     print("https://github.com/fashionfu/Agrometeorology/commits/main")
        # else:
        #     print("\n请稍后手动推送: git push origin main --force")
        
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        sys.exit(1)
    finally:
        # 清理临时文件
        if os.path.exists(commit_msg_file):
            os.remove(commit_msg_file)
            print(f"\n已清理临时文件: {commit_msg_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("Git 提交信息中文乱码修复工具")
    print("=" * 60)
    print("")
    
    try:
        fix_commit_message()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

