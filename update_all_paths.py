#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历所有代码文件，查找并更新数据文件路径引用
"""

import os
import re
from pathlib import Path

# 需要查找和替换的文件名
files_to_update = {
    "影响因子1103_final.csv": "metadata/影响因子1103_final.csv",
    "影响因子1103_final.xlsx": "metadata/影响因子1103_final.xlsx",
    "样本数据.xlsx": "metadata/样本数据.xlsx",
    "张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20.xlsx": 
        "metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20.xlsx",
    "张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx": 
        "metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx",
}

# 需要扫描的文件扩展名
extensions = ['.py', '.md', '.txt']

def find_all_code_files(root_dir='.'):
    """查找所有需要检查的代码文件"""
    files = []
    for ext in extensions:
        for path in Path(root_dir).rglob(f'*{ext}'):
            # 跳过虚拟环境和缓存目录
            if any(skip in str(path) for skip in ['.venv', '__pycache__', '.git', 'node_modules']):
                continue
            files.append(path)
    return files

def update_file_paths(filepath):
    """更新文件中的路径引用"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠ 无法读取文件 {filepath}: {e}")
        return False
    
    original_content = content
    changes_made = []
    
    # 遍历所有需要替换的文件名
    for old_name, new_name in files_to_update.items():
        # 如果已经包含metadata/，跳过
        if f'metadata/{old_name}' in content:
            continue
        
        # 各种可能的匹配模式
        patterns = [
            # 直接字符串引用 "文件名" 或 '文件名'
            (rf'(["\'])({re.escape(old_name)})\1', rf'\1{re.escape(new_name)}\1'),
            # default="文件名" 或 default='文件名'
            (rf'(default=["\'])({re.escape(old_name)})(["\'])', rf'\1{re.escape(new_name)}\3'),
            # 赋值语句: variable = "文件名"
            (rf'(\w+\s*=\s*["\'])({re.escape(old_name)})(["\'])', rf'\1{re.escape(new_name)}\3'),
            # 函数参数: function("文件名")
            (rf'([(,]\s*["\'])({re.escape(old_name)})(["\']\s*[,)])', rf'\1{re.escape(new_name)}\3'),
            # r"文件名" 格式
            (rf'(r["\'])({re.escape(old_name)})(["\'])', rf'\1{re.escape(new_name)}\3'),
            # 绝对路径（包含完整路径的情况）
            (rf'(F:\\[^"\']*)({re.escape(old_name)})', rf'\1metadata/{old_name}'),
        ]
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                changes_made.append(f"  - 替换: {old_name} -> {new_name}")
    
    # 如果内容有变化，写回文件
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 已更新: {filepath}")
            for change in changes_made:
                print(change)
            return True
        except Exception as e:
            print(f"⚠ 无法写入文件 {filepath}: {e}")
            return False
    
    return False

def main():
    print("=" * 70)
    print("遍历查找并更新所有代码文件中的数据文件路径")
    print("=" * 70)
    
    # 查找所有代码文件
    print("\n正在查找所有代码文件...")
    code_files = find_all_code_files()
    print(f"找到 {len(code_files)} 个文件")
    
    # 更新文件
    print("\n开始更新文件...")
    print("-" * 70)
    
    updated_count = 0
    for filepath in code_files:
        if update_file_paths(filepath):
            updated_count += 1
    
    print("-" * 70)
    print(f"\n完成！共更新 {updated_count} 个文件")
    
    print("\n提示:")
    print("  - 请检查更新后的文件是否正确")
    print("  - 确保所有数据文件已移动到 metadata/ 文件夹")

if __name__ == "__main__":
    main()
