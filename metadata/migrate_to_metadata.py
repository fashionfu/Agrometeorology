#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动将数据文件移动到metadata文件夹，并更新代码中的路径引用
"""

import os
import re
import shutil
from pathlib import Path

# 需要移动的文件（数据文件）
files_to_move = [
    "metadata/影响因子1103_final\.csv",
    "metadata/影响因子1103_final\.xlsx",
    "metadata/样本数据\.xlsx",
    "metadata/张桂香19\-25校内大果园花果带菌率数据分析\-\-给张工分析数据\-10\.20\.xlsx",
    "metadata/张桂香19\-25校内大果园花果带菌率数据分析\-\-给张工分析数据\-10\.20_预警\.xlsx",
]

# 需要更新的脚本文件
scripts_to_update = [
    "scripts/analyze_thresholds.py",
    "scripts/analyze_thresholds_1104.py",
    "scripts/analyze_thresholds_1105.py",
    "scripts/analyze_thresholds_batch.py",
    "scripts/predict_warning.py",
]

def update_file_paths(filepath):
    """更新文件中的路径引用"""
    if not Path(filepath).exists():
        print(f"⚠ 文件不存在，跳过: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 更新路径引用：将文件名替换为 metadata/文件名
    for filename in files_to_move:
        # 匹配模式1: default="文件名" 或 default='文件名'
        patterns = [
            (rf'default=["\']({re.escape(filename)})["\']', rf'default="metadata/\1"'),
            (rf'default=(["\'])({re.escape(filename)})\1', rf'default=\1metadata/\2\1'),
            # 匹配直接字符串 "文件名" 或 '文件名'（在赋值或参数中）
            (rf'(["\'"])({re.escape(filename)})\1', rf'\1metadata/\2\1'),
            # 匹配r"..."格式
            (rf'r["\']({re.escape(filename)})["\']', rf'r"metadata/\1"'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 已更新: {filepath}")
        return True
    else:
        print(f"  (无需更新): {filepath}")
        return False

def main():
    print("=" * 60)
    print("数据文件迁移到metadata文件夹")
    print("=" * 60)
    
    # 创建metadata文件夹
    metadata_dir = Path("metadata")
    metadata_dir.mkdir(exist_ok=True)
    print(f"\n✓ 已创建/确认文件夹: {metadata_dir}")
    
    # 移动文件
    print("\n" + "-" * 60)
    print("步骤1: 移动数据文件")
    print("-" * 60)
    moved_files = []
    for filename in files_to_move:
        src = Path(filename)
        if src.exists():
            dst = metadata_dir / filename
            if dst.exists():
                print(f"⚠ 文件已存在，跳过: metadata/{filename}")
            else:
                shutil.move(str(src), str(dst))
                moved_files.append(filename)
                print(f"✓ 已移动: {filename} -> metadata/{filename}")
        else:
            print(f"⚠ 文件不存在，跳过: {filename}")
    
    print(f"\n共移动 {len(moved_files)} 个文件")
    
    # 更新代码中的路径引用
    print("\n" + "-" * 60)
    print("步骤2: 更新代码中的文件路径")
    print("-" * 60)
    
    updated_count = 0
    for script_path in scripts_to_update:
        if update_file_paths(script_path):
            updated_count += 1
    
    print(f"\n共更新 {updated_count} 个脚本文件")
    
    print("\n" + "=" * 60)
    print("迁移完成！")
    print("=" * 60)
    print("\n提示:")
    print("  - 所有数据文件已移动到 metadata/ 文件夹")
    print("  - 相关脚本的默认路径已更新")
    print("  - 如果脚本运行时报错找不到文件，请检查工作目录是否为项目根目录")

if __name__ == "__main__":
    main()
