#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将根目录下的数据文件移动到metadata文件夹，并更新所有代码中的文件路径引用
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
    "scripts/merge_data_with_weather.py",
    "scripts/generate_word_report.py",
]

# 创建metadata文件夹
metadata_dir = Path("metadata")
metadata_dir.mkdir(exist_ok=True)
print(f"✓ 已创建文件夹: {metadata_dir}")

# 移动文件
moved_files = []
for filename in files_to_move:
    src = Path(filename)
    if src.exists():
        dst = metadata_dir / filename
        shutil.move(str(src), str(dst))
        moved_files.append(filename)
        print(f"✓ 已移动: {filename} -> metadata/{filename}")
    else:
        print(f"⚠ 文件不存在，跳过: {filename}")

print(f"\n共移动 {len(moved_files)} 个文件")

# 更新代码中的路径引用
print("\n开始更新代码中的文件路径...")

def update_file_paths(filepath):
    """更新文件中的路径引用"""
    if not Path(filepath).exists():
        print(f"⚠ 文件不存在，跳过: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 更新路径引用：将文件名替换为 metadata/文件名
    for filename in moved_files:
        # 匹配模式1: default="文件名"
        pattern1 = rf'default=["\']({re.escape(filename)})["\']'
        replacement1 = rf'default="metadata/\1"'
        content = re.sub(pattern1, replacement1, content)
        
        # 匹配模式2: default='文件名'
        pattern2 = rf"default=['\"]({re.escape(filename)})['\"]"
        replacement2 = rf"default='metadata/\1'"
        content = re.sub(pattern2, replacement2, content)
        
        # 匹配模式3: "文件名" 或 '文件名' (在字符串中)
        pattern3 = rf'["\']({re.escape(filename)})["\']'
        replacement3 = rf'"metadata/\1"'
        content = re.sub(pattern3, replacement3, content)
        
        # 匹配模式4: 直接文件路径（不带引号的情况，在注释或字符串中）
        # 这个需要更谨慎，只替换明显是参数默认值的情况
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 已更新: {filepath}")
        return True
    else:
        print(f"  (无需更新): {filepath}")
        return False

# 更新所有脚本文件
updated_count = 0
for script_path in scripts_to_update:
    if update_file_paths(script_path):
        updated_count += 1

print(f"\n共更新 {updated_count} 个脚本文件")
print("\n完成！")
