#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查样本数据文件中的列名"""
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_file = os.path.join(project_root, "样本数据_验证修复.xlsx")

print(f"读取文件: {data_file}")
df = pd.read_excel(data_file, sheet_name=0, nrows=1)

print(f"\n所有列名 ({len(df.columns)} 列):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n包含'7'或'七'的列名:")
seven_cols = [col for col in df.columns if '7' in str(col) or '七' in str(col)]
for col in seven_cols:
    print(f"  - {col}")

print("\n检查目标特征匹配:")
target_features = [
    ("7天平均气温", ["7", "平均", "气温"]),
    ("7天日均降雨量", ["7", "日均", "降雨"]),
    ("7天平均相对湿度", ["7", "平均", "相对湿度"]),
    ("7天累积降雨时数", ["7", "累积", "降雨时数"]),
    ("7天累积日照时数", ["7", "累积", "日照时数"]),
]

for target_name, keywords in target_features:
    matches = []
    for col in df.columns:
        col_str = str(col)
        if all(kw in col_str for kw in keywords):
            matches.append(col)
    if matches:
        print(f"  ✓ {target_name}: {matches}")
    else:
        print(f"  ✗ {target_name}: 未找到匹配列")

