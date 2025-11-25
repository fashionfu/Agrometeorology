#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新样本数据.xlsx中2024-04-21的记录为2025-04-21，并从影响因子1103_final.xlsx更新数据
"""
import pandas as pd
import os
from datetime import datetime

# 文件路径
project_root = r"F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉"
sample_file = os.path.join(project_root, "样本数据.xlsx")
factor_file = os.path.join(project_root, "影响因子1103_final.xlsx")

print("=" * 80)
print("更新2025-04-21数据")
print("=" * 80)

# 1. 读取样本数据.xlsx
print(f"\n1. 读取样本数据: {sample_file}")
df_sample = pd.read_excel(sample_file, sheet_name=0)
print(f"   原始数据: {len(df_sample)} 行, {len(df_sample.columns)} 列")

# 找到日期列
date_col = None
for col in df_sample.columns:
    if '日期' in str(col) or '时间' in str(col):
        date_col = col
        break

if date_col is None:
    print("   错误: 未找到日期列")
    exit(1)

print(f"   日期列: {date_col}")

# 转换日期
df_sample[date_col] = pd.to_datetime(df_sample[date_col], errors='coerce')

# 2. 找到2024-04-21的记录
target_date_2024 = pd.Timestamp('2024-04-21')
mask_2024 = df_sample[date_col] == target_date_2024
if mask_2024.sum() == 0:
    print(f"\n2. 未找到2024-04-21的记录")
    exit(1)

print(f"\n2. 找到 {mask_2024.sum()} 条2024-04-21的记录")
print(f"   记录索引: {df_sample[mask_2024].index.tolist()}")

# 显示原始记录
print("\n   原始记录:")
for idx in df_sample[mask_2024].index:
    row = df_sample.loc[idx]
    print(f"   索引 {idx}: 日期={row[date_col]}, 预警={row.get('预警', 'N/A')}")

# 3. 读取影响因子1103_final.xlsx
print(f"\n3. 读取影响因子数据: {factor_file}")
if not os.path.exists(factor_file):
    print(f"   错误: 文件不存在 {factor_file}")
    exit(1)

df_factor = pd.read_excel(factor_file, sheet_name=0)
print(f"   影响因子数据: {len(df_factor)} 行, {len(df_factor.columns)} 列")
print(f"   列名: {list(df_factor.columns)[:10]}...")  # 只显示前10列

# 找到影响因子文件的日期列
factor_date_col = None
for col in df_factor.columns:
    if '日期' in str(col) or '时间' in str(col):
        factor_date_col = col
        break

if factor_date_col is None:
    print("   错误: 未找到影响因子文件的日期列")
    exit(1)

print(f"   影响因子日期列: {factor_date_col}")

# 转换影响因子日期
df_factor[factor_date_col] = pd.to_datetime(df_factor[factor_date_col], errors='coerce')

# 4. 找到2025-04-21的影响因子数据
target_date_2025 = pd.Timestamp('2025-04-21')
mask_2025_factor = df_factor[factor_date_col] == target_date_2025

if mask_2025_factor.sum() == 0:
    print(f"\n4. 警告: 影响因子文件中未找到2025-04-21的数据")
    print("   将只更新日期，不更新其他数据")
    update_factor_data = False
else:
    print(f"\n4. 找到 {mask_2025_factor.sum()} 条2025-04-21的影响因子数据")
    update_factor_data = True
    factor_row = df_factor[mask_2025_factor].iloc[0]  # 取第一条
    print(f"   影响因子数据列数: {len(factor_row)}")

# 5. 更新样本数据
print(f"\n5. 更新样本数据...")
for idx in df_sample[mask_2024].index:
    # 更新日期
    df_sample.loc[idx, date_col] = target_date_2025
    print(f"   索引 {idx}: 日期已更新为 2025-04-21")
    
    # 如果找到影响因子数据，尝试更新匹配的列
    if update_factor_data:
        # 找到两个文件共同的列（排除日期列）
        common_cols = set(df_sample.columns) & set(df_factor.columns)
        common_cols = [c for c in common_cols if c != date_col and c != factor_date_col]
        
        updated_count = 0
        for col in common_cols:
            if pd.notna(factor_row[col]):
                old_val = df_sample.loc[idx, col]
                df_sample.loc[idx, col] = factor_row[col]
                if old_val != factor_row[col]:
                    updated_count += 1
                    print(f"      - {col}: {old_val} -> {factor_row[col]}")
        
        if updated_count > 0:
            print(f"   索引 {idx}: 已更新 {updated_count} 个字段")
        else:
            print(f"   索引 {idx}: 未找到可更新的字段（可能列名不匹配）")

# 6. 保存更新后的数据
print(f"\n6. 保存更新后的数据...")
# 备份原文件
backup_file = sample_file.replace('.xlsx', '_backup.xlsx')
if not os.path.exists(backup_file):
    import shutil
    shutil.copy2(sample_file, backup_file)
    print(f"   已创建备份: {backup_file}")

# 保存更新后的数据
df_sample.to_excel(sample_file, index=False, engine='openpyxl')
print(f"   已保存到: {sample_file}")

# 7. 验证更新结果
print(f"\n7. 验证更新结果...")
df_verify = pd.read_excel(sample_file, sheet_name=0)
df_verify[date_col] = pd.to_datetime(df_verify[date_col], errors='coerce')
mask_2025_verify = df_verify[date_col] == target_date_2025
print(f"   2025-04-21的记录数: {mask_2025_verify.sum()}")

if mask_2025_verify.sum() > 0:
    print("\n   更新后的记录:")
    for idx in df_verify[mask_2025_verify].index:
        row = df_verify.loc[idx]
        print(f"   索引 {idx}: 日期={row[date_col]}, 预警={row.get('预警', 'N/A')}")

print("\n" + "=" * 80)
print("更新完成！")
print("=" * 80)

