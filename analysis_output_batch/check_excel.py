#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查看生成的Excel文件内容"""
import pandas as pd

excel_file = "决策树预测结果对比表_depth7_leaf2.xlsx"

# 读取Excel文件
xls = pd.ExcelFile(excel_file)
print("=" * 80)
print("Excel文件工作表列表:")
print("=" * 80)
for i, sheet in enumerate(xls.sheet_names, 1):
    print(f"  {i}. {sheet}")

# 查看预测对比表
print("\n" + "=" * 80)
print("预测对比表概览:")
print("=" * 80)
df = pd.read_excel(excel_file, sheet_name='预测对比表')
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"\n列名:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n前5行数据:")
print(df.head().to_string())

# 查看统计汇总
print("\n" + "=" * 80)
print("统计汇总:")
print("=" * 80)
df_summary = pd.read_excel(excel_file, sheet_name='统计汇总')
print(df_summary.to_string(index=False))

# 查看错误预测详情
print("\n" + "=" * 80)
print("错误预测详情:")
print("=" * 80)
df_errors = pd.read_excel(excel_file, sheet_name='错误预测详情')
print(f"错误预测数量: {len(df_errors)}")
if len(df_errors) > 0:
    print(df_errors.to_string(index=False))

# 查看混淆矩阵
print("\n" + "=" * 80)
print("混淆矩阵:")
print("=" * 80)
df_cm = pd.read_excel(excel_file, sheet_name='混淆矩阵')
print(df_cm.to_string(index=True))

