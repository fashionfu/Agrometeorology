#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析前三种最优方案"""
import pandas as pd

# 读取数据
df = pd.read_csv('experiments_summary.csv', encoding='utf-8-sig')

# 将综合评分转换为数值
df['综合评分_数值'] = pd.to_numeric(df['综合评分'], errors='coerce')

# 按综合评分排序
df_sorted = df.sort_values('综合评分_数值', ascending=False)

print("=" * 80)
print("前三种最优方案（按综合评分排序）")
print("=" * 80)
print()

# 输出前三种方案
top3_data = []
for i, (idx, row) in enumerate(df_sorted.head(3).iterrows(), 1):
    avg_acc = float(row['平均规则准确率']) * 100
    overall_acc = float(row['总体准确率']) * 100
    score = float(row['综合评分'])
    
    print(f"方案{i}: {row['参数名称']}")
    print(f"  参数组合: depth={row['max_depth']}, leaf={row['min_samples_leaf']}")
    print(f"  综合评分: {score:.2f}")
    print(f"  平均规则准确率: {avg_acc:.1f}%")
    print(f"  总体准确率: {overall_acc:.1f}%")
    print(f"  覆盖样本数: {row['覆盖样本总数']}")
    print(f"  规则总数: {row['规则总数']}")
    print()
    
    top3_data.append({
        '方案': f'方案{i}',
        '参数组合': row['参数名称'],
        'max_depth': int(row['max_depth']),
        'min_samples_leaf': int(row['min_samples_leaf']),
        '综合评分': score,
        '平均规则准确率': avg_acc,
        '总体准确率': overall_acc
    })

print("=" * 80)
print("表格格式：")
print("=" * 80)
print()
print("| 方案 | 参数组合 | max_depth | min_samples_leaf | 综合评分 | 平均规则准确率 | 总体准确率 |")
print("|------|---------|-----------|------------------|----------|---------------|-----------|")
for item in top3_data:
    print(f"| {item['方案']} | {item['参数组合']} | {item['max_depth']} | {item['min_samples_leaf']} | {item['综合评分']:.2f} | {item['平均规则准确率']:.1f}% | {item['总体准确率']:.1f}% |")

