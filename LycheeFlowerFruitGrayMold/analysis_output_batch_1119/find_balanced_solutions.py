#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查找三种推荐方案：
- 方案一：平衡方案（训练准确率87.5%，验证准确率76.9%）
- 方案二：训练准确率更高的方案
- 方案三：验证准确率更高的方案
"""
import pandas as pd
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'experiments_summary.csv')

# 读取数据
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 转换数据类型
df['总体准确率_num'] = df['总体准确率'].astype(float)
df['验证准确率_num'] = df['验证准确率'].astype(float)
df['综合评分_num'] = df['综合评分'].astype(float)
df['覆盖样本总数_num'] = df['覆盖样本总数'].astype(int)

print("=" * 100)
print("三种推荐方案筛选")
print("=" * 100)

# 方案一：平衡方案（训练准确率87.5%，验证准确率76.9%，综合评分21.0）
print("\n【方案一：平衡方案】")
balanced = df[
    (df['总体准确率_num'] >= 0.874) & (df['总体准确率_num'] <= 0.876) & 
    (df['验证准确率_num'] >= 0.769) & (df['验证准确率_num'] <= 0.77) & 
    (df['综合评分_num'] >= 20.9) & (df['综合评分_num'] <= 21.1)
].copy()

if len(balanced) > 0:
    # 选择depth6_leaf4作为代表
    solution1 = balanced[balanced['参数名称'] == 'depth6_leaf4']
    if len(solution1) == 0:
        solution1 = balanced.iloc[0:1]
    print(f"  推荐: {solution1.iloc[0]['参数名称']}")
    print(f"  参数: max_depth={solution1.iloc[0]['max_depth']}, min_samples_leaf={solution1.iloc[0]['min_samples_leaf']}")
    print(f"  综合评分: {solution1.iloc[0]['综合评分']}")
    print(f"  训练准确率: {solution1.iloc[0]['总体准确率']:.1%}")
    print(f"  验证准确率: {solution1.iloc[0]['验证准确率']:.1%}")
    print(f"  覆盖样本数: {solution1.iloc[0]['覆盖样本总数']}")
else:
    print("  未找到平衡方案")

# 方案二：训练准确率更高的方案（>87.5%，优先选择覆盖样本数多的，同时兼顾验证准确率）
print("\n【方案二：训练准确率更高的方案】")
train_high = df[
    (df['总体准确率_num'] > 0.875) & 
    (df['验证准确率_num'] >= 0.4) & 
    (df['综合评分_num'] >= 15) & 
    (df['覆盖样本总数_num'] >= 20)  # 提高覆盖样本数要求到20
].copy()

solution2 = None
if len(train_high) > 0:
    # 优先选择覆盖样本数多的，然后考虑验证准确率、训练准确率和综合评分
    train_high_sorted = train_high.sort_values(
        by=['覆盖样本总数_num', '验证准确率_num', '总体准确率_num', '综合评分_num'], 
        ascending=[False, False, False, False]
    )
    
    solution2 = train_high_sorted.iloc[0:1]
    print(f"  推荐: {solution2.iloc[0]['参数名称']}")
    print(f"  参数: max_depth={solution2.iloc[0]['max_depth']}, min_samples_leaf={solution2.iloc[0]['min_samples_leaf']}")
    print(f"  综合评分: {solution2.iloc[0]['综合评分']}")
    print(f"  训练准确率: {solution2.iloc[0]['总体准确率']:.1%}")
    print(f"  验证准确率: {solution2.iloc[0]['验证准确率']:.1%}")
    print(f"  覆盖样本数: {solution2.iloc[0]['覆盖样本总数']}")
else:
    # 如果没找到，放宽覆盖样本数要求到15
    train_high = df[
        (df['总体准确率_num'] > 0.875) & 
        (df['验证准确率_num'] >= 0.4) & 
        (df['综合评分_num'] >= 15) & 
        (df['覆盖样本总数_num'] >= 15)
    ].copy()
    
    if len(train_high) > 0:
        train_high_sorted = train_high.sort_values(
            by=['覆盖样本总数_num', '验证准确率_num', '总体准确率_num', '综合评分_num'], 
            ascending=[False, False, False, False]
        )
        
        solution2 = train_high_sorted.iloc[0:1]
        print(f"  推荐: {solution2.iloc[0]['参数名称']}")
        print(f"  参数: max_depth={solution2.iloc[0]['max_depth']}, min_samples_leaf={solution2.iloc[0]['min_samples_leaf']}")
        print(f"  综合评分: {solution2.iloc[0]['综合评分']}")
        print(f"  训练准确率: {solution2.iloc[0]['总体准确率']:.1%}")
        print(f"  验证准确率: {solution2.iloc[0]['验证准确率']:.1%}")
        print(f"  覆盖样本数: {solution2.iloc[0]['覆盖样本总数']}")
    else:
        print("  未找到训练准确率更高的方案")

# 方案三：验证准确率更高的方案（>76.9%）
print("\n【方案三：验证准确率更高的方案】")
val_high = df[
    (df['验证准确率_num'] > 0.77) & 
    (df['总体准确率_num'] >= 0.8) & 
    (df['综合评分_num'] >= 10) & 
    (df['覆盖样本总数_num'] >= 10)
].copy()

if len(val_high) > 0:
    # 按验证准确率、综合评分、训练准确率、覆盖样本数排序
    val_high_sorted = val_high.sort_values(
        by=['验证准确率_num', '综合评分_num', '总体准确率_num', '覆盖样本总数_num'], 
        ascending=[False, False, False, False]
    )
    
    solution3 = val_high_sorted.iloc[0:1]
    print(f"  推荐: {solution3.iloc[0]['参数名称']}")
    print(f"  参数: max_depth={solution3.iloc[0]['max_depth']}, min_samples_leaf={solution3.iloc[0]['min_samples_leaf']}")
    print(f"  综合评分: {solution3.iloc[0]['综合评分']}")
    print(f"  训练准确率: {solution3.iloc[0]['总体准确率']:.1%}")
    print(f"  验证准确率: {solution3.iloc[0]['验证准确率']:.1%}")
    print(f"  覆盖样本数: {solution3.iloc[0]['覆盖样本总数']}")
else:
    print("  未找到验证准确率更高的方案")

# 汇总输出
print("\n" + "=" * 100)
print("三种推荐方案汇总：")
print("=" * 100)

display_cols = ['参数名称', 'max_depth', 'min_samples_leaf', '综合评分', '总体准确率', '覆盖样本总数', '验证准确率']

if len(balanced) > 0:
    print("\n方案一（平衡方案）:")
    print(solution1[display_cols].to_string(index=False))

if solution2 is not None and len(solution2) > 0:
    print("\n方案二（训练准确率更高）:")
    print(solution2[display_cols].to_string(index=False))

if len(val_high) > 0:
    print("\n方案三（验证准确率更高）:")
    print(solution3[display_cols].to_string(index=False))

