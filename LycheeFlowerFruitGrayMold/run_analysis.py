#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行分析脚本"""
import sys
import os

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from analyze_thresholds_single import run_analysis, generate_report

# 运行分析
data_file = os.path.join(os.path.dirname(__file__), '样本数据.xlsx')
out_dir = os.path.join(os.path.dirname(__file__), 'analysis_output')

print("=" * 60)
print("开始运行分析...")
print("=" * 60)

results = run_analysis(data_file, out_dir, max_depth=4, min_samples_leaf=3)
generate_report(results, out_dir, max_depth=4, min_samples_leaf=3)

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)

