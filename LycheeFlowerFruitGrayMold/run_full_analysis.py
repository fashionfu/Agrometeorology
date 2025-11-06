#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的分析流程：检查数据、运行分析、生成总结
"""
import sys
import os
import pandas as pd

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from analyze_thresholds_single import run_analysis, generate_report

def main():
    # 1. 检查数据文件
    print("=" * 80)
    print("步骤1: 检查数据文件")
    print("=" * 80)
    
    data_file = os.path.join(os.path.dirname(__file__), '样本数据.xlsx')
    
    try:
        df = pd.read_excel(data_file, sheet_name=0)
        print(f"✓ 数据文件读取成功")
        print(f"  - 行数: {len(df)}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"  - 列名: {list(df.columns)[:10]}...")  # 只显示前10个
        
        # 检查预警列
        if '预警' in df.columns:
            print(f"  - 预警列唯一值: {df['预警'].unique()}")
        else:
            print("  ⚠ 警告: 未找到'预警'列")
        
    except Exception as e:
        print(f"✗ 数据文件读取失败: {e}")
        return
    
    # 2. 运行分析
    print("\n" + "=" * 80)
    print("步骤2: 运行分析（皮尔逊相关、灰色关联、决策树）")
    print("=" * 80)
    
    out_dir = os.path.join(os.path.dirname(__file__), 'analysis_output')
    
    try:
        results = run_analysis(data_file, out_dir, max_depth=4, min_samples_leaf=3)
        generate_report(results, out_dir, max_depth=4, min_samples_leaf=3)
        print("\n✓ 分析完成！")
    except Exception as e:
        print(f"\n✗ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 检查结果
    print("\n" + "=" * 80)
    print("步骤3: 检查分析结果")
    print("=" * 80)
    
    result_files = [
        'tree_feature_importances.csv',
        'feature_scores_warning_1.csv',
        'feature_scores_warning_2.csv',
        'feature_scores_warning_3.csv',
        'tree_rules_warning_1.txt',
        'tree_rules_warning_2.txt',
        'tree_rules_warning_3.txt',
        '1104.md'
    ]
    
    for f in result_files:
        path = os.path.join(out_dir, f)
        if os.path.exists(path):
            print(f"  ✓ {f}")
        else:
            print(f"  ⚠ {f} (未找到)")
    
    # 4. 生成总结文档
    print("\n" + "=" * 80)
    print("步骤4: 生成总结文档")
    print("=" * 80)
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from generate_summary import generate_1106_summary
        generate_1106_summary(out_dir)
        print("\n✓ 总结文档生成完成！")
    except Exception as e:
        print(f"\n⚠ 总结文档生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("完成！结果保存在: " + out_dir)
    print("=" * 80)

if __name__ == "__main__":
    main()

