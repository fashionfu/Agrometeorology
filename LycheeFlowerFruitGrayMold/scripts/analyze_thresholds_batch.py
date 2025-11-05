#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验：测试10种不同的树深度和叶子节点数组合
"""
import argparse
import os
import sys

import pandas as pd

# 导入主分析模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_thresholds_1104 import run_analysis, generate_report


def run_batch_experiments(
    factors_xlsx: str,
    infection_xlsx: str,
    base_out_dir: str = "analysis_1104_batch"
):
    """运行批量实验"""
    
    # 定义10种参数组合
    experiments = [
        {"depth": 3, "leaf": 5, "name": "depth3_leaf5"},
        {"depth": 4, "leaf": 3, "name": "depth4_leaf3"},
        {"depth": 4, "leaf": 2, "name": "depth4_leaf2"},
        {"depth": 5, "leaf": 3, "name": "depth5_leaf3"},
        {"depth": 5, "leaf": 2, "name": "depth5_leaf2"},
        {"depth": 6, "leaf": 3, "name": "depth6_leaf3"},
        {"depth": 6, "leaf": 2, "name": "depth6_leaf2"},
        {"depth": 7, "leaf": 3, "name": "depth7_leaf3"},
        {"depth": 7, "leaf": 2, "name": "depth7_leaf2"},
        {"depth": 5, "leaf": 5, "name": "depth5_leaf5"},
    ]
    
    os.makedirs(base_out_dir, exist_ok=True)
    
    # 汇总结果
    summary_data = []
    
    print("=" * 80)
    print("批量实验：测试10种树深度和叶子节点数组合")
    print("=" * 80)
    print(f"\n共 {len(experiments)} 个实验组合\n")
    
    for i, exp in enumerate(experiments, 1):
        depth = exp["depth"]
        leaf = exp["leaf"]
        name = exp["name"]
        out_dir = os.path.join(base_out_dir, f"analysis_{name}")
        
        print(f"\n{'='*80}")
        print(f"实验 {i}/{len(experiments)}: max_depth={depth}, min_samples_leaf={leaf}")
        print(f"输出目录: {out_dir}")
        print(f"{'='*80}\n")
        
        try:
            # 运行分析
            results = run_analysis(
                factors_xlsx,
                infection_xlsx,
                out_dir,
                max_depth=depth,
                min_samples_leaf=leaf
            )
            
            # 生成报告（传递实际参数）
            generate_report(results, out_dir, max_depth=depth, min_samples_leaf=leaf)
            
            # 收集特征重要性信息
            importance_file = os.path.join(out_dir, "tree_feature_importances.csv")
            if os.path.exists(importance_file):
                df_imp = pd.read_csv(importance_file, encoding="utf-8-sig")
                # 统计使用的特征数量（重要性>0）
                used_features = df_imp[df_imp["importance"] > 0]
                feature_count = len(used_features)
                top_feature = used_features.iloc[0]["feature"] if len(used_features) > 0 else "None"
                top_importance = used_features.iloc[0]["importance"] if len(used_features) > 0 else 0.0
            else:
                feature_count = 0
                top_feature = "N/A"
                top_importance = 0.0
            
            # 统计规则数量
            total_rules = 0
            for level in [1, 2, 3]:
                if level in results["rules_by_level"]:
                    total_rules += len(results["rules_by_level"][level])
            
            summary_data.append({
                "实验编号": i,
                "参数名称": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "使用特征数": feature_count,
                "最重要特征": top_feature,
                "最高重要性": f"{top_importance:.4f}",
                "规则总数": total_rules,
                "输出目录": out_dir
            })
            
            print(f"\n✓ 实验 {i} 完成")
            print(f"  - 使用特征数: {feature_count}")
            print(f"  - 最重要特征: {top_feature} (重要性={top_importance:.4f})")
            print(f"  - 规则总数: {total_rules}")
            
        except Exception as e:
            print(f"\n✗ 实验 {i} 失败: {str(e)}")
            summary_data.append({
                "实验编号": i,
                "参数名称": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "使用特征数": "失败",
                "最重要特征": "失败",
                "最高重要性": "失败",
                "规则总数": "失败",
                "输出目录": out_dir
            })
            import traceback
            traceback.print_exc()
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("生成汇总报告...")
    print(f"{'='*80}\n")
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = os.path.join(base_out_dir, "experiments_summary.csv")
    df_summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"已保存汇总报告: {summary_file}")
    
    # 生成对比报告
    report_lines = []
    report_lines.append("# 批量实验汇总报告\n\n")
    report_lines.append("## 实验参数组合\n\n")
    report_lines.append("测试了10种不同的决策树参数组合：\n\n")
    report_lines.append("| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 使用特征数 | 最重要特征 | 规则总数 |\n")
    report_lines.append("|---------|---------|-----------|------------------|-----------|-----------|----------|\n")
    
    for row in summary_data:
        report_lines.append(
            f"| {row['实验编号']} | {row['参数名称']} | {row['max_depth']} | {row['min_samples_leaf']} | "
            f"{row['使用特征数']} | {row['最重要特征']} | {row['规则总数']} |\n"
        )
    
    report_lines.append("\n## 详细结果\n\n")
    report_lines.append("每个实验的详细结果保存在对应的子文件夹中：\n\n")
    
    for exp in experiments:
        name = exp["name"]
        out_dir = f"analysis_{name}"
        report_lines.append(f"- **{exp['name']}** (depth={exp['depth']}, leaf={exp['leaf']}): `{out_dir}/`\n")
        report_lines.append(f"  - 特征重要性: `{out_dir}/tree_feature_importances.csv`\n")
        report_lines.append(f"  - 预警规则: `{out_dir}/tree_rules_warning_*.txt`\n")
        report_lines.append(f"  - 详细报告: `{out_dir}/1104.md`\n\n")
    
    report_file = os.path.join(base_out_dir, "README.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    
    print(f"已生成汇总说明: {report_file}")
    
    print(f"\n{'='*80}")
    print("批量实验完成！")
    print(f"{'='*80}\n")
    print(f"所有结果保存在: {base_out_dir}/")
    print(f"汇总报告: {base_out_dir}/experiments_summary.csv")
    print(f"说明文档: {base_out_dir}/README.md")


def main():
    parser = argparse.ArgumentParser(description="批量实验：10种树深度和叶子节点数组合")
    parser.add_argument(
        "--factors",
        default=r"F:\02_MeteorologyWork\02_正式\2025-11月正式工作\03_荔枝霜疫霉并的阈值模型\metadata/影响因子1103_final.xlsx",
        help="气象因子Excel文件路径"
    )
    parser.add_argument(
        "--infection",
        default=r"F:\02_MeteorologyWork\02_正式\2025-11月正式工作\03_荔枝霜疫霉并的阈值模型\metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20.xlsx",
        help="感染率数据Excel文件路径"
    )
    parser.add_argument(
        "--out",
        default=r"F:\02_MeteorologyWork\02_正式\2025-11月正式工作\03_荔枝霜疫霉并的阈值模型\analysis_1104_batch",
        help="批量实验输出基础目录"
    )
    args = parser.parse_args()
    
    run_batch_experiments(args.factors, args.infection, args.out)


if __name__ == "__main__":
    main()

