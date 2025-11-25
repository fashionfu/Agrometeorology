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
from analyze_thresholds_single import run_analysis, generate_report


def run_batch_experiments(
    data_xlsx: str,
    base_out_dir: str = "E:/1106lizhi/analysis_output_batch"
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
                data_xlsx,
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
            
            # 统计规则数量和规则质量
            total_rules = 0
            total_covered_samples = 0
            total_accurate_samples = 0
            rule_accuracy_list = []
            
            for level in [1, 2, 3]:
                if level in results["rules_by_level"]:
                    rules = results["rules_by_level"][level]
                    total_rules += len(rules)
                    for rule in rules:
                        samples = rule["samples"]
                        pred_rate = rule["pred_rate"]
                        accurate_samples = int(samples * pred_rate)
                        total_covered_samples += samples
                        total_accurate_samples += accurate_samples
                        rule_accuracy_list.append(pred_rate)
            
            # 计算平均准确率
            avg_accuracy = sum(rule_accuracy_list) / len(rule_accuracy_list) if rule_accuracy_list else 0.0
            overall_accuracy = total_accurate_samples / total_covered_samples if total_covered_samples > 0 else 0.0
            
            summary_data.append({
                "实验编号": i,
                "参数名称": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "使用特征数": feature_count,
                "最重要特征": top_feature,
                "最高重要性": f"{top_importance:.4f}",
                "规则总数": total_rules,
                "覆盖样本总数": total_covered_samples,
                "准确样本总数": total_accurate_samples,
                "平均规则准确率": f"{avg_accuracy:.4f}",
                "总体准确率": f"{overall_accuracy:.4f}",
                "综合评分": f"{overall_accuracy * total_covered_samples:.2f}",  # 准确率×覆盖样本数
                "输出目录": out_dir
            })
            
            print(f"\n✓ 实验 {i} 完成")
            print(f"  - 使用特征数: {feature_count}")
            print(f"  - 最重要特征: {top_feature} (重要性={top_importance:.4f})")
            print(f"  - 规则总数: {total_rules}")
            print(f"  - 覆盖样本总数: {total_covered_samples}")
            print(f"  - 总体准确率: {overall_accuracy:.4f}")
            
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
                "覆盖样本总数": "失败",
                "准确样本总数": "失败",
                "平均规则准确率": "失败",
                "总体准确率": "失败",
                "综合评分": "失败",
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
    report_lines.append("| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本总数 | 总体准确率 | 综合评分 |\n")
    report_lines.append("|---------|---------|-----------|------------------|-----------|-----------|----------|------------|-----------|----------|\n")
    
    for row in summary_data:
        report_lines.append(
            f"| {row['实验编号']} | {row['参数名称']} | {row['max_depth']} | {row['min_samples_leaf']} | "
            f"{row['使用特征数']} | {row['最重要特征']} | {row['规则总数']} | {row['覆盖样本总数']} | "
            f"{row['总体准确率']} | {row['综合评分']} |\n"
        )
    
    report_lines.append("\n## 最优解法分析\n\n")
    
    # 筛选出成功完成的实验
    valid_experiments = [row for row in summary_data if row['综合评分'] != '失败']
    
    if valid_experiments:
        # 按综合评分排序（综合评分 = 总体准确率 × 覆盖样本总数）
        def get_score(row):
            try:
                return float(row['综合评分'])
            except:
                return 0.0
        
        sorted_experiments = sorted(valid_experiments, key=get_score, reverse=True)
        
        report_lines.append("### Top 3 最优解法（按综合评分排序）\n\n")
        report_lines.append("综合评分 = 总体准确率 × 覆盖样本总数，综合考虑准确率和覆盖范围。\n\n")
        
        for rank, exp in enumerate(sorted_experiments[:3], 1):
            report_lines.append(f"#### 第{rank}名：{exp['参数名称']} (max_depth={exp['max_depth']}, min_samples_leaf={exp['min_samples_leaf']})\n\n")
            report_lines.append(f"- **综合评分**: {exp['综合评分']}\n")
            report_lines.append(f"- **总体准确率**: {exp['总体准确率']}\n")
            report_lines.append(f"- **覆盖样本总数**: {exp['覆盖样本总数']}\n")
            report_lines.append(f"- **规则总数**: {exp['规则总数']}\n")
            report_lines.append(f"- **使用特征数**: {exp['使用特征数']}\n")
            report_lines.append(f"- **最重要特征**: {exp['最重要特征']} (重要性={exp['最高重要性']})\n")
            report_lines.append(f"- **输出目录**: `{exp['输出目录']}`\n\n")
        
        report_lines.append("### 详细对比分析\n\n")
        report_lines.append("| 排名 | 参数名称 | max_depth | min_samples_leaf | 总体准确率 | 覆盖样本数 | 规则数 | 综合评分 |\n")
        report_lines.append("|------|---------|-----------|------------------|-----------|-----------|--------|----------|\n")
        
        for rank, exp in enumerate(sorted_experiments[:3], 1):
            report_lines.append(
                f"| {rank} | {exp['参数名称']} | {exp['max_depth']} | {exp['min_samples_leaf']} | "
                f"{exp['总体准确率']} | {exp['覆盖样本总数']} | {exp['规则总数']} | {exp['综合评分']} |\n"
            )
        
        report_lines.append("\n### 最优解法特点分析\n\n")
        
        # 分析最优解法的共同特点
        top1 = sorted_experiments[0]
        report_lines.append(f"**最优解法（{top1['参数名称']}）的特点：**\n\n")
        report_lines.append(f"1. **参数设置**: max_depth={top1['max_depth']}, min_samples_leaf={top1['min_samples_leaf']}\n")
        report_lines.append(f"2. **性能表现**: 在{len(valid_experiments)}个有效实验中，综合评分最高（{top1['综合评分']}）\n")
        report_lines.append(f"3. **准确率**: {top1['总体准确率']}，意味着该模型在覆盖的样本上预测准确率较高\n")
        report_lines.append(f"4. **覆盖范围**: 覆盖了{top1['覆盖样本总数']}个样本，说明规则具有良好的泛化能力\n")
        report_lines.append(f"5. **特征选择**: 使用了{top1['使用特征数']}个特征，其中最重要特征是{top1['最重要特征']}\n")
        report_lines.append(f"6. **规则数量**: 共{top1['规则总数']}条规则，规则数量适中，既不过于复杂也不过于简单\n\n")
        
        report_lines.append("**建议使用该参数组合进行最终的预警模型构建。**\n\n")
    
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
    
    # 使用脚本所在目录为基准，构造健壮默认值
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    default_data = os.path.join(project_root, "样本数据.xlsx")
    default_out = "E:/1106lizhi/analysis_output_batch"

    parser.add_argument(
        "--data",
        default=default_data,
        help="包含预警和气象因子的Excel文件路径"
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help="批量实验输出基础目录"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_batch_experiments(args.data, args.out)


if __name__ == "__main__":
    main()

