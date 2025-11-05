#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成最终对比总结报告
"""
import os
import re
import pandas as pd
from pathlib import Path


def parse_rule_file(filepath: str):
    """解析规则文件，提取规则信息"""
    rules = []
    if not os.path.exists(filepath):
        return rules
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 匹配规则模式
    pattern = r"规则(\d+):\s*覆盖样本=(\d+),\s*预测为.*?准确率=([\d.]+)"
    matches = re.findall(pattern, content)
    
    for match in matches:
        rule_num = int(match[0])
        samples = int(match[1])
        accuracy = float(match[2])
        rules.append({
            "rule_num": rule_num,
            "samples": samples,
            "accuracy": accuracy
        })
    
    return rules


def analyze_experiment(exp_dir: str, exp_name: str):
    """分析单个实验的结果"""
    result = {
        "name": exp_name,
        "dir": exp_dir,
        "feature_count": 0,
        "used_features": [],
        "top_feature": "",
        "top_importance": 0.0,
        "rules_by_level": {},
        "avg_accuracy": 0.0,
        "total_samples": 0,
        "score": 0.0  # 综合评分 = 平均准确率 × 覆盖样本比例
    }
    
    # 读取特征重要性
    importance_file = os.path.join(exp_dir, "tree_feature_importances.csv")
    if os.path.exists(importance_file):
        df_imp = pd.read_csv(importance_file, encoding="utf-8-sig")
        used_features = df_imp[df_imp["importance"] > 0]
        result["feature_count"] = len(used_features)
        result["used_features"] = used_features["feature"].tolist()
        if len(used_features) > 0:
            result["top_feature"] = used_features.iloc[0]["feature"]
            result["top_importance"] = float(used_features.iloc[0]["importance"])
    
    # 读取各预警等级的规则
    total_accuracy = 0
    total_rules = 0
    total_covered = 0
    
    for level in [0, 1, 2, 3]:
        rule_file = os.path.join(exp_dir, f"tree_rules_warning_{level}.txt")
        rules = parse_rule_file(rule_file)
        result["rules_by_level"][level] = rules
        
        if rules:
            for r in rules:
                total_accuracy += r["accuracy"] * r["samples"]
                total_covered += r["samples"]
                total_rules += 1
    
    if total_covered > 0:
        result["avg_accuracy"] = total_accuracy / total_covered
        result["total_samples"] = total_covered
        # 综合评分：准确率 × (覆盖样本数 / 57) 作为权重
        result["score"] = result["avg_accuracy"] * (total_covered / 57.0) * 100
    
    return result


def generate_final_summary():
    """生成最终对比总结报告"""
    
    base_dir = Path(".")
    
    # 分析所有实验
    experiments = []
    
    # 1. analysis_1104 (depth=3, leaf=5)
    if (base_dir / "analysis_1104").exists():
        exp = analyze_experiment("analysis_1104", "depth3_leaf5 (基准)")
        exp["max_depth"] = 3
        exp["min_samples_leaf"] = 5
        experiments.append(exp)
    
    # 2. analysis_1104_max-depth4min-samples-leaf3 (depth=4, leaf=3)
    if (base_dir / "analysis_1104_max-depth4min-samples-leaf3").exists():
        exp = analyze_experiment("analysis_1104_max-depth4min-samples-leaf3", "depth4_leaf3 (对比)")
        exp["max_depth"] = 4
        exp["min_samples_leaf"] = 3
        experiments.append(exp)
    
    # 3. analysis_1104_batch 中的10个实验
    batch_dir = base_dir / "analysis_1104_batch"
    if batch_dir.exists():
        for exp_dir in batch_dir.glob("analysis_depth*_leaf*"):
            exp_name = exp_dir.name.replace("analysis_", "")
            exp = analyze_experiment(str(exp_dir), exp_name)
            # 从名称提取参数
            match = re.match(r"depth(\d+)_leaf(\d+)", exp_name)
            if match:
                exp["max_depth"] = int(match.group(1))
                exp["min_samples_leaf"] = int(match.group(2))
            experiments.append(exp)
    
    # 按综合评分排序
    experiments.sort(key=lambda x: x["score"], reverse=True)
    
    # 生成报告
    report_lines = []
    
    report_lines.append("# 荔枝霜疫霉预警等级阈值模型 - 参数组合对比总结\n\n")
    report_lines.append("## 执行摘要\n\n")
    
    # 找出最佳组合
    best = experiments[0]
    report_lines.append(f"**最佳参数组合**: {best['name']} (max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']})\n\n")
    report_lines.append(f"- **综合评分**: {best['score']:.2f}\n")
    report_lines.append(f"- **平均准确率**: {best['avg_accuracy']*100:.1f}%\n")
    report_lines.append(f"- **使用特征数**: {best['feature_count']}\n")
    report_lines.append(f"- **最重要特征**: {best['top_feature']} (重要性={best['top_importance']:.4f})\n")
    report_lines.append(f"- **规则总数**: {sum(len(rules) for rules in best['rules_by_level'].values())}\n\n")
    
    report_lines.append("---\n\n")
    
    # 详细对比表
    report_lines.append("## 1. 所有组合综合对比\n\n")
    report_lines.append("| 排名 | 参数组合 | max_depth | min_samples_leaf | 综合评分 | 平均准确率 | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本数 |\n")
    report_lines.append("|------|---------|-----------|------------------|---------|-----------|-----------|-----------|----------|-----------|\n")
    
    for i, exp in enumerate(experiments, 1):
        total_rules = sum(len(rules) for rules in exp["rules_by_level"].values())
        report_lines.append(
            f"| {i} | {exp['name']} | {exp['max_depth']} | {exp['min_samples_leaf']} | "
            f"{exp['score']:.2f} | {exp['avg_accuracy']*100:.1f}% | {exp['feature_count']} | "
            f"{exp['top_feature']} | {total_rules} | {exp['total_samples']} |\n"
        )
    
    report_lines.append("\n---\n\n")
    
    # 详细分析每个组合
    report_lines.append("## 2. 各组合详细结果\n\n")
    
    for i, exp in enumerate(experiments, 1):
        report_lines.append(f"### 2.{i} {exp['name']} (max_depth={exp['max_depth']}, min_samples_leaf={exp['min_samples_leaf']})\n\n")
        
        report_lines.append(f"**综合评分**: {exp['score']:.2f} (排名: {i})\n\n")
        report_lines.append(f"**平均准确率**: {exp['avg_accuracy']*100:.1f}%\n\n")
        report_lines.append(f"**特征分析**:\n\n")
        report_lines.append(f"- 使用特征数: {exp['feature_count']}\n")
        report_lines.append(f"- 使用的特征: {', '.join(exp['used_features']) if exp['used_features'] else '无'}\n")
        report_lines.append(f"- 最重要特征: {exp['top_feature']} (重要性={exp['top_importance']:.4f})\n\n")
        
        report_lines.append(f"**规则统计**:\n\n")
        for level in [1, 2, 3]:
            rules = exp['rules_by_level'].get(level, [])
            if rules:
                level_name = {1: "1级（轻度）", 2: "2级（中度）", 3: "3级（重度）"}.get(level, f"{level}级")
                report_lines.append(f"- {level_name}预警: {len(rules)} 条规则\n")
                for r in rules[:2]:  # 只显示前2条
                    report_lines.append(f"  - 规则{r['rule_num']}: 覆盖样本={r['samples']}, 准确率={r['accuracy']*100:.1f}%\n")
        
        report_lines.append(f"\n**结果文件位置**: `{exp['dir']}/`\n\n")
        report_lines.append("---\n\n")
    
    # 关键发现
    report_lines.append("## 3. 关键发现\n\n")
    
    # 找出使用最多特征的组合
    max_features_exp = max(experiments, key=lambda x: x["feature_count"])
    report_lines.append(f"### 3.1 特征使用情况\n\n")
    report_lines.append(f"- **使用最多特征的组合**: {max_features_exp['name']} (使用了 {max_features_exp['feature_count']} 个特征)\n")
    report_lines.append(f"- **使用的特征**: {', '.join(max_features_exp['used_features'])}\n\n")
    
    # 分析哪些组合使用了短期特征（3天、当天）
    short_term_combos = []
    for exp in experiments:
        has_short = any("3天" in f or "当天" in f for f in exp['used_features'])
        if has_short:
            short_term_combos.append(exp['name'])
    
    if short_term_combos:
        report_lines.append(f"- **使用短期特征（3天/当天）的组合**: {', '.join(short_term_combos)}\n\n")
    else:
        report_lines.append(f"- **使用短期特征（3天/当天）的组合**: 无\n\n")
    
    # 准确率分析
    report_lines.append(f"### 3.2 准确率分析\n\n")
    all_100 = [exp for exp in experiments if exp['avg_accuracy'] >= 0.95]
    if all_100:
        report_lines.append(f"- **高准确率组合（≥95%）**: {', '.join([e['name'] for e in all_100])}\n\n")
    
    # 规则数量分析
    report_lines.append(f"### 3.3 规则复杂度分析\n\n")
    rule_counts = [(exp['name'], sum(len(rules) for rules in exp['rules_by_level'].values())) for exp in experiments]
    rule_counts.sort(key=lambda x: x[1], reverse=True)
    report_lines.append(f"- **规则最多的组合**: {rule_counts[0][0]} ({rule_counts[0][1]} 条规则)\n")
    report_lines.append(f"- **规则最少的组合**: {rule_counts[-1][0]} ({rule_counts[-1][1]} 条规则)\n\n")
    
    # 推荐
    report_lines.append("## 4. 推荐方案\n\n")
    
    report_lines.append(f"### 4.1 最佳准确率方案\n\n")
    report_lines.append(f"**推荐**: {best['name']} (max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']})\n\n")
    report_lines.append(f"**理由**:\n")
    report_lines.append(f"- 综合评分最高: {best['score']:.2f}\n")
    report_lines.append(f"- 平均准确率: {best['avg_accuracy']*100:.1f}%\n")
    report_lines.append(f"- 使用了 {best['feature_count']} 个特征，特征重要性分布合理\n\n")
    
    # 平衡方案（如果有）
    balanced = [e for e in experiments if 4 <= e['max_depth'] <= 6 and e['min_samples_leaf'] >= 3]
    if balanced:
        balanced.sort(key=lambda x: x['score'], reverse=True)
        balanced_best = balanced[0]
        if balanced_best['name'] != best['name']:
            report_lines.append(f"### 4.2 平衡方案（避免过拟合）\n\n")
            report_lines.append(f"**推荐**: {balanced_best['name']} (max_depth={balanced_best['max_depth']}, min_samples_leaf={balanced_best['min_samples_leaf']})\n\n")
            report_lines.append(f"**理由**:\n")
            report_lines.append(f"- 参数设置适中，降低过拟合风险\n")
            report_lines.append(f"- 综合评分: {balanced_best['score']:.2f}\n")
            report_lines.append(f"- 平均准确率: {balanced_best['avg_accuracy']*100:.1f}%\n\n")
    
    # 保存报告
    output_file = "1104_final组合.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    
    print(f"已生成最终对比报告: {output_file}")
    print(f"\n最佳组合: {best['name']} (评分: {best['score']:.2f})")


if __name__ == "__main__":
    generate_final_summary()

