#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成批量实验结果的统计报告
"""
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


def parse_rule_file(filepath: str) -> List[Dict]:
    """解析规则文件"""
    rules = []
    if not os.path.exists(filepath):
        return rules
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配规则
    pattern = r'规则(\d+):\s*覆盖样本=(\d+),\s*预测为.*?=(\d+),\s*准确率=([\d.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        rule_num, samples, pred_count, accuracy = match
        # 提取条件
        condition_match = re.search(r'条件:\s*(.+?)(?:\n|$)', content[content.find(f'规则{rule_num}'):])
        condition = condition_match.group(1).strip() if condition_match else ""
        
        # 提取类别分布
        dist_match = re.search(r'类别分布:\s*(\{.*?\})', content[content.find(f'规则{rule_num}'):])
        class_dist = dist_match.group(1) if dist_match else ""
        
        rules.append({
            'rule_num': int(rule_num),
            'samples': int(samples),
            'pred_count': int(pred_count),
            'accuracy': float(accuracy),
            'condition': condition,
            'class_dist': class_dist,
            'score': int(samples) * float(accuracy)
        })
    
    return rules


def calculate_accuracy_and_score(rules_by_level: Dict[int, List[Dict]]) -> Tuple[float, float]:
    """计算平均准确率和综合评分"""
    total_accuracy = 0.0
    total_score = 0.0
    rule_count = 0
    
    for level, rules in rules_by_level.items():
        if level == 0:  # 跳过0级
            continue
        for rule in rules:
            total_accuracy += rule['accuracy']
            total_score += rule['score']
            rule_count += 1
    
    avg_accuracy = (total_accuracy / rule_count * 100) if rule_count > 0 else 0.0
    total_score = total_score
    
    return avg_accuracy, total_score


def extract_features_from_conditions(rules: List[Dict]) -> List[str]:
    """从规则条件中提取使用的特征"""
    features = set()
    for rule in rules:
        condition = rule['condition']
        # 提取特征名（如"10天平均相对湿度"、"当天平均气温"等）
        pattern = r'([\u4e00-\u9fa5\d]+(?:天|当天)?(?:平均|累积)?(?:相对湿度|气温|雨量|降雨时数|日照时数))'
        matches = re.findall(pattern, condition)
        features.update(matches)
    return sorted(list(features))


def generate_batch_summary(batch_dir: str = "analysis_1104_batch") -> Dict:
    """生成批量实验汇总"""
    
    # 读取实验汇总
    summary_file = os.path.join(batch_dir, "experiments_summary.csv")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"未找到汇总文件: {summary_file}")
    
    df_summary = pd.read_csv(summary_file, encoding="utf-8-sig")
    
    # 推荐的三个方案（根据之前的分析）
    recommended_experiments = [
        {"name": "depth4_leaf2", "depth": 4, "leaf": 2},
        {"name": "depth4_leaf3", "depth": 4, "leaf": 3},
        {"name": "depth7_leaf2", "depth": 7, "leaf": 2},
    ]
    
    results = {}
    
    for exp in recommended_experiments:
        exp_name = exp["name"]
        exp_dir = os.path.join(batch_dir, f"analysis_{exp_name}")
        
        if not os.path.exists(exp_dir):
            continue
        
        # 读取规则文件
        rules_by_level = {}
        for level in [0, 1, 2, 3]:
            rule_file = os.path.join(exp_dir, f"tree_rules_warning_{level}.txt")
            rules_by_level[level] = parse_rule_file(rule_file)
        
        # 计算准确率和评分
        avg_accuracy, total_score = calculate_accuracy_and_score(rules_by_level)
        
        # 读取特征评分文件
        feature_scores = {}
        for level in [1, 2, 3]:
            score_file = os.path.join(exp_dir, f"feature_scores_warning_{level}.csv")
            if os.path.exists(score_file):
                feature_scores[level] = pd.read_csv(score_file, encoding="utf-8-sig")
        
        # 读取特征重要性
        importance_file = os.path.join(exp_dir, "tree_feature_importances.csv")
        feature_importance = None
        if os.path.exists(importance_file):
            feature_importance = pd.read_csv(importance_file, encoding="utf-8-sig")
        
        # 提取使用的特征
        used_features = []
        for level in [1, 2, 3]:
            if level in rules_by_level:
                features = extract_features_from_conditions(rules_by_level[level])
                used_features.extend(features)
        used_features = sorted(list(set(used_features)))
        
        results[exp_name] = {
            "depth": exp["depth"],
            "leaf": exp["leaf"],
            "rules_by_level": rules_by_level,
            "avg_accuracy": avg_accuracy,
            "total_score": total_score,
            "feature_scores": feature_scores,
            "feature_importance": feature_importance,
            "used_features": used_features
        }
    
    return results


def generate_markdown_report(results: Dict, output_file: str):
    """生成Markdown格式的统计报告"""
    
    lines = []
    
    # 标题
    lines.append("# 荔枝霜疫霉预警等级阈值模型分析\n\n")
    
    # 数据说明
    lines.append("## 数据说明\n\n")
    lines.append("- **气象因子数据**: 影响因子1103_final.xlsx（已通过回归分析筛选的关键气象因子）\n\n")
    lines.append("- **感染率数据**: 张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20.xlsx\n\n")
    lines.append("- **目标变量**: 预警列（0-3级预警等级）\n\n")
    lines.append("- **样本筛选**: 排除预警为\"未定义\"的样本（当天无数据采集）\n\n")
    lines.append("- **有效样本数**: 54 条\n\n")
    
    lines.append("### 预警等级分布\n\n")
    lines.append("- 0级（不发生）: 14 条\n")
    lines.append("- 1级（轻度）: 13 条\n")
    lines.append("- 2级（中度）: 21 条\n")
    lines.append("- 3级（重度）: 9 条\n\n")
    
    # 三种推荐方案参数
    lines.append("## 三种推荐方案参数\n\n")
    lines.append("| 方案 | 参数组合 | max_depth | min_samples_leaf | 综合评分 | 平均准确率 |\n")
    lines.append("|------|---------|-----------|------------------|---------|-----------|\n")
    
    scheme_names = ["方案一", "方案二", "方案三"]
    scheme_keys = ["depth4_leaf2", "depth4_leaf3", "depth7_leaf2"]
    
    for i, (scheme_name, scheme_key) in enumerate(zip(scheme_names, scheme_keys)):
        if scheme_key in results:
            r = results[scheme_key]
            lines.append(f"| {scheme_name} | {scheme_key} | {r['depth']} | {r['leaf']} | "
                        f"{r['total_score']:.2f} | {r['avg_accuracy']:.1f}% |\n")
    
    lines.append("\n")
    
    # 1级预警规则
    lines.append("## 1级（轻度）预警规则\n\n")
    
    for i, (scheme_name, scheme_key) in enumerate(zip(scheme_names, scheme_keys)):
        if scheme_key not in results:
            continue
        
        r = results[scheme_key]
        rules = r['rules_by_level'].get(1, [])
        
        lines.append(f"### {scheme_name}：{'最佳准确率方案' if i == 0 else '平衡方案' if i == 1 else '特征丰富度方案'} ({scheme_key})\n\n")
        
        for rule in rules[:4]:  # 最多显示4条规则
            lines.append(f"**规则{rule['rule_num']}（覆盖样本={rule['samples']}，准确率={int(rule['accuracy']*100)}%）**：\n\n")
            lines.append(f"条件: {rule['condition']}\n\n")
            lines.append(f"类别分布: {rule['class_dist']}\n\n")
            lines.append(f"评分: score={rule['score']:.0f}={rule['samples']}*{rule['accuracy']:.2f}\n\n")
        
        # 最优规则
        if rules:
            best_rule = rules[0]
            lines.append(f"**最优规则**: {best_rule['condition']}（samples={best_rule['samples']}，准确率={best_rule['accuracy']:.3f}，score={best_rule['score']:.0f}）\n\n")
    
    # 1级预警规则对比总结
    lines.append("### 1级预警规则对比总结\n\n")
    lines.append("| 方案 | 规则数 | 最优规则准确率 | 最优规则覆盖样本 | 使用的关键特征 |\n")
    lines.append("|------|--------|--------------|----------------|--------------|\n")
    
    for scheme_name, scheme_key in zip(scheme_names, scheme_keys):
        if scheme_key not in results:
            continue
        r = results[scheme_key]
        rules = r['rules_by_level'].get(1, [])
        if rules:
            best_rule = rules[0]
            features_str = "、".join(r['used_features'][:5])  # 最多显示5个特征
            lines.append(f"| {scheme_name} | {len(rules)} | {int(best_rule['accuracy']*100)}% | {best_rule['samples']} | {features_str} |\n")
    
    lines.append("\n")
    
    # 找出共同的最优规则
    common_rules = {}
    for scheme_key in scheme_keys:
        if scheme_key in results:
            rules = results[scheme_key]['rules_by_level'].get(1, [])
            if rules:
                best_condition = rules[0]['condition']
                if best_condition not in common_rules:
                    common_rules[best_condition] = []
                common_rules[best_condition].append(scheme_key)
    
    if common_rules:
        most_common = max(common_rules.items(), key=lambda x: len(x[1]))
        if len(most_common[1]) >= 2:
            lines.append(f"**共同最优规则**: {most_common[0]}\n\n")
    
    lines.append("**解读**: 当10天累积雨量较低（≤19.5mm），当天相对湿度较低（≤74.5%）时，发生轻度预警（1级）。\n\n")
    lines.append("**关键气象条件**:\n\n")
    lines.append("- 10天累积雨量 ≤ 19.5mm\n")
    lines.append("- 当天相对湿度 ≤ 74.5%\n\n")
    
    # 2级预警规则
    lines.append("## 2级（中度）预警规则\n\n")
    
    for i, (scheme_name, scheme_key) in enumerate(zip(scheme_names, scheme_keys)):
        if scheme_key not in results:
            continue
        
        r = results[scheme_key]
        rules = r['rules_by_level'].get(2, [])
        
        lines.append(f"### {scheme_name}：{'最佳准确率方案' if i == 0 else '平衡方案' if i == 1 else '特征丰富度方案'} ({scheme_key})\n\n")
        
        for rule in rules[:3]:  # 最多显示3条规则
            lines.append(f"**规则{rule['rule_num']}（覆盖样本={rule['samples']}，准确率={int(rule['accuracy']*100)}%）**：\n\n")
            lines.append(f"条件: {rule['condition']}\n\n")
            lines.append(f"类别分布: {rule['class_dist']}\n\n")
            lines.append(f"评分: score={rule['score']:.0f}\n\n")
        
        # 最优规则
        if rules:
            best_rule = rules[0]
            lines.append(f"**最优规则**: {best_rule['condition']}（samples={best_rule['samples']}，准确率={best_rule['accuracy']:.3f}，score={best_rule['score']:.0f}）\n\n")
    
    # 2级预警规则对比总结
    lines.append("### 2级预警规则对比总结\n\n")
    lines.append("| 方案 | 规则数 | 最优规则准确率 | 最优规则覆盖样本 | 使用的关键特征 |\n")
    lines.append("|------|--------|--------------|----------------|--------------|\n")
    
    for scheme_name, scheme_key in zip(scheme_names, scheme_keys):
        if scheme_key not in results:
            continue
        r = results[scheme_key]
        rules = r['rules_by_level'].get(2, [])
        if rules:
            best_rule = rules[0]
            features_str = "、".join(r['used_features'][:5])
            lines.append(f"| {scheme_name} | {len(rules)} | {int(best_rule['accuracy']*100)}% | {best_rule['samples']} | {features_str} |\n")
    
    lines.append("\n")
    
    # 找出2级的共同最优规则
    common_rules_2 = {}
    for scheme_key in scheme_keys:
        if scheme_key in results:
            rules = results[scheme_key]['rules_by_level'].get(2, [])
            if rules:
                best_condition = rules[0]['condition']
                if best_condition not in common_rules_2:
                    common_rules_2[best_condition] = []
                common_rules_2[best_condition].append(scheme_key)
    
    if common_rules_2:
        most_common = max(common_rules_2.items(), key=lambda x: len(x[1]))
        if len(most_common[1]) >= 2:
            lines.append(f"**核心最优规则（{len(most_common[1])}个方案）**: {most_common[0]}\n\n")
    
    lines.append("**解读**: 当10天累积雨量较高（>19.5mm），10天累积日照时数在合理范围内（>24.85且≤50.75小时）时，发生中度预警（2级）。\n\n")
    lines.append("**关键气象条件**:\n\n")
    lines.append("- 10天累积雨量 > 19.5mm\n")
    lines.append("- 10天累积日照时数 > 24.85小时 且 ≤ 50.75小时\n\n")
    
    # 3级预警规则
    lines.append("## 3级（重度）预警规则\n\n")
    
    for i, (scheme_name, scheme_key) in enumerate(zip(scheme_names, scheme_keys)):
        if scheme_key not in results:
            continue
        
        r = results[scheme_key]
        rules = r['rules_by_level'].get(3, [])
        
        lines.append(f"### {scheme_name}：{'最佳准确率方案' if i == 0 else '平衡方案' if i == 1 else '特征丰富度方案'} ({scheme_key})\n\n")
        
        for rule in rules[:3]:
            lines.append(f"**规则{rule['rule_num']}（覆盖样本={rule['samples']}，准确率={int(rule['accuracy']*100)}%）**：\n\n")
            lines.append(f"条件: {rule['condition']}\n\n")
            lines.append(f"类别分布: {rule['class_dist']}\n\n")
            lines.append(f"评分: score={rule['score']:.0f}\n\n")
        
        # 最优规则
        if rules:
            best_rule = rules[0]
            lines.append(f"**最优规则**: {best_rule['condition']}（samples={best_rule['samples']}，准确率={best_rule['accuracy']:.3f}，score={best_rule['score']:.0f}）\n\n")
    
    # 3级预警规则对比总结
    lines.append("### 3级预警规则对比总结\n\n")
    lines.append("| 方案 | 规则数 | 最优规则准确率 | 最优规则覆盖样本 | 使用的关键特征 | 短期特征使用 |\n")
    lines.append("|------|--------|--------------|----------------|--------------|------------|\n")
    
    for scheme_name, scheme_key in zip(scheme_names, scheme_keys):
        if scheme_key not in results:
            continue
        r = results[scheme_key]
        rules = r['rules_by_level'].get(3, [])
        if rules:
            best_rule = rules[0]
            features_str = "、".join(r['used_features'][:5])
            # 检查是否有短期特征（当天、3天）
            short_term_features = [f for f in r['used_features'] if '当天' in f or '3天' in f]
            short_term_str = "、".join(short_term_features[:2]) if short_term_features else "无"
            lines.append(f"| {scheme_name} | {len(rules)} | {int(best_rule['accuracy']*100)}% | "
                        f"{best_rule['samples']} | {features_str} | {short_term_str} |\n")
    
    lines.append("\n")
    
    # 找出3级的共同核心条件
    lines.append("**核心条件**: 三种方案都包含：10天累积雨量 > 19.500 且 10天累积日照时数 ≤ 24.850 且 当天平均气温 ≤ 23.050\n\n")
    lines.append("**解读**: 当10天累积雨量较高（>19.5mm），10天累积日照时数偏少（≤24.85小时），且当天平均气温较低（≤23.05°C）时，发生重度预警（3级）。\n\n")
    lines.append("**关键气象条件**:\n\n")
    lines.append("- 10天累积雨量 > 19.5mm\n")
    lines.append("- 10天累积日照时数 ≤ 24.85小时\n")
    lines.append("- 当天平均气温 ≤ 23.05°C\n\n")
    
    # 关联强度分析
    lines.append("## 关联强度（GRA 分数，值越大关联越强）\n\n")
    
    # 使用方案一的数据
    if "depth4_leaf2" in results:
        r = results["depth4_leaf2"]
        
        for level in [1, 2, 3]:
            if level not in r['feature_scores']:
                continue
            
            df_scores = r['feature_scores'][level]
            lines.append(f"### {level}级（{'轻度' if level == 1 else '中度' if level == 2 else '重度'}）预警关联强度\n\n")
            
            # 主要关联因子（GRA > 0.7）
            high_gra = df_scores[df_scores['gra'] > 0.7].head(5)
            if len(high_gra) > 0:
                lines.append("#### 主要关联因子（GRA分数 > 0.7）\n\n")
                lines.append("| 因子 | GRA分数 | Pearson相关系数 | 说明 |\n")
                lines.append("|------|---------|---------------|------|\n")
                for _, row in high_gra.iterrows():
                    pearson = row['pearson']
                    note = ""
                    if '日照' in row['feature'] and pearson < 0:
                        note = "负相关（日照少更易发生）"
                    elif '降雨' in row['feature'] or '雨量' in row['feature']:
                        note = "正相关"
                    elif '气温' in row['feature'] and pearson < 0:
                        note = "负相关（气温低更易发生）"
                    else:
                        note = "正相关" if pearson > 0 else "负相关"
                    lines.append(f"| {row['feature']} | {row['gra']:.3f} | {pearson:.3f} | {note} |\n")
                lines.append("\n")
            
            # 次要关联因子（GRA 0.6-0.7）
            mid_gra = df_scores[(df_scores['gra'] >= 0.6) & (df_scores['gra'] <= 0.7)].head(5)
            if len(mid_gra) > 0:
                lines.append("#### 次要关联因子（GRA分数 0.6-0.7）\n\n")
                lines.append("| 因子 | GRA分数 | Pearson相关系数 | 说明 |\n")
                lines.append("|------|---------|---------------|------|\n")
                for _, row in mid_gra.iterrows():
                    pearson = row['pearson']
                    note = ""
                    if '日照' in row['feature'] and pearson < 0:
                        note = "负相关（日照少更易发生）"
                    elif '降雨' in row['feature'] or '雨量' in row['feature']:
                        note = "正相关"
                    elif '气温' in row['feature'] and pearson < 0:
                        note = "负相关（气温低更易发生）"
                    else:
                        note = "正相关" if pearson > 0 else "负相关"
                    lines.append(f"| {row['feature']} | {row['gra']:.3f} | {pearson:.3f} | {note} |\n")
                lines.append("\n")
    
    # 方法概述
    lines.append("## 方法概述\n\n")
    lines.append("**皮尔逊相关（点二列相关）**：对每个预警等级（1,2,3）分别进行二分类分析，衡量特征与该预警等级的相关强度。\n\n")
    lines.append("**灰色关联分析 GRA（分辨系数 ρ=0.5）**：衡量特征与目标序列\"形状接近度\"，对非线性/不同量纲更稳健。\n\n")
    lines.append("**决策树多分类**：使用预警等级（0-3级）作为目标变量，从多气象因子中提取每个预警等级的阈值组合规则。\n\n")
    lines.append("### 三种方案参数对比\n\n")
    lines.append("| 方案 | max_depth | min_samples_leaf | 特点 |\n")
    lines.append("|------|-----------|------------------|------|\n")
    
    for scheme_name, scheme_key in zip(scheme_names, scheme_keys):
        if scheme_key in results:
            r = results[scheme_key]
            if scheme_name == "方案一":
                features_note = "综合评分和准确率最高，使用温度和雨量特征"
            elif scheme_name == "方案二":
                features_note = "参数更保守，降低过拟合风险"
            else:
                features_note = "深度较深，使用最多特征"
            lines.append(f"| {scheme_name} | {r['depth']} | {r['leaf']} | {features_note} |\n")
    
    lines.append("\n")
    
    # 参考文献
    lines.append("## 参考文献\n\n")
    lines.append("[1]吕安瑞,严梦荧,张妍,等.基于土壤中霜疫霉活力变化防治荔枝霜疫病初探[C]//中国植物病理学会.中国植物病理学会2024年学术年会论文集.华南农业大学园艺学院华南农业大学植物保护学院;,2024:167.DOI:10.26914/c.cnkihy.2024.022408.\n\n")
    lines.append("结果表明，荔枝霜疫霉可在果园土壤中存活16个月，成功越夏、越冬，成为荔枝霜疫病初侵染源；土壤中荔枝霜疫霉存在2个活跃期，第一个是冬末初春（12月中旬至次年开花前），此时土壤中荔枝霜疫霉活力逐渐增强，处于催醒状态，作为初侵染源具有侵染能力；第二个活跃期3月下旬至4月上中旬，为荔枝开花后至第二次生理落果期，此时土壤中复苏的荔枝霜疫霉在雨水的作用下，侵染落地花果，以多种生物学形态习居在土壤中；健康荔枝花果落地后染病快速，花落地后3h即可发病，24h后感染率超50％；幼果发病略慢，接触土壤45h感染率大于50％。落花和落果是土壤中霜疫霉初次侵染和再侵染的寄主，导致荔枝霜疫霉辗转传播侵染。\n\n")
    
    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"已生成Markdown报告: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成批量实验结果统计报告")
    parser.add_argument(
        "--batch-dir",
        default="analysis_1104_batch",
        help="批量实验输出目录"
    )
    parser.add_argument(
        "--output",
        default="analysis_1104_batch/批量实验结果统计.md",
        help="输出报告文件路径"
    )
    args = parser.parse_args()
    
    print("正在读取批量实验结果...")
    results = generate_batch_summary(args.batch_dir)
    
    print("正在生成统计报告...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_markdown_report(results, args.output)
    
    print(f"\n报告已生成: {args.output}")


if __name__ == "__main__":
    main()


