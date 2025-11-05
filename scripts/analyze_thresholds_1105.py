#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析样本数据.xlsx，提取0-3级预警等级的气象因子阈值规则
用于汇报的气象因子阈值分析
"""

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# ---------------------- Utilities ----------------------


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """查找日期列"""
    candidates = [
        c for c in df.columns if any(tok in str(c) for tok in ["日期", "时间", "日", "date", "Date"])
    ]
    if candidates:
        return str(candidates[0])
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)
    return None


def _find_warning_column(df: pd.DataFrame) -> Optional[str]:
    """查找预警等级列"""
    warning_keywords = ["预警", "预警等级", "等级", "warning", "Warning", "level", "Level"]
    for col in df.columns:
        col_str = str(col)
        if any(keyword in col_str for keyword in warning_keywords):
            return col
    return None


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """选择气象因子列"""
    # 排除非特征列
    exclude_keywords = ["日期", "时间", "预警", "等级", "感病率", "感染率", "date", "warning"]
    
    feature_cols = []
    for col in df.columns:
        col_str = str(col)
        # 排除包含排除关键词的列
        if any(exc in col_str for exc in exclude_keywords):
            continue
        # 只保留数值列
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    return feature_cols


# ---------------------- Decision Tree Rule Extraction ----------------------


@dataclass
class Rule:
    """决策树规则"""
    path: List[str]  # 从根到叶子节点的条件路径
    samples: int  # 覆盖样本数
    accuracy: float  # 准确率（该规则下该等级的样本比例）
    class_dist: Dict[int, int]  # 类别分布
    warning_level: int  # 预警等级


def extract_rules_for_level(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    target_level: int,
) -> List[Rule]:
    """提取特定预警等级的规则"""
    t = clf.tree_
    rules: List[Rule] = []

    def recurse(node: int, path: List[str], idx: np.ndarray):
        if t.feature[node] == -2:  # 叶子节点
            samples = idx.sum()
            if samples == 0:
                return
            
            # 计算类别分布
            y_subset = y[idx]
            class_counts = {}
            for level in np.unique(y):
                class_counts[int(level)] = int(np.sum(y_subset == level))
            
            # 计算该等级的比例（准确率）
            target_count = class_counts.get(target_level, 0)
            accuracy = float(target_count) / float(samples) if samples > 0 else 0.0
            
            # 只保存准确率>=0.5的规则（即该规则主要预测该等级）
            if accuracy >= 0.5:
                rules.append(Rule(
                    path=list(path),
                    samples=int(samples),
                    accuracy=accuracy,
                    class_dist=class_counts,
                    warning_level=target_level
                ))
            return

        # 内部节点，继续递归
        feat_idx = t.feature[node]
        threshold = t.threshold[node]
        name = feature_names[feat_idx]

        left_idx = idx & (X[:, feat_idx] <= threshold)
        right_idx = idx & (X[:, feat_idx] > threshold)

        recurse(
            node=t.children_left[node],
            path=path + [f"{name} ≤ {threshold:.3f}"],
            idx=left_idx
        )
        recurse(
            node=t.children_right[node],
            path=path + [f"{name} > {threshold:.3f}"],
            idx=right_idx
        )

    recurse(0, [], np.ones(X.shape[0], dtype=bool))
    
    # 按评分排序（样本数 * 准确率）
    rules.sort(key=lambda r: r.samples * r.accuracy, reverse=True)
    return rules


# ---------------------- Main Analysis Function ----------------------


def analyze_sample_data(
    sample_file: str,
    output_dir: str,
    max_depth: int = 4,
    min_samples_leaf: int = 2,
):
    """
    分析样本数据，提取0-3级预警规则
    
    参数:
        sample_file: 样本数据Excel文件路径
        output_dir: 输出目录
        max_depth: 决策树最大深度
        min_samples_leaf: 叶子节点最小样本数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("样本数据阈值分析：提取0-3级预警等级的气象因子阈值规则")
    print("=" * 80)
    
    # 读取数据
    print(f"\n正在读取数据文件: {sample_file}")
    df = pd.read_excel(sample_file, sheet_name=0)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 查找预警等级列
    warning_col = _find_warning_column(df)
    if warning_col is None:
        raise ValueError("未找到预警等级列，请检查数据文件中是否包含'预警'相关列")
    
    print(f"\n预警等级列: {warning_col}")
    
    # 选择特征列
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("未找到气象因子列")
    
    print(f"气象因子列 ({len(feature_cols)}个): {feature_cols}")
    
    # 准备数据
    # 将预警等级转换为数字
    warning_levels = df[warning_col].astype(str)
    
    # 识别预警等级映射
    level_map = {}
    unique_levels = warning_levels.unique()
    print(f"\n发现预警等级: {unique_levels}")
    
    # 标准化预警等级名称
    for level_str in unique_levels:
        level_str = str(level_str).strip()
        if "0" in level_str or "不发生" in level_str or level_str == "0":
            level_map[level_str] = 0
        elif "1" in level_str or "轻度" in level_str:
            level_map[level_str] = 1
        elif "2" in level_str or "中度" in level_str:
            level_map[level_str] = 2
        elif "3" in level_str or "重度" in level_str:
            level_map[level_str] = 3
    
    print(f"预警等级映射: {level_map}")
    
    # 转换预警等级
    y = warning_levels.map(level_map)
    
    # 删除无效的样本
    valid_mask = ~y.isna()
    df_clean = df[valid_mask].copy()
    y_clean = y[valid_mask].values.astype(int)
    
    print(f"\n有效样本数: {len(df_clean)} 条")
    
    # 统计各等级样本数
    level_counts = {}
    level_names_display = {0: "不发生", 1: "轻度", 2: "中度", 3: "重度"}
    for level in [0, 1, 2, 3]:
        count = int(np.sum(y_clean == level))
        level_counts[level] = count
        if count > 0:
            print(f"{level}级（{level_names_display[level]}）: {count} 条")
    
    # 准备特征矩阵
    X = df_clean[feature_cols].astype(float).values
    
    # 删除包含NaN的行
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y_clean = y_clean[mask]
    df_clean = df_clean[mask].reset_index(drop=True)
    
    print(f"去除NaN后有效样本数: {len(X)} 条")
    
    # 训练决策树模型
    print(f"\n正在训练决策树模型 (max_depth={max_depth}, min_samples_leaf={min_samples_leaf})...")
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X, y_clean)
    
    print("模型训练完成")
    
    # 提取各等级的规则
    all_rules = {}
    for level in [0, 1, 2, 3]:
        print(f"\n正在提取{level}级预警规则...")
        rules = extract_rules_for_level(clf, feature_cols, X, y_clean, level)
        all_rules[level] = rules
        print(f"  找到 {len(rules)} 条规则")
    
    # 生成Markdown报告
    generate_markdown_report(
        all_rules,
        level_counts,
        feature_cols,
        output_dir,
        max_depth,
        min_samples_leaf
    )
    
    # 保存各等级的规则到文本文件
    for level, rules in all_rules.items():
        save_rules_to_file(rules, level, output_dir)
    
    # 保存特征重要性
    feature_importance_df = pd.DataFrame({
        '特征名称': feature_cols,
        '重要性': clf.feature_importances_
    }).sort_values('重要性', ascending=False)
    feature_importance_df.to_csv(
        os.path.join(output_dir, '特征重要性.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    
    print(f"\n{'='*80}")
    print(f"分析完成！所有结果已保存到: {output_dir}")
    print(f"{'='*80}")


def generate_markdown_report(
    all_rules: Dict[int, List[Rule]],
    level_counts: Dict[int, int],
    feature_cols: List[str],
    output_dir: str,
    max_depth: int,
    min_samples_leaf: int,
):
    """生成Markdown格式的汇报文档"""
    
    level_names = {
        0: "0级（不发生）",
        1: "1级（轻度）",
        2: "2级（中度）",
        3: "3级（重度）"
    }
    
    md_content = []
    md_content.append("# 荔枝霜疫霉预警等级气象因子阈值分析报告")
    md_content.append("")
    md_content.append("## 数据说明")
    md_content.append("")
    md_content.append(f"数据来源: 样本数据.xlsx")
    md_content.append("")
    total_samples = sum(level_counts.values())
    md_content.append(f"有效样本数: {total_samples} 条")
    md_content.append("")
    md_content.append("预警等级分布：")
    md_content.append("")
    for level in [0, 1, 2, 3]:
        count = level_counts.get(level, 0)
        md_content.append(f"{level_names[level]}: {count} 条")
    md_content.append("")
    md_content.append("## 模型参数")
    md_content.append("")
    md_content.append(f"- 决策树最大深度: {max_depth}")
    md_content.append(f"- 叶子节点最小样本数: {min_samples_leaf}")
    md_content.append("")
    md_content.append("## 预警等级阈值规则")
    md_content.append("")
    
    # 为每个等级生成规则
    for level in [0, 1, 2, 3]:
        rules = all_rules.get(level, [])
        if not rules:
            continue
        
        md_content.append(f"### {level_names[level]}预警规则")
        md_content.append("")
        
        for i, rule in enumerate(rules, 1):
            score = rule.samples * rule.accuracy
            
            md_content.append(f"**规则{i}（覆盖样本={rule.samples}，准确率={rule.accuracy*100:.1f}%）**：")
            md_content.append("")
            
            if rule.path:
                condition_str = " 且 ".join(rule.path)
                md_content.append(f"条件: {condition_str}")
            else:
                md_content.append("条件: (无条件，整体为该等级)")
            
            # 类别分布
            class_dist_str = ", ".join([f"{k}: {v}" for k, v in sorted(rule.class_dist.items())])
            md_content.append(f"")
            md_content.append(f"类别分布: {{{class_dist_str}}}")
            
            md_content.append(f"")
            md_content.append(f"评分: score={score:.2f}={rule.samples}*{rule.accuracy:.3f}")
            md_content.append("")
        
        # 最优规则
        if rules:
            best_rule = rules[0]
            if best_rule.path:
                condition_str = " 且 ".join(best_rule.path)
                md_content.append(f"**最优规则**: {condition_str}（samples={best_rule.samples}，准确率={best_rule.accuracy:.3f}，score={best_rule.samples*best_rule.accuracy:.2f}）")
            else:
                md_content.append(f"**最优规则**: (无条件，整体为该等级)")
            md_content.append("")
            md_content.append("")
    
    # 保存Markdown文件
    md_file = os.path.join(output_dir, '预警等级阈值规则报告.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"✓ 已生成Markdown报告: {md_file}")


def save_rules_to_file(rules: List[Rule], level: int, output_dir: str):
    """保存规则到文本文件"""
    level_names = {
        0: "0级（不发生）",
        1: "1级（轻度）",
        2: "2级（中度）",
        3: "3级（重度）"
    }
    
    filename = f"tree_rules_warning_{level}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{level_names[level]}预警规则\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{level_names[level]}规则，按评分（样本数*准确率）排序\n\n")
        
        for i, rule in enumerate(rules, 1):
            score = rule.samples * rule.accuracy
            f.write(f"规则{i}: 覆盖样本={rule.samples}, 准确率={rule.accuracy:.3f}, 评分={score:.2f}\n")
            
            if rule.path:
                f.write("条件: " + " 且 ".join(rule.path) + "\n")
            else:
                f.write("条件: (无条件，整体为该等级)\n")
            
            # 类别分布
            class_dist_str = ", ".join([f"{k}: {v}" for k, v in sorted(rule.class_dist.items())])
            f.write(f"类别分布: {{{class_dist_str}}}\n")
            f.write("\n")
    
    print(f"✓ 已保存{level_names[level]}规则: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="分析样本数据，提取0-3级预警等级的气象因子阈值规则")
    parser.add_argument(
        "--sample",
        default="metadata/样本数据.xlsx",
        help="样本数据Excel文件路径"
    )
    parser.add_argument(
        "--out",
        default="analysis_1105",
        help="输出目录"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="决策树最大深度"
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="叶子节点最小样本数"
    )
    
    args = parser.parse_args()
    
    analyze_sample_data(
        args.sample,
        args.out,
        args.max_depth,
        args.min_samples_leaf
    )


if __name__ == "__main__":
    main()
