#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验：测试10种不同的树深度和叶子节点数组合
强制使用'7天累计雨量'作为决策树的第一个判断条件
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 尝试从 scipy.stats 导入 entropy，如果没有则使用自己实现的版本
try:
    from scipy.stats import entropy
except ImportError:
    def entropy(probs, base=2):
        """计算熵（如果 scipy 不可用时的备选实现）"""
        probs = np.array(probs)
        probs = probs[probs > 0]  # 只考虑非零概率
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log(probs) / np.log(base))

# 导入主分析模块的部分函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_thresholds_single import (
    _find_date_column, _to_datetime_series, _normalize_warning_level,
    _find_warning_column, _select_feature_columns,
    grey_relational_grade, pearson_with_binary, extract_warning_rules,
    generate_report
)


def find_7day_rainfall_feature(feature_cols):
    """查找'7天累计雨量'特征"""
    candidates = []
    for col in feature_cols:
        col_str = str(col)
        # 匹配包含"7"和"累积"或"累计"或"累积雨量"或"累计雨量"的列
        if ("7" in col_str or "七" in col_str) and \
           (any(kw in col_str for kw in ["累积", "累计", "累积雨量", "累计雨量", "累积降雨", "累计降雨"])):
            candidates.append(col)
    
    if candidates:
        return candidates[0]  # 返回第一个匹配的
    return None


def find_best_split_for_feature(X, y, feature_idx):
    """为指定特征找到最佳分割点"""
    feature_values = X[:, feature_idx]
    unique_values = np.unique(feature_values[~np.isnan(feature_values)])
    
    if len(unique_values) < 2:
        return None, None, None
    
    best_threshold = None
    best_gain = -np.inf
    best_left_idx = None
    best_right_idx = None
    
    # 尝试每个可能的分割点
    for threshold in unique_values[:-1]:
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        
        # 计算信息增益
        parent_entropy = entropy_impurity(y)
        left_entropy = entropy_impurity(y[left_mask])
        right_entropy = entropy_impurity(y[right_mask])
        
        left_weight = left_mask.sum() / len(y)
        right_weight = right_mask.sum() / len(y)
        
        gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_left_idx = left_mask
            best_right_idx = right_mask
    
    return best_threshold, best_left_idx, best_right_idx


def entropy_impurity(y):
    """计算熵不纯度"""
    if len(y) == 0:
        return 0.0
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return entropy(probs, base=2)


def find_best_split_all_features(X, y, feature_names):
    """在所有特征中找到最佳分割点（标准决策树算法）"""
    best_feature_idx = None
    best_threshold = None
    best_gain = -np.inf
    best_left_idx = None
    best_right_idx = None
    
    for feature_idx in range(X.shape[1]):
        threshold, left_idx, right_idx = find_best_split_for_feature(X, y, feature_idx)
        if threshold is None:
            continue
        
        # 计算信息增益
        parent_entropy = entropy_impurity(y)
        left_entropy = entropy_impurity(y[left_idx])
        right_entropy = entropy_impurity(y[right_idx])
        
        left_weight = left_idx.sum() / len(y)
        right_weight = right_idx.sum() / len(y)
        
        gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
        
        if gain > best_gain:
            best_gain = gain
            best_feature_idx = feature_idx
            best_threshold = threshold
            best_left_idx = left_idx
            best_right_idx = right_idx
    
    return best_feature_idx, best_threshold, best_left_idx, best_right_idx


def build_tree_with_forced_root(X, y, feature_names, forced_feature_name, 
                                max_depth, min_samples_leaf, current_depth=0):
    """构建决策树，强制使用指定特征作为根节点，子节点使用标准算法"""
    # 如果达到最大深度或样本数太少，返回叶子节点
    if current_depth >= max_depth or len(y) < min_samples_leaf * 2:
        unique, counts = np.unique(y, return_counts=True)
        pred_class = unique[np.argmax(counts)]
        return {
            'type': 'leaf',
            'pred_class': int(pred_class),
            'samples': len(y),
            'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)}
        }
    
    # 根节点：强制使用指定特征
    if current_depth == 0:
        # 找到强制特征在特征列表中的索引
        forced_feature_idx = None
        for i, name in enumerate(feature_names):
            if name == forced_feature_name:
                forced_feature_idx = i
                break
        
        if forced_feature_idx is None:
            raise ValueError(f"未找到强制特征: {forced_feature_name}")
        
        # 使用强制特征进行分割
        threshold, left_idx, right_idx = find_best_split_for_feature(X, y, forced_feature_idx)
        
        if threshold is None:
            # 无法分割，返回叶子节点
            unique, counts = np.unique(y, return_counts=True)
            pred_class = unique[np.argmax(counts)]
            return {
                'type': 'leaf',
                'pred_class': int(pred_class),
                'samples': len(y),
                'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)}
            }
        
        # 递归构建左右子树（子节点使用标准算法）
        left_tree = build_tree_with_forced_root(
            X[left_idx], y[left_idx], feature_names, forced_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if left_idx.sum() >= min_samples_leaf else None
        
        right_tree = build_tree_with_forced_root(
            X[right_idx], y[right_idx], feature_names, forced_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if right_idx.sum() >= min_samples_leaf else None
        
        return {
            'type': 'split',
            'feature': forced_feature_name,
            'feature_idx': forced_feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'left_samples': int(left_idx.sum()),
            'right_samples': int(right_idx.sum())
        }
    else:
        # 子节点：使用标准决策树算法选择最佳特征
        best_feature_idx, best_threshold, best_left_idx, best_right_idx = \
            find_best_split_all_features(X, y, feature_names)
        
        if best_feature_idx is None or best_threshold is None:
            # 无法分割，返回叶子节点
            unique, counts = np.unique(y, return_counts=True)
            pred_class = unique[np.argmax(counts)]
            return {
                'type': 'leaf',
                'pred_class': int(pred_class),
                'samples': len(y),
                'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)}
            }
        
        # 递归构建左右子树
        left_tree = build_tree_with_forced_root(
            X[best_left_idx], y[best_left_idx], feature_names, forced_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if best_left_idx.sum() >= min_samples_leaf else None
        
        right_tree = build_tree_with_forced_root(
            X[best_right_idx], y[best_right_idx], feature_names, forced_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if best_right_idx.sum() >= min_samples_leaf else None
        
        return {
            'type': 'split',
            'feature': feature_names[best_feature_idx],
            'feature_idx': best_feature_idx,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'left_samples': int(best_left_idx.sum()),
            'right_samples': int(best_right_idx.sum())
        }


def extract_rules_from_custom_tree(tree, feature_names, X, y, warning_labels, path=None):
    """从自定义树中提取规则"""
    if path is None:
        path = []
    
    all_rules = {level: [] for level in warning_labels.keys()}
    
    if tree['type'] == 'leaf':
        samples = tree['samples']
        pred_class = tree['pred_class']
        class_dist = tree['class_distribution']
        
        if pred_class in warning_labels:
            pred_count = class_dist.get(pred_class, 0)
            pred_rate = pred_count / samples if samples > 0 else 0.0
            
            # 根据样本数量调整阈值
            if samples < 5:
                min_rate = 0.8
            elif samples < 10:
                min_rate = 0.7
            elif samples < 20:
                min_rate = 0.6
            else:
                min_rate = 0.5
            
            if pred_rate >= min_rate:
                all_rules[pred_class].append({
                    "path": list(path),
                    "samples": samples,
                    "pred_rate": pred_rate,
                    "pred_count": pred_count,
                    "class_distribution": class_dist
                })
    else:
        # 分割节点
        feature_name = tree['feature']
        threshold = tree['threshold']
        
        # 左子树（<= threshold）
        if tree['left'] is not None:
            left_path = path + [f"{feature_name} ≤ {threshold:.3f}"]
            left_rules = extract_rules_from_custom_tree(
                tree['left'], feature_names, X, y, warning_labels, left_path
            )
            for level in left_rules:
                all_rules[level].extend(left_rules[level])
        
        # 右子树（> threshold）
        if tree['right'] is not None:
            right_path = path + [f"{feature_name} > {threshold:.3f}"]
            right_rules = extract_rules_from_custom_tree(
                tree['right'], feature_names, X, y, warning_labels, right_path
            )
            for level in right_rules:
                all_rules[level].extend(right_rules[level])
    
    return all_rules


def run_analysis_with_forced_feature(data_xlsx: str, out_dir: str, 
                                     max_depth: int = 3, min_samples_leaf: int = 5,
                                     forced_feature_name: str = None):
    """运行完整分析流程，强制使用指定特征作为根节点"""
    os.makedirs(out_dir, exist_ok=True)
    
    print("正在读取数据文件...")
    df = pd.read_excel(data_xlsx, sheet_name=0)
    
    print(f"数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 识别日期列
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError("无法识别日期列，请检查数据文件的日期列名称")
    
    df[date_col] = _to_datetime_series(df[date_col])
    date_na = df[date_col].isna().sum()
    if date_na > 0:
        df = df.dropna(subset=[date_col])
    
    # 识别预警列
    warning_col = _find_warning_column(df)
    if not warning_col:
        raise ValueError("未找到预警列，请确认数据文件中包含'预警'列")
    
    # 排除预警为"未定义"的样本
    df = df[df[warning_col].astype(str).str.strip() != "未定义"]
    
    # 转换预警值为数值
    df["预警数值"] = df[warning_col].map(_normalize_warning_level)
    df = df.dropna(subset=["预警数值"])
    
    # 统计预警等级分布
    warning_counts = df["预警数值"].value_counts().sort_index()
    warning_labels = {0: "0级（不发生）", 1: "1级（轻度）", 2: "2级（中度）", 3: "3级（重度）"}
    
    # 选择特征列
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError(f"未匹配到气象因子列。可用列: {list(df.columns)}")
    
    print(f"\n识别到 {len(feature_cols)} 个气象因子特征")
    
    # 查找强制使用的特征
    if forced_feature_name is None:
        forced_feature_name = find_7day_rainfall_feature(feature_cols)
    
    if forced_feature_name is None:
        raise ValueError("未找到'7天累计雨量'特征，请检查数据文件")
    
    print(f"\n强制使用特征作为根节点: {forced_feature_name}")
    
    # 检查异常值
    for col in feature_cols:
        if col in df.columns:
            if "相对湿度" in col or "湿度" in col:
                df[col] = df[col].replace([-9999, -9999.0, -9999.00], np.nan)
                df[col] = df[col].clip(lower=0, upper=100)
            elif "气温" in col or "温度" in col:
                df[col] = df[col].clip(lower=-50, upper=50)
            elif "雨量" in col or "降雨" in col:
                df[col] = df[col].clip(lower=0, upper=1000)
            elif "日照" in col or "时数" in col:
                df[col] = df[col].clip(lower=0, upper=None)
    
    # 准备特征和标签
    X = df[feature_cols].astype(float).to_numpy()
    y_warning = df["预警数值"].astype(int).to_numpy()
    
    # 删除特征中有NaN的行
    mask_valid = ~np.isnan(X).any(axis=1)
    Xv = X[mask_valid]
    yw = y_warning[mask_valid]
    
    print(f"有效样本（无缺失特征）: {mask_valid.sum()} / {len(mask_valid)}")
    
    # 1. 特征筛选：对每个预警等级分别进行关联分析
    print("\n=== 1. 特征筛选分析 ===")
    
    feature_scores = {}
    for level in [1, 2, 3]:
        y_binary = (yw == level).astype(int)
        if y_binary.sum() == 0:
            continue
        pear = pearson_with_binary(Xv, y_binary)
        gra = grey_relational_grade(Xv, y_binary.astype(float), rho=0.5)
        feature_scores[level] = {
            "pearson": pear,
            "gra": gra
        }
    
    # 保存特征评分
    for level in feature_scores:
        df_score = pd.DataFrame({
            "feature": feature_cols,
            "pearson": feature_scores[level]["pearson"],
            "gra": feature_scores[level]["gra"],
        })
        df_score["综合评分"] = np.abs(df_score["pearson"]) * 0.5 + df_score["gra"] * 0.5
        df_score = df_score.sort_values(by=["综合评分"], ascending=False)
        
        df_score.to_csv(
            os.path.join(out_dir, f"feature_scores_warning_{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        print(f"已保存预警{level}级特征评分: feature_scores_warning_{level}.csv")
    
    # 2. 决策树阈值分析（强制使用指定特征作为根节点）
    print(f"\n=== 2. 决策树阈值分析（强制使用'{forced_feature_name}'作为根节点）===")
    
    # 构建自定义决策树
    custom_tree = build_tree_with_forced_root(
        Xv, yw, feature_cols, forced_feature_name,
        max_depth, min_samples_leaf
    )
    
    # 提取规则
    all_rules = extract_rules_from_custom_tree(
        custom_tree, feature_cols, Xv, yw, warning_labels
    )
    
    # 对每个等级的规则排序
    for level in all_rules:
        all_rules[level].sort(key=lambda r: (r["pred_rate"], r["samples"]), reverse=True)
    
    # 保存规则
    def save_warning_rules(rules_by_level: dict, out_dir: str):
        for level, rules in rules_by_level.items():
            if not rules:
                continue
            label = warning_labels.get(level, f"{level}级")
            path = os.path.join(out_dir, f"tree_rules_warning_{level}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{label}预警规则，按预测准确率与覆盖样本排序\n")
                f.write(f"（强制使用'{forced_feature_name}'作为根节点）\n\n")
                for i, r in enumerate(rules, 1):
                    f.write(f"规则{i}: 覆盖样本={r['samples']}, 预测为{label}={r['pred_count']}, 准确率={r['pred_rate']:.3f}\n")
                    f.write(f"  类别分布: {r['class_distribution']}\n")
                    if r["path"]:
                        f.write("  条件: " + " 且 ".join(r["path"]) + "\n")
                    else:
                        f.write("  条件: (无条件)\n")
                    f.write("\n")
            print(f"已保存{label}规则: tree_rules_warning_{level}.txt")
    
    save_warning_rules(all_rules, out_dir)
    
    # 为了兼容性，创建一个sklearn风格的决策树用于计算特征重要性
    # 但实际规则来自自定义树
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(Xv, yw)
    
    # 保存特征重要性（注意：这是标准决策树的重要性，不是自定义树的）
    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values(by="importance", ascending=False)
    
    df_importance.to_csv(
        os.path.join(out_dir, "tree_feature_importances.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    
    print("已保存特征重要性文件: tree_feature_importances.csv")
    print(f"注意: 实际规则强制使用'{forced_feature_name}'作为根节点")
    
    # 返回结果用于生成报告
    return {
        "df": df,
        "feature_cols": feature_cols,
        "rules_by_level": all_rules,
        "feature_scores": feature_scores,
        "warning_labels": warning_labels,
        "forced_feature": forced_feature_name,
        "stats": {
            "total_samples": len(df),
            "valid_samples": mask_valid.sum(),
            "warning_distribution": dict(warning_counts),
        }
    }


def run_batch_experiments(
    data_xlsx: str,
    base_out_dir: str = None
):
    """运行批量实验（强制使用'7天累计雨量'作为根节点）"""
    
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
    print("强制使用'7天累计雨量'作为决策树的根节点")
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
            # 运行分析（强制使用'7天累计雨量'）
            results = run_analysis_with_forced_feature(
                data_xlsx,
                out_dir,
                max_depth=depth,
                min_samples_leaf=leaf
            )
            
            # 生成报告
            generate_report(results, out_dir, max_depth=depth, min_samples_leaf=leaf)
            
            # 收集特征重要性信息
            importance_file = os.path.join(out_dir, "tree_feature_importances.csv")
            if os.path.exists(importance_file):
                df_imp = pd.read_csv(importance_file, encoding="utf-8-sig")
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
            
            forced_feature = results.get("forced_feature", "7天累计雨量")
            
            summary_data.append({
                "实验编号": i,
                "参数名称": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "强制根节点特征": forced_feature,
                "使用特征数": feature_count,
                "最重要特征": top_feature,
                "最高重要性": f"{top_importance:.4f}",
                "规则总数": total_rules,
                "覆盖样本总数": total_covered_samples,
                "准确样本总数": total_accurate_samples,
                "平均规则准确率": f"{avg_accuracy:.4f}",
                "总体准确率": f"{overall_accuracy:.4f}",
                "综合评分": f"{overall_accuracy * total_covered_samples:.2f}",
                "输出目录": out_dir
            })
            
            print(f"\n✓ 实验 {i} 完成")
            print(f"  - 强制根节点特征: {forced_feature}")
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
                "强制根节点特征": "失败",
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
    report_lines.append("# 批量实验汇总报告（强制使用'7天累计雨量'作为根节点）\n\n")
    report_lines.append("## 实验参数组合\n\n")
    report_lines.append("测试了10种不同的决策树参数组合，所有实验都强制使用'7天累计雨量'作为决策树的根节点：\n\n")
    report_lines.append("| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 强制根节点特征 | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本总数 | 总体准确率 | 综合评分 |\n")
    report_lines.append("|---------|---------|-----------|------------------|--------------|-----------|-----------|----------|------------|-----------|----------|\n")
    
    for row in summary_data:
        report_lines.append(
            f"| {row['实验编号']} | {row['参数名称']} | {row['max_depth']} | {row['min_samples_leaf']} | "
            f"{row['强制根节点特征']} | {row['使用特征数']} | {row['最重要特征']} | {row['规则总数']} | "
            f"{row['覆盖样本总数']} | {row['总体准确率']} | {row['综合评分']} |\n"
        )
    
    report_lines.append("\n## 最优解法分析\n\n")
    
    # 筛选出成功完成的实验
    valid_experiments = [row for row in summary_data if row['综合评分'] != '失败']
    
    if valid_experiments:
        # 按综合评分排序
        def get_score(row):
            try:
                return float(row['综合评分'])
            except:
                return 0.0
        
        sorted_experiments = sorted(valid_experiments, key=get_score, reverse=True)
        
        report_lines.append("### Top 3 最优解法（按综合评分排序）\n\n")
        report_lines.append("综合评分 = 总体准确率 × 覆盖样本总数，综合考虑准确率和覆盖范围。\n\n")
        report_lines.append("**注意**: 所有实验都强制使用'7天累计雨量'作为决策树的根节点。\n\n")
        
        for rank, exp in enumerate(sorted_experiments[:3], 1):
            report_lines.append(f"#### 第{rank}名：{exp['参数名称']} (max_depth={exp['max_depth']}, min_samples_leaf={exp['min_samples_leaf']})\n\n")
            report_lines.append(f"- **综合评分**: {exp['综合评分']}\n")
            report_lines.append(f"- **总体准确率**: {exp['总体准确率']}\n")
            report_lines.append(f"- **覆盖样本总数**: {exp['覆盖样本总数']}\n")
            report_lines.append(f"- **规则总数**: {exp['规则总数']}\n")
            report_lines.append(f"- **使用特征数**: {exp['使用特征数']}\n")
            report_lines.append(f"- **强制根节点特征**: {exp['强制根节点特征']}\n")
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
        report_lines.append(f"2. **强制根节点**: 所有实验都强制使用'{top1['强制根节点特征']}'作为决策树的根节点\n")
        report_lines.append(f"3. **性能表现**: 在{len(valid_experiments)}个有效实验中，综合评分最高（{top1['综合评分']}）\n")
        report_lines.append(f"4. **准确率**: {top1['总体准确率']}，意味着该模型在覆盖的样本上预测准确率较高\n")
        report_lines.append(f"5. **覆盖范围**: 覆盖了{top1['覆盖样本总数']}个样本，说明规则具有良好的泛化能力\n")
        report_lines.append(f"6. **特征选择**: 使用了{top1['使用特征数']}个特征，其中最重要特征是{top1['最重要特征']}\n")
        report_lines.append(f"7. **规则数量**: 共{top1['规则总数']}条规则，规则数量适中，既不过于复杂也不过于简单\n\n")
        
        report_lines.append("**建议使用该参数组合进行最终的预警模型构建。**\n\n")
        report_lines.append("**重要说明**: 本批次实验强制使用'7天累计雨量'作为决策树的根节点，所有规则都包含该特征作为第一个判断条件。\n\n")
    
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
    parser = argparse.ArgumentParser(description="批量实验：10种树深度和叶子节点数组合（强制使用'7天累计雨量'作为根节点）")
    
    # 使用脚本所在目录为基准，构造健壮默认值
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    default_data = os.path.join(project_root, "样本数据.xlsx")
    default_out = os.path.join(project_root, "analysis_output_batch_1118")

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

    # 如果传入的是相对路径，则基于项目根目录
    if not os.path.isabs(args.out):
        args.out = os.path.join(project_root, args.out)
    
    os.makedirs(args.out, exist_ok=True)
    run_batch_experiments(args.data, args.out)


if __name__ == "__main__":
    main()