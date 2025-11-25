#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验：测试100种不同的树深度和叶子节点数组合
直接使用数据文件中的'预警'列，只考虑7天的特定特征
强制使用'7天日均降雨量'作为决策树的根节点（第一个判断条件）
必须使用'7天平均气温'作为决策树的区分节点之一（在深度1强制使用）
其他子节点使用所有特征进行详细划分
"""
import argparse
import os
import sys
import re

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
    _find_date_column, _to_datetime_series,
    _find_warning_column, _normalize_warning_level,
    grey_relational_grade, pearson_with_binary
)


def find_7day_daily_rainfall_feature(feature_cols):
    """查找'7天日均降雨量'特征"""
    candidates = []
    for col in feature_cols:
        col_str = str(col)
        # 匹配包含"7"和"日均"，以及"降雨"或"雨量"的列
        if ("7" in col_str or "七" in col_str) and \
           any(kw in col_str for kw in ["日均"]) and \
           any(kw in col_str for kw in ["降雨", "降雨量", "雨量", "降水", "降水量"]):
            candidates.append(col)
    
    if candidates:
        return candidates[0]  # 返回第一个匹配的
    return None


def find_7day_avg_temperature_feature(feature_cols):
    """查找'7天平均气温'特征"""
    candidates = []
    for col in feature_cols:
        col_str = str(col)
        # 匹配包含"7"和"平均"或"日均"，以及"温度"或"气温"的列
        if ("7" in col_str or "七" in col_str) and \
           any(kw in col_str for kw in ["平均", "日均"]) and \
           any(kw in col_str for kw in ["温度", "气温", "平均温度", "平均气温"]):
            candidates.append(col)
    
    if candidates:
        return candidates[0]  # 返回第一个匹配的
    return None


def entropy_impurity(y):
    """计算熵不纯度"""
    if len(y) == 0:
        return 0.0
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return entropy(probs, base=2)


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


def build_tree_with_forced_root_and_temperature(X, y, feature_names, 
                                                  forced_root_feature_name, 
                                                  forced_temperature_feature_name,
                                                  max_depth, min_samples_leaf, current_depth=0):
    """
    构建决策树，强制使用指定特征作为根节点，在深度1强制使用'7天平均气温'，其他子节点使用标准算法
    
    参数:
        forced_root_feature_name: 强制作为根节点的特征（'7天日均降雨量'）
        forced_temperature_feature_name: 强制在深度1使用的特征（'7天平均气温'）
    """
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
    
    # 根节点（深度0）：强制使用'7天日均降雨量'
    if current_depth == 0:
        # 找到强制根节点特征在特征列表中的索引
        forced_feature_idx = None
        for i, name in enumerate(feature_names):
            if name == forced_root_feature_name:
                forced_feature_idx = i
                break
        
        if forced_feature_idx is None:
            raise ValueError(f"未找到强制根节点特征: {forced_root_feature_name}")
        
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
        
        # 递归构建左右子树（深度1将强制使用'7天平均气温'）
        left_tree = build_tree_with_forced_root_and_temperature(
            X[left_idx], y[left_idx], feature_names, 
            forced_root_feature_name, forced_temperature_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if left_idx.sum() >= min_samples_leaf else None
        
        right_tree = build_tree_with_forced_root_and_temperature(
            X[right_idx], y[right_idx], feature_names,
            forced_root_feature_name, forced_temperature_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if right_idx.sum() >= min_samples_leaf else None
        
        return {
            'type': 'split',
            'feature': forced_root_feature_name,
            'feature_idx': forced_feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'left_samples': int(left_idx.sum()),
            'right_samples': int(right_idx.sum())
        }
    # 深度1：强制使用'7天平均气温'
    elif current_depth == 1:
        # 找到'7天平均气温'特征在特征列表中的索引
        temp_feature_idx = None
        for i, name in enumerate(feature_names):
            if name == forced_temperature_feature_name:
                temp_feature_idx = i
                break
        
        if temp_feature_idx is None:
            raise ValueError(f"未找到强制温度特征: {forced_temperature_feature_name}")
        
        # 使用'7天平均气温'进行分割
        threshold, left_idx, right_idx = find_best_split_for_feature(X, y, temp_feature_idx)
        
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
        
        # 递归构建左右子树（深度2及以后使用标准算法）
        left_tree = build_tree_with_forced_root_and_temperature(
            X[left_idx], y[left_idx], feature_names,
            forced_root_feature_name, forced_temperature_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if left_idx.sum() >= min_samples_leaf else None
        
        right_tree = build_tree_with_forced_root_and_temperature(
            X[right_idx], y[right_idx], feature_names,
            forced_root_feature_name, forced_temperature_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if right_idx.sum() >= min_samples_leaf else None
        
        return {
            'type': 'split',
            'feature': forced_temperature_feature_name,
            'feature_idx': temp_feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'left_samples': int(left_idx.sum()),
            'right_samples': int(right_idx.sum())
        }
    else:
        # 深度2及以后：使用标准决策树算法选择最佳特征
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
        left_tree = build_tree_with_forced_root_and_temperature(
            X[best_left_idx], y[best_left_idx], feature_names,
            forced_root_feature_name, forced_temperature_feature_name,
            max_depth, min_samples_leaf, current_depth + 1
        ) if best_left_idx.sum() >= min_samples_leaf else None
        
        right_tree = build_tree_with_forced_root_and_temperature(
            X[best_right_idx], y[best_right_idx], feature_names,
            forced_root_feature_name, forced_temperature_feature_name,
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


def extract_rules_from_custom_tree(tree, feature_names, X, y, warning_labels, path=None, sample_idx=None):
    """从自定义树中提取规则"""
    if path is None:
        path = []
    if sample_idx is None:
        sample_idx = np.ones(X.shape[0], dtype=bool)
    
    all_rules = {level: [] for level in warning_labels.keys()}
    
    if tree['type'] == 'leaf':
        # 获取当前叶子节点的实际样本
        leaf_y = y[sample_idx]
        if len(leaf_y) == 0:
            return all_rules
        
        samples = len(leaf_y)
        unique, counts = np.unique(leaf_y, return_counts=True)
        max_idx = np.argmax(counts)
        pred_class = int(unique[max_idx])
        pred_count = int(counts[max_idx])
        pred_rate = float(pred_count) / float(samples) if samples > 0 else 0.0
        
        class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        
        # 根据样本数量调整阈值
        if samples < 5:
            min_rate = 0.8
        elif samples < 10:
            min_rate = 0.7
        elif samples < 20:
            min_rate = 0.6
        else:
            min_rate = 0.5
        
        if pred_class in warning_labels and pred_rate >= min_rate:
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
        feature_idx = tree['feature_idx']
        
        # 根据当前样本索引和特征阈值分割
        left_idx = sample_idx & (X[:, feature_idx] <= threshold)
        right_idx = sample_idx & (X[:, feature_idx] > threshold)
        
        # 递归处理左右子树
        if tree['left'] is not None and left_idx.sum() > 0:
            left_rules = extract_rules_from_custom_tree(
                tree['left'], feature_names, X, y,
                warning_labels, path + [f"{feature_name} ≤ {threshold:.3f}"], left_idx
            )
            for level in left_rules:
                all_rules[level].extend(left_rules[level])
        
        if tree['right'] is not None and right_idx.sum() > 0:
            right_rules = extract_rules_from_custom_tree(
                tree['right'], feature_names, X, y,
                warning_labels, path + [f"{feature_name} > {threshold:.3f}"], right_idx
            )
            for level in right_rules:
                all_rules[level].extend(right_rules[level])
    
    return all_rules


def predict_with_custom_tree(tree, X):
    """使用自定义树进行预测"""
    predictions = []
    for i in range(X.shape[0]):
        node = tree
        while node['type'] == 'split':
            feature_idx = node['feature_idx']
            threshold = node['threshold']
            if X[i, feature_idx] <= threshold:
                node = node['left']
            else:
                node = node['right']
            if node is None:
                # 如果子树为None，使用父节点的多数类
                break
        if node is None or node['type'] != 'leaf':
            # 默认预测为0级
            predictions.append(0)
        else:
            predictions.append(node['pred_class'])
    return np.array(predictions)


def calculate_feature_importance_from_tree(tree, feature_names, total_samples):
    """从自定义树计算特征重要性"""
    importance = np.zeros(len(feature_names))
    
    def traverse(node, parent_samples):
        if node['type'] == 'split':
            feature_idx = node['feature_idx']
            left_samples = node.get('left_samples', 0)
            right_samples = node.get('right_samples', 0)
            
            # 计算该节点的重要性（基于样本权重）
            if parent_samples > 0:
                weight = parent_samples / total_samples
                importance[feature_idx] += weight
            
            # 递归处理子树
            if node['left'] is not None:
                traverse(node['left'], left_samples)
            if node['right'] is not None:
                traverse(node['right'], right_samples)
    
    traverse(tree, total_samples)
    return importance


def calculate_warning_level_from_infection_rate(down_rate, up_rate):
    """
    根据感病率计算预警等级
    
    预警标准：
    (1) 树下落地花果感病率≤10%，树上感病率=0，则预警等级为轻度（1级，不易发生）
    (2) 树下落地花果感病率>10%且≤40%，树上感病率=0，预警等级为中度（2级，较易发生）
    (3) 树下落地花果感病率>40%，树上感病率>0，预警等级为重度（3级，易发生）
    
    返回: 1, 2, 3 或 None（如果不符合任何标准）
    """
    if down_rate is None or up_rate is None:
        return None
    
    down_rate = float(down_rate)
    up_rate = float(up_rate)
    
    # 轻度：树下≤10%，树上=0
    if down_rate <= 10.0 and up_rate == 0.0:
        return 1
    
    # 中度：树下>10%且≤40%，树上=0
    if 10.0 < down_rate <= 40.0 and up_rate == 0.0:
        return 2
    
    # 重度：树下>40%，树上>0
    if down_rate > 40.0 and up_rate > 0.0:
        return 3
    
    # 不符合标准的情况返回None
    return None


def select_7day_features_only(df: pd.DataFrame) -> list:
    """
    只选择7天的特定特征：
    - 7天平均气温
    - 7天日均降雨量
    - 7天平均相对湿度
    - 7天累积降雨时数
    - 7天累积日照时数
    """
    features = []
    
    for col in df.columns:
        col_str = str(col)
        
        # 检查7天平均气温（必须包含"7"和"平均"或"日均"，以及"温度"或"气温"）
        if ("7" in col_str or "七" in col_str) and \
           any(p in col_str for p in ["平均", "日均"]) and \
           any(p in col_str for p in ["温度", "气温", "平均温度", "平均气温"]):
            features.append(col)
        # 检查7天日均降雨量（必须包含"7"和"日均"，以及"降雨"或"雨量"）
        elif ("7" in col_str or "七" in col_str) and \
             any(p in col_str for p in ["日均"]) and \
             any(p in col_str for p in ["降雨", "降雨量", "雨量", "降水", "降水量"]):
            features.append(col)
        # 检查7天平均相对湿度（必须包含"7"和"平均"或"日均"，以及"相对湿度"或"湿度"）
        elif ("7" in col_str or "七" in col_str) and \
             any(p in col_str for p in ["平均", "日均"]) and \
             any(p in col_str for p in ["相对湿度", "湿度", "平均相对湿度"]):
            features.append(col)
        # 检查7天累积降雨时数（必须包含"7"和"累积"或"累计"，以及"降雨时数"或"降水时数"）
        elif ("7" in col_str or "七" in col_str) and \
             any(p in col_str for p in ["累积", "累计"]) and \
             any(p in col_str for p in ["降雨时数", "降水时数"]):
            features.append(col)
        # 检查7天累积日照时数（必须包含"7"和"累积"或"累计"，以及"日照"或"日照时数"）
        elif ("7" in col_str or "七" in col_str) and \
             any(p in col_str for p in ["累积", "累计"]) and \
             any(p in col_str for p in ["日照", "日照时数"]):
            features.append(col)
    
    # 去重
    out = []
    for c in features:
        if c not in out:
            out.append(c)
    
    return out


def extract_warning_rules(clf: DecisionTreeClassifier, feature_names: list, X: np.ndarray, y: np.ndarray, warning_labels: dict) -> dict:
    """提取每个预警等级的阈值规则（只考虑1,2,3级）"""
    t = clf.tree_
    all_rules: dict = {level: [] for level in warning_labels.keys()}
    
    def recurse(node: int, path: list, idx: np.ndarray):
        if t.feature[node] == -2:  # leaf node
            samples = idx.sum()
            if samples == 0:
                return
            # 统计该叶子节点中每个预警等级的样本数
            leaf_y = y[idx]
            unique, counts = np.unique(leaf_y, return_counts=True)
            # 找到最多的类别
            max_idx = np.argmax(counts)
            pred_class = unique[max_idx]
            pred_rate = float(counts[max_idx]) / float(samples)
            
            # 根据样本数量调整阈值
            if samples < 5:
                min_rate = 0.8  # 小样本需要更高准确率
            elif samples < 10:
                min_rate = 0.7
            elif samples < 20:
                min_rate = 0.6
            else:
                min_rate = 0.5  # 大样本可以稍微放宽
            
            if pred_class in warning_labels and pred_rate >= min_rate:
                all_rules[pred_class].append({
                    "path": list(path),
                    "samples": int(samples),
                    "pred_rate": pred_rate,
                    "pred_count": int(counts[max_idx]),
                    "class_distribution": {int(u): int(c) for u, c in zip(unique, counts)}
                })
            return
        
        feat_idx = t.feature[node]
        thr = t.threshold[node]
        name = feature_names[feat_idx]
        left_idx = idx & (X[:, feat_idx] <= thr)
        right_idx = idx & (X[:, feat_idx] > thr)
        recurse(t.children_left[node], path + [f"{name} ≤ {thr:.3f}"], left_idx)
        recurse(t.children_right[node], path + [f"{name} > {thr:.3f}"], right_idx)
    
    recurse(0, [], np.ones(X.shape[0], dtype=bool))
    
    # 对每个等级的规则排序
    for level in all_rules:
        all_rules[level].sort(key=lambda r: (r["pred_rate"], r["samples"]), reverse=True)
    
    return all_rules


def run_analysis_1120(data_xlsx: str, out_dir: str, max_depth: int = 3, min_samples_leaf: int = 5):
    """运行完整分析流程（基于新的预警标准）"""
    os.makedirs(out_dir, exist_ok=True)
    
    print("正在读取数据文件...")
    df = pd.read_excel(data_xlsx, sheet_name=0)
    
    print(f"数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")
    
    # 识别日期列
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError("无法识别日期列，请检查数据文件的日期列名称")
    
    print(f"日期列: {date_col}")
    
    # 标准化日期
    df[date_col] = _to_datetime_series(df[date_col])
    date_na = df[date_col].isna().sum()
    if date_na > 0:
        print(f"警告: 数据中有 {date_na} 个无效日期")
        df = df.dropna(subset=[date_col])
    
    # 识别预警列
    warning_col = _find_warning_column(df)
    if not warning_col:
        raise ValueError("未找到预警列，请确认数据文件中包含'预警'列")
    
    print(f"预警列: {warning_col}")
    
    # 排除预警为"未定义"的样本
    before_undef = len(df)
    df = df[df[warning_col].astype(str).str.strip() != "未定义"]
    after_undef = len(df)
    if before_undef > after_undef:
        print(f"排除预警为'未定义'的样本: {before_undef} -> {after_undef} (删除 {before_undef - after_undef} 条)")
    
    # 转换预警值为数值
    print("\n=== 使用预警列数据 ===")
    df["预警数值"] = df[warning_col].map(_normalize_warning_level)
    
    # 删除预警值无法识别的样本
    before_drop = len(df)
    df = df.dropna(subset=["预警数值"])
    after_drop = len(df)
    if before_drop > after_drop:
        print(f"删除预警值无法识别的样本: {before_drop} -> {after_drop} (删除 {before_drop - after_drop} 条)")
    
    df["预警数值"] = df["预警数值"].clip(lower=0, upper=3).astype(int)
    
    # 统计预警等级分布（0-3级全量）
    warning_counts = df["预警数值"].value_counts().sort_index()
    warning_labels_all = {
        0: "0级（不发生）",
        1: "1级（轻度，不易发生）",
        2: "2级（中度，较易发生）",
        3: "3级（重度，易发生）"
    }
    warning_labels_display = {k: warning_labels_all[k] for k in [1, 2, 3]}
    
    print("\n预警等级分布:")
    for level, count in warning_counts.items():
        label = warning_labels_all.get(level, f"{level}级")
        print(f"  {label}: {count} 条")
    
    # 划分训练/验证（2025年为验证集）
    df["观测年份"] = df[date_col].dt.year
    val_mask = df["观测年份"] == 2025
    train_mask = ~val_mask
    
    train_samples = int(train_mask.sum())
    val_samples = int(val_mask.sum())
    
    if train_samples == 0:
        raise ValueError("没有可用于训练的样本（非2025年的观测数据）。")
    
    print(f"\n样本划分：训练 {train_samples} 条（≤2024年），验证 {val_samples} 条（2025年）。")
    
    # 选择特征列（只选择7天的特定特征）
    feature_cols = select_7day_features_only(df)
    if not feature_cols:
        raise ValueError(f"未匹配到7天特征列。可用列: {list(df.columns)}")
    
    print(f"\n识别到 {len(feature_cols)} 个7天特征:")
    for f in feature_cols:
        print(f"  - {f}")
    
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
    
    # 准备特征和标签（使用训练集均值填充缺失，避免丢弃样本）
    feature_df = df[feature_cols].astype(float)
    train_feature_means = feature_df[train_mask].mean()
    train_feature_means = train_feature_means.fillna(0.0)
    feature_df = feature_df.fillna(train_feature_means)
    
    X_all = feature_df.to_numpy()
    y_all = df["预警数值"].astype(int).to_numpy()
    
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_val = X_all[val_mask]
    y_val = y_all[val_mask]
    
    # 1. 特征筛选：对每个预警等级分别进行关联分析
    print("\n=== 1. 特征筛选分析 ===")
    
    feature_scores = {}
    for level in [1, 2, 3]:
        y_binary = (y_train == level).astype(int)
        if y_binary.sum() == 0:
            continue
        pear = pearson_with_binary(X_train, y_binary)
        gra = grey_relational_grade(X_train, y_binary.astype(float), rho=0.5)
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
    
    # 2. 决策树阈值分析（多分类，只考虑1,2,3级，强制使用'7天日均降雨量'作为根节点，深度1强制使用'7天平均气温'）
    print("\n=== 2. 决策树阈值分析（强制使用'7天日均降雨量'作为根节点，深度1强制使用'7天平均气温'）===")
    
    # 查找强制使用的特征
    forced_root_feature_name = find_7day_daily_rainfall_feature(feature_cols)
    if forced_root_feature_name is None:
        raise ValueError("未找到'7天日均降雨量'特征，请检查数据文件")
    
    forced_temperature_feature_name = find_7day_avg_temperature_feature(feature_cols)
    if forced_temperature_feature_name is None:
        raise ValueError("未找到'7天平均气温'特征，请检查数据文件")
    
    print(f"强制使用特征作为根节点: {forced_root_feature_name}")
    print(f"强制在深度1使用特征: {forced_temperature_feature_name}")
    print(f"其他子节点将使用所有特征进行划分: {', '.join(feature_cols)}")
    
    # 构建自定义决策树（强制根节点和深度1）
    custom_tree = build_tree_with_forced_root_and_temperature(
        X_train, y_train, feature_cols, 
        forced_root_feature_name, forced_temperature_feature_name,
        max_depth, min_samples_leaf
    )
    
    # 提取规则
    all_rules = extract_rules_from_custom_tree(
        custom_tree, feature_cols, X_train, y_train, warning_labels_display
    )
    
    # 对每个等级的规则排序
    for level in all_rules:
        all_rules[level].sort(key=lambda r: (r["pred_rate"], r["samples"]), reverse=True)
    
    # 保存规则
    def save_warning_rules(rules_by_level: dict, out_dir: str):
        for level, rules in rules_by_level.items():
            if not rules:
                continue
            label = warning_labels_display.get(level, f"{level}级")
            path = os.path.join(out_dir, f"tree_rules_warning_{level}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{label}预警规则，按预测准确率与覆盖样本排序\n\n")
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
    
    # 保存特征重要性
    feature_importance = calculate_feature_importance_from_tree(
        custom_tree, feature_cols, len(X_train)
    )
    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance,
    }).sort_values(by="importance", ascending=False)
    
    df_importance.to_csv(
        os.path.join(out_dir, "tree_feature_importances.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    
    print("已保存特征重要性文件: tree_feature_importances.csv")
    
    # 验证集表现
    validation_metrics = {
        "samples": val_samples,
        "accuracy": None,
        "correct": 0
    }
    if val_samples > 0:
        y_val_pred = predict_with_custom_tree(custom_tree, X_val)
        val_correct = int((y_val_pred == y_val).sum())
        val_accuracy = float(val_correct / val_samples)
        validation_metrics["accuracy"] = val_accuracy
        validation_metrics["correct"] = val_correct
        print(f"验证集（2025年）准确率: {val_accuracy:.4f} ({val_correct}/{val_samples})")
    else:
        print("提示: 数据中没有2025年观测，无法评估验证集准确率。")
    
    # 返回结果用于生成报告
    return {
        "df": df,
        "feature_cols": feature_cols,
        "rules_by_level": all_rules,
        "feature_scores": feature_scores,
        "warning_labels": warning_labels_display,
        "warning_labels_all": warning_labels_all,
        "stats": {
            "total_samples": len(df),
            "train_samples": train_samples,
            "validation_samples": val_samples,
            "warning_distribution": dict(warning_counts),
        },
        "validation_metrics": validation_metrics
    }


def generate_report_1120(results: dict, out_dir: str, max_depth: int = 3, min_samples_leaf: int = 5):
    """生成详细分析报告"""
    report_lines = []
    
    stats = results["stats"]
    validation_metrics = results.get("validation_metrics", {})
    total_samples = stats.get("total_samples", 0)
    train_samples = stats.get("train_samples", total_samples)
    val_samples = stats.get("validation_samples", 0)
    
    report_lines.append("# 荔枝霜疫霉预警等级阈值模型分析报告（基于新预警标准）\n\n")
    report_lines.append("## 数据说明\n\n")
    report_lines.append("- **数据文件**: 样本数据_验证修复.xlsx（包含19-25年采集的数据）\n")
    report_lines.append("- **目标变量**: 预警等级（直接使用数据文件中的'预警'列：0-3级）\n")
    report_lines.append("- **特征选择**: 只使用7天的特定特征（平均气温、日均降雨量、平均相对湿度、累积降雨时数、累积日照时数）\n")
    report_lines.append(f"- **样本总数**: {total_samples} 条（训练 {train_samples}，验证 {val_samples}）\n")
    if val_samples > 0 and validation_metrics.get("accuracy") is not None:
        report_lines.append(f"- **2025验证集准确率**: {validation_metrics['accuracy']:.3f}（{validation_metrics['correct']}/{validation_metrics['samples']}）\n")
    report_lines.append("\n")
    
    # 预警等级分布
    report_lines.append("### 预警等级分布\n\n")
    warning_labels = results["warning_labels"]
    warning_dist = results["stats"]["warning_distribution"]
    for level, count in sorted(warning_dist.items()):
        if level in warning_labels:
            label = warning_labels[level]
            report_lines.append(f"- {label}: {count} 条\n")
    report_lines.append("\n")
    
    # 1. 结论概览
    report_lines.append("## 1. 结论概览\n\n")
    
    rules_by_level = results["rules_by_level"]
    
    # 对每个预警等级（1,2,3）输出规则
    for level in [1, 2, 3]:
        if level not in rules_by_level or not rules_by_level[level]:
            continue
        label = warning_labels.get(level, f"{level}级")
        report_lines.append(f"### 1.{level}. {label}预警的关键因子与阈值（规则来自决策树）\n\n")
        
        rules = rules_by_level[level]
        for i, r in enumerate(rules[:3], 1):  # 取前3条规则
            report_lines.append(f"\t{i}.规则{i}（覆盖样本={r['samples']}，准确率={int(r['pred_rate']*100)}%）：{' 且 '.join(r['path'])}\n\n")
        
        if len(rules) > 0:
            best = rules[0]
            report_lines.append(f"最优规则：{' 且 '.join(best['path'])}（samples={best['samples']}，准确率={best['pred_rate']:.3f}，score={int(best['pred_rate']*best['samples'])}）\n\n")
    
    # 2. 关联强度
    report_lines.append("## 2. 关联强度（GRA 分数，值越大关联越强）\n\n")
    
    feature_scores = results["feature_scores"]
    for level in [1, 2, 3]:
        if level not in feature_scores:
            continue
        label = warning_labels.get(level, f"{level}级")
        report_lines.append(f"### 2.{level} {label}预警（节选，来自 feature_scores_warning_{level}.csv）\n\n")
        
        # 读取对应的CSV文件
        csv_path = os.path.join(out_dir, f"feature_scores_warning_{level}.csv")
        if os.path.exists(csv_path):
            df_score = pd.read_csv(csv_path, encoding="utf-8-sig")
            for _, row in df_score.head(8).iterrows():
                feature_name = row["feature"]
                gra_score = row["gra"]
                pear_val = row["pearson"]
                note = ""
                if "日照" in feature_name and pear_val < 0:
                    note = "（与发生呈反向关系，Pearson为负）"
                report_lines.append(f"\t{feature_name}: {gra_score:.3f}{note}\n\n")
    
    # 3. 条件设置
    report_lines.append("## 3. 条件设置\n\n")
    
    for level in [1, 2, 3]:
        if level not in rules_by_level or not rules_by_level[level]:
            continue
        label = warning_labels.get(level, f"{level}级")
        report_lines.append(f"### 3.{level} {label}预警更易出现在：\n\n")
        best_rule = rules_by_level[level][0]
        report_lines.append(f"根据决策树分析，主要阈值条件为：{' 且 '.join(best_rule['path'])}\n\n")
    
    # 4. 方法概述
    report_lines.append("## 4. 方法概述\n\n")
    
    report_lines.append("**预警等级来源**：\n\n")
    report_lines.append("- 直接使用数据文件中的'预警'列，排除'未定义'的样本\n")
    report_lines.append("- 预警等级：0级（不发生）、1级（轻度）、2级（中度）、3级（重度）\n\n")
    
    report_lines.append("**特征选择**：只使用7天的以下特征：\n\n")
    report_lines.append("- 7天平均气温\n")
    report_lines.append("- 7天日均降雨量\n")
    report_lines.append("- 7天平均相对湿度\n")
    report_lines.append("- 7天累积降雨时数\n")
    report_lines.append("- 7天累积日照时数\n\n")
    
    report_lines.append("**皮尔逊相关（点二列相关）**：对每个预警等级（1,2,3）分别进行二分类分析，衡量特征与该预警等级的相关强度。\n\n")
    
    report_lines.append("**灰色关联分析 GRA（分辨系数 ρ=0.5）**：衡量特征与目标序列\"形状接近度\"，对非线性/不同量纲更稳健。\n\n")
    
    report_lines.append(f"**决策树多分类（深度 max_depth={max_depth}, min_samples_leaf={min_samples_leaf}）**：使用预警等级（1-3级）作为目标变量，强制使用'7天日均降雨量'作为根节点（第一个判断条件），在深度1强制使用'7天平均气温'作为区分节点，其他子节点使用所有特征（包括7天平均相对湿度、7天累计降雨时数、7天累计日照时数等）进行详细划分，从7天特征中提取每个预警等级的阈值组合规则。\n\n")
    
    # 5. 参考文献
    report_lines.append("## 5. 参考文献\n\n")
    
    report_lines.append("[1]吕安瑞,严梦荧,张妍,等.基于土壤中霜疫霉活力变化防治荔枝霜疫病初探[C]//中国植物病理学会.中国植物病理学会2024年学术年会论文集.华南农业大学园艺学院华南农业大学植物保护学院;,2024:167.DOI:10.26914/c.cnkihy.2024.022408.\n\n")
    
    report_lines.append("结果表明，荔枝霜疫霉可在果园土壤中存活１６个月，成功越夏、越冬，成为荔枝霜疫病初侵染源；土壤中荔枝霜疫霉存在２个活跃期，第一个是冬末初春（１２月中旬至次年开花前），此时土壤中荔枝霜疫霉活力逐渐增强，处于催醒状态，作为初侵染源具有侵染能力；第二个活跃期３月下旬至４月上中旬，为荔枝开花后至第二次生理落果期，此时土壤中复苏的荔枝霜疫霉在雨水的作用下，侵染落地花果，以多种生物学形态习居在土壤中；健康荔枝花果落地后染病快速，花落地后３ｈ即可发病，２４ｈ后感染率超５０％；幼果发病略慢，接触土壤４５ｈ感染率大于５０％。落花和落果是土壤中霜疫霉初次侵染和再侵染的寄主，导致荔枝霜疫霉辗转传播侵染。\n\n")
    
    # 保存报告
    report_path = os.path.join(out_dir, "1120.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    
    print(f"\n已生成详细报告: {report_path}")


def run_batch_experiments_1120(
    data_xlsx: str,
    base_out_dir: str = "analysis_output_batch_1120"
):
    """运行批量实验（基于新预警标准）"""
    
    # 定义100种参数组合（深度2-11，叶子节点1-10的笛卡尔积）
    experiments = []
    depth_range = list(range(2, 12))  # 2到11，共10个值
    leaf_range = list(range(1, 11))  # 1到10，共10个值
    
    for depth in depth_range:
        for leaf in leaf_range:
            experiments.append({
                "depth": depth,
                "leaf": leaf,
                "name": f"depth{depth}_leaf{leaf}"
            })
    
    # 确保正好100种组合
    assert len(experiments) == 100, f"实验组合数量应为100，实际为{len(experiments)}"
    
    os.makedirs(base_out_dir, exist_ok=True)
    
    # 汇总结果
    summary_data = []
    
    print("=" * 80)
    print("批量实验：测试100种树深度和叶子节点数组合")
    print("直接使用数据文件中的'预警'列，只考虑7天特征")
    print("强制使用'7天日均降雨量'作为决策树的根节点（第一个判断条件）")
    print("在深度1强制使用'7天平均气温'作为区分节点")
    print("其他子节点使用所有特征（包括7天平均相对湿度、7天累计降雨时数等）进行详细划分")
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
            results = run_analysis_1120(
                data_xlsx,
                out_dir,
                max_depth=depth,
                min_samples_leaf=leaf
            )
            
            # 生成报告
            generate_report_1120(results, out_dir, max_depth=depth, min_samples_leaf=leaf)
            
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
            
            train_samples = results["stats"].get("train_samples", 0)
            val_samples = results["stats"].get("validation_samples", 0)
            validation_metrics = results.get("validation_metrics", {})
            val_accuracy_value = validation_metrics.get("accuracy")
            val_accuracy_str = f"{val_accuracy_value:.4f}" if val_accuracy_value is not None else "-"
            
            # 统计规则数量和规则质量（基于训练集）
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
            
            summary_row = {
                "实验编号": i,
                "参数名称": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "训练样本数": train_samples,
                "验证样本数": val_samples,
                "验证准确率": val_accuracy_str,
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
            }
            summary_data.append(summary_row)
            
            print(f"\n✓ 实验 {i} 完成")
            print(f"  - 使用特征数: {feature_count}")
            print(f"  - 最重要特征: {top_feature} (重要性={top_importance:.4f})")
            print(f"  - 规则总数: {total_rules}")
            print(f"  - 覆盖样本总数: {total_covered_samples}")
            print(f"  - 总体准确率: {overall_accuracy:.4f}")
            if val_samples > 0:
                print(f"  - 验证集准确率(2025): {val_accuracy_str}")
            else:
                print("  - 验证集: 暂无2025年样本")
            
        except Exception as e:
            print(f"\n✗ 实验 {i} 失败: {str(e)}")
            summary_data.append({
                "实验编号": i,
                "参数名称": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "训练样本数": "失败",
                "验证样本数": "失败",
                "验证准确率": "失败",
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
    report_lines.append("# 批量实验汇总报告（基于新预警标准）\n\n")
    report_lines.append("## 实验参数组合\n\n")
    report_lines.append(f"测试了{len(experiments)}种不同的决策树参数组合：\n\n")
    report_lines.append("- **特征选择**: 只使用7天的特定特征（平均气温、日均降雨量、平均相对湿度、累积降雨时数、累积日照时数）\n")
    report_lines.append("- **预警标准**: 直接使用数据文件中的'预警'列（0-3级），排除'未定义'的样本\n")
    report_lines.append("- **根节点**: 强制使用'7天日均降雨量'作为决策树的根节点（第一个判断条件）\n")
    report_lines.append("- **深度1节点**: 强制使用'7天平均气温'作为区分节点\n")
    report_lines.append("- **其他子节点**: 使用所有特征（包括7天平均相对湿度、7天累计降雨时数、7天累计日照时数等）进行详细划分\n\n")
    report_lines.append("| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 训练样本数 | 验证样本数 | 验证准确率 | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本总数 | 总体准确率 | 综合评分 |\n")
    report_lines.append("|---------|---------|-----------|------------------|-----------|-----------|-----------|-----------|-----------|----------|------------|-----------|----------|\n")
    
    for row in summary_data:
        report_lines.append(
            f"| {row['实验编号']} | {row['参数名称']} | {row['max_depth']} | {row['min_samples_leaf']} | "
            f"{row['训练样本数']} | {row['验证样本数']} | {row['验证准确率']} | "
            f"{row['使用特征数']} | {row['最重要特征']} | {row['规则总数']} | {row['覆盖样本总数']} | "
            f"{row['总体准确率']} | {row['综合评分']} |\n"
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
        
        for rank, exp in enumerate(sorted_experiments[:3], 1):
            report_lines.append(f"#### 第{rank}名：{exp['参数名称']} (max_depth={exp['max_depth']}, min_samples_leaf={exp['min_samples_leaf']})\n\n")
            report_lines.append(f"- **综合评分**: {exp['综合评分']}\n")
            report_lines.append(f"- **总体准确率**: {exp['总体准确率']}\n")
            report_lines.append(f"- **覆盖样本总数**: {exp['覆盖样本总数']}\n")
            report_lines.append(f"- **规则总数**: {exp['规则总数']}\n")
            report_lines.append(f"- **使用特征数**: {exp['使用特征数']}\n")
            report_lines.append(f"- **最重要特征**: {exp['最重要特征']} (重要性={exp['最高重要性']})\n")
            report_lines.append(f"- **训练 / 验证样本**: {exp['训练样本数']} / {exp['验证样本数']}\n")
            report_lines.append(f"- **验证准确率(2025)**: {exp['验证准确率']}\n")
            report_lines.append(f"- **输出目录**: `{exp['输出目录']}`\n\n")
        
        report_lines.append("### 详细对比分析\n\n")
        report_lines.append("| 排名 | 参数名称 | max_depth | min_samples_leaf | 总体准确率 | 覆盖样本数 | 验证准确率 | 验证样本数 | 规则数 | 综合评分 |\n")
        report_lines.append("|------|---------|-----------|------------------|-----------|-----------|-----------|-----------|--------|----------|\n")
        
        for rank, exp in enumerate(sorted_experiments[:3], 1):
            report_lines.append(
                f"| {rank} | {exp['参数名称']} | {exp['max_depth']} | {exp['min_samples_leaf']} | "
                f"{exp['总体准确率']} | {exp['覆盖样本总数']} | {exp['验证准确率']} | {exp['验证样本数']} | {exp['规则总数']} | {exp['综合评分']} |\n"
            )
        
        report_lines.append("\n### 最优解法特点分析\n\n")
        
        # 分析最优解法的共同特点
        top1 = sorted_experiments[0]
        report_lines.append(f"**最优解法（{top1['参数名称']}）的特点：**\n\n")
        report_lines.append(f"1. **参数设置**: max_depth={top1['max_depth']}, min_samples_leaf={top1['min_samples_leaf']}\n")
        report_lines.append(f"2. **性能表现**: 在{len(valid_experiments)}个有效实验中，综合评分最高（{top1['综合评分']}）\n")
        report_lines.append(f"3. **准确率**: {top1['总体准确率']}，意味着该模型在覆盖的样本上预测准确率较高\n")
        report_lines.append(f"4. **覆盖范围**: 覆盖了{top1['覆盖样本总数']}个样本，说明规则具有良好的泛化能力\n")
        report_lines.append(f"5. **验证表现**: 2025年验证样本 {top1['验证样本数']} 条，准确率 {top1['验证准确率']}\n")
        report_lines.append(f"6. **特征选择**: 使用了{top1['使用特征数']}个特征，其中最重要特征是{top1['最重要特征']}\n")
        report_lines.append(f"7. **规则数量**: 共{top1['规则总数']}条规则，规则数量适中，既不过于复杂也不过于简单\n\n")
        
        report_lines.append("**建议使用该参数组合进行最终的预警模型构建。**\n\n")
    
    report_lines.append("\n## 详细结果\n\n")
    report_lines.append("每个实验的详细结果保存在对应的子文件夹中：\n\n")
    
    for exp in experiments:
        name = exp["name"]
        out_dir = f"analysis_{name}"
        report_lines.append(f"- **{exp['name']}** (depth={exp['depth']}, leaf={exp['leaf']}): `{out_dir}/`\n")
        report_lines.append(f"  - 特征重要性: `{out_dir}/tree_feature_importances.csv`\n")
        report_lines.append(f"  - 预警规则: `{out_dir}/tree_rules_warning_*.txt`\n")
        report_lines.append(f"  - 详细报告: `{out_dir}/1120.md`\n\n")
    
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
    parser = argparse.ArgumentParser(description="批量实验：100种树深度和叶子节点数组合（基于新预警标准）")
    
    # 使用脚本所在目录为基准，构造健壮默认值
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    default_data = os.path.join(project_root, "样本数据_验证修复.xlsx")
    default_out = os.path.join(project_root, "analysis_output_batch_1120")

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
    run_batch_experiments_1120(args.data, args.out)


if __name__ == "__main__":
    main()