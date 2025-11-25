#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
荔枝霜疫霉感染率阈值模型分析（单次分析）
- 读取影响因子xlsx和原始感染率数据
- 排除预警为"未定义"的样本
- 分析感染率>0的气象因子阈值
"""
import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """识别日期列"""
    candidates = [c for c in df.columns if any(tok in str(c) for tok in ["日期", "时间", "日", "date", "Date"])]
    if candidates:
        return str(candidates[0])
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)
    return None


def _to_datetime_series(s: pd.Series) -> pd.Series:
    """转换为日期序列"""
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _normalize_percent(value) -> Optional[float]:
    """标准化百分比值"""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        has_percent = "%" in s
        s_num = re.sub(r"[^0-9.\-]", "", s)
        if s_num in ("", ".", "-", "-."):
            return None
        try:
            num = float(s_num)
        except ValueError:
            return None
        if has_percent:
            return max(0.0, min(100.0, num))
        if 0 <= num <= 1:
            return num * 100.0
        return max(0.0, min(100.0, num))
    try:
        num = float(value)
    except Exception:
        return None
    if 0 <= num <= 1:
        return num * 100.0
    return max(0.0, min(100.0, num))


def _pick_infection_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """识别树下和树上感病率列"""
    def match(tokens: List[str]) -> Optional[str]:
        for c in df.columns:
            name = str(c)
            if all(tok in name for tok in tokens):
                return name
        return None
    down = match(["树下"]) or match(["落地"]) or match(["地面"]) or None
    up = match(["树上"]) or match(["挂树"]) or None
    return down, up


def _find_warning_column(df: pd.DataFrame) -> Optional[str]:
    """识别预警列（优先找"预警"，不是"预警等级"）"""
    # 优先找精确匹配"预警"的列（不包含"等级"）
    exact_match = [c for c in df.columns if str(c).strip() == "预警"]
    if exact_match:
        return str(exact_match[0])
    # 其次找包含"预警"但不包含"等级"的列
    candidates = [c for c in df.columns 
                  if "预警" in str(c) and "等级" not in str(c)]
    if candidates:
        return str(candidates[0])
    # 最后找包含"预警"的列
    candidates = [c for c in df.columns if "预警" in str(c)]
    if candidates:
        return str(candidates[0])
    return None


def _normalize_warning_level(value) -> Optional[int]:
    """将预警值转换为数值：0, 1, 2, 3"""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    # 直接是数字 0, 1, 2, 3
    if s.isdigit():
        val = int(s)
        if 0 <= val <= 3:
            return val
    # 文字描述匹配（优先级从高到低）
    # 3级（重度）
    if "重度" in s or "3" in s or "易发生" in s:
        return 3
    # 2级（中度）
    if "中度" in s or "2" in s or "较易" in s:
        return 2
    # 1级（轻度）
    if "轻度" in s or "1" in s or "不易" in s:
        return 1
    # 0级（不发生）
    if "0" in s or "不发生" in s or "无" in s:
        return 0
    # 未定义
    return None


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """选择气象因子列"""
    patterns = [
        ("日均温度", ["温度", "日均温度", "Tmean", "日平均气温", "平均温度", "平均气温", "气温"]),
        ("日均湿度", ["湿度", "日均湿度", "RH", "平均相对湿度"]),
        ("日均降雨量", ["降雨", "降雨量", "日降雨", "日均降雨", "降水", "降水量", "雨量", "Rain", "Precip"]),
        ("累计降雨量", ["累计降雨量", "累计降水", "累积降雨", "累积降水", "累积雨量", "RainCum", "PrecipCum"]),
        ("累计降雨时数", ["降雨时数", "降水时数", "累计降雨时数", "累计降水时数", "RainHours", "PrecipHours"]),
        ("累计日照", ["日照", "累计日照", "日照时数", "Sunshine", "Solar", "Radiation"]),
    ]
    features: List[str] = []
    for _, keys in patterns:
        for c in df.columns:
            name = str(c)
            if any(k in name for k in keys):
                features.append(name)
    out: List[str] = []
    for c in features:
        if c not in out:
            out.append(c)
    return out


def grey_relational_grade(X: np.ndarray, y: np.ndarray, rho: float = 0.5) -> np.ndarray:
    """灰色关联分析"""
    def minmax(a: np.ndarray) -> np.ndarray:
        a = a.astype(float)
        amin, amax = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
            return np.zeros_like(a, dtype=float)
        return (a - amin) / (amax - amin)
    y_norm = minmax(y)
    grades = []
    for i in range(X.shape[1]):
        xi_norm = minmax(X[:, i])
        delta = np.abs(xi_norm - y_norm)
        delta_min = np.nanmin(delta)
        delta_max = np.nanmax(delta)
        denom = delta + rho * delta_max
        rel = (delta_min + rho * delta_max) / np.where(denom == 0, np.finfo(float).eps, denom)
        grades.append(np.nanmean(rel))
    return np.array(grades)


def pearson_with_binary(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """皮尔逊相关（点二列相关）"""
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    yc = y - np.nanmean(y)
    num = np.nansum(Xc * yc.reshape(-1, 1), axis=0)
    den = np.sqrt(np.nansum(Xc**2, axis=0)) * np.sqrt(np.nansum(yc**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den == 0, 0.0, num / den)
    return r


def extract_warning_rules(clf: DecisionTreeClassifier, feature_names: List[str], X: np.ndarray, y: np.ndarray, warning_labels: Dict[int, str]) -> Dict[int, List[dict]]:
    """提取每个预警等级的阈值规则"""
    t = clf.tree_
    all_rules: Dict[int, List[dict]] = {level: [] for level in warning_labels.keys()}
    
    def recurse(node: int, path: List[str], idx: np.ndarray):
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


def run_analysis(data_xlsx: str, out_dir: str, max_depth: int = 3, min_samples_leaf: int = 5):
    """运行完整分析流程（使用单一数据文件）"""
    os.makedirs(out_dir, exist_ok=True)
    
    print("正在读取数据文件...")
    # 读取包含所有数据的Excel文件
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
    
    # 检查日期转换结果
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
    before_count = len(df)
    df = df[df[warning_col].astype(str).str.strip() != "未定义"]
    after_count = len(df)
    print(f"排除预警'未定义'样本: {before_count} -> {after_count} (排除 {before_count - after_count} 条)")
    
    # 转换预警值为数值
    df["预警数值"] = df[warning_col].map(_normalize_warning_level)
    before_drop = len(df)
    df = df.dropna(subset=["预警数值"])
    after_drop = len(df)
    
    print(f"\n=== 样本数量追踪 ===")
    print(f"1. 读取后（未转换预警值）: {before_drop} 条")
    print(f"2. 转换预警值后: {before_drop} -> {after_drop} 条（删除 {before_drop - after_drop} 条）")
    
    print(f"有效数据: {len(df)} 行")
    
    # 统计预警等级分布
    warning_counts = df["预警数值"].value_counts().sort_index()
    print("\n预警等级分布:")
    warning_labels = {0: "0级（不发生）", 1: "1级（轻度）", 2: "2级（中度）", 3: "3级（重度）"}
    for level, count in warning_counts.items():
        label = warning_labels.get(int(level), f"{level}级")
        print(f"  {label}: {count} 条")
    
    # 选择特征列
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError(f"未匹配到气象因子列。可用列: {list(df.columns)}")
    
    print(f"\n识别到 {len(feature_cols)} 个气象因子特征:")
    for f in feature_cols:
        print(f"  - {f}")
    
    # 检查异常值
    print(f"\n=== 异常值检查 ===")
    outlier_info = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        # 检查负值
        negatives = df[df[col] < 0]
        if len(negatives) > 0:
            outlier_info[col] = {
                "负值数量": len(negatives),
                "最小负值": float(negatives[col].min())
            }
    
    if outlier_info:
        print("以下异常值需要处理：")
        for col, info in outlier_info.items():
            print(f"\n{col}:")
            print(f"  负值数量: {info['负值数量']}")
            print(f"  最小负值: {info['最小负值']}")
        
        # 处理异常值
        for col in feature_cols:
            if col in df.columns and col in outlier_info:
                # 处理明显的填充值（如-9999）
                if "相对湿度" in col or "湿度" in col:
                    df[col] = df[col].replace([-9999, -9999.0, -9999.00], np.nan)
                    df[col] = df[col].clip(lower=0, upper=100)
                elif "气温" in col or "温度" in col:
                    df[col] = df[col].clip(lower=-50, upper=50)
                elif "雨量" in col or "降雨" in col:
                    df[col] = df[col].clip(lower=0, upper=1000)
                elif "日照" in col or "时数" in col:
                    df[col] = df[col].clip(lower=0, upper=None)
        print("\n✓ 已处理异常值")
    else:
        print("✓ 未发现异常值")
    
    # 检查缺失值
    print(f"\n=== 气象因子缺失情况 ===")
    missing_info = {}
    for idx, row in df.iterrows():
        date = row[date_col]
        missing_features = []
        for col in feature_cols:
            if col in row and (pd.isna(row[col]) or (isinstance(row[col], (int, float)) and np.isnan(row[col]))):
                missing_features.append(col)
        if missing_features:
            missing_info[date] = missing_features
    
    if missing_info:
        print(f"发现 {len(missing_info)} 个日期存在气象因子缺失：")
        for date, missing_features in list(missing_info.items())[:20]:  # 只显示前20个
            print(f"\n日期: {date}")
            print(f"  缺失的特征: {', '.join(missing_features)}")
        if len(missing_info) > 20:
            print(f"\n... 还有 {len(missing_info) - 20} 个日期存在缺失")
        
        # 保存到文件
        missing_report = []
        for date, missing_features in missing_info.items():
            missing_report.append({
                "日期": date,
                "缺失特征": ", ".join(missing_features),
                "缺失数量": len(missing_features)
            })
        df_missing = pd.DataFrame(missing_report)
        missing_file = os.path.join(out_dir, "缺失数据日期清单.csv")
        df_missing.to_csv(missing_file, index=False, encoding="utf-8-sig")
        print(f"\n已保存缺失数据日期清单: {missing_file}")
    else:
        print("✓ 所有日期的气象因子数据完整")
    
    # 准备特征和标签
    X = df[feature_cols].astype(float).to_numpy()
    y_warning = df["预警数值"].astype(int).to_numpy()
    
    # 删除特征中有NaN的行
    before_valid = len(df)
    mask_valid = ~np.isnan(X).any(axis=1)
    Xv = X[mask_valid]
    yw = y_warning[mask_valid]
    after_valid = mask_valid.sum()
    
    print(f"\n3. 特征缺失处理后: {before_valid} -> {after_valid} 条（删除 {before_valid - after_valid} 条）")
    
    # 检查哪些特征导致样本被删除
    if before_valid > after_valid:
        missing_by_feature = {}
        for col in feature_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_by_feature[col] = missing_count
        if missing_by_feature:
            print(f"   特征缺失情况:")
            for col, count in missing_by_feature.items():
                print(f"     {col}: {count} 个缺失值")
    
    print(f"有效样本（无缺失特征）: {mask_valid.sum()} / {len(mask_valid)}")
    
    # 1. 特征筛选：对每个预警等级分别进行关联分析
    print("\n=== 1. 特征筛选分析 ===")
    
    # 对每个预警等级（1,2,3）进行二分类分析
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
    
    # 保存特征评分（对每个预警等级）
    for level in feature_scores:
        df_score = pd.DataFrame({
            "feature": feature_cols,
            "pearson": feature_scores[level]["pearson"],
            "gra": feature_scores[level]["gra"],
        })
        # 综合排序：考虑Pearson和GRA
        df_score["综合评分"] = np.abs(df_score["pearson"]) * 0.5 + df_score["gra"] * 0.5
        df_score = df_score.sort_values(by=["综合评分"], ascending=False)
        
        df_score.to_csv(
            os.path.join(out_dir, f"feature_scores_warning_{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        print(f"已保存预警{level}级特征评分: feature_scores_warning_{level}.csv")
    
    # 2. 决策树阈值分析（多分类）
    print("\n=== 2. 决策树阈值分析（使用预警等级作为目标变量）===")
    
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    clf.fit(Xv, yw)
    
    # 提取每个预警等级的规则
    all_rules = extract_warning_rules(clf, feature_cols, Xv, yw, warning_labels)
    
    # 保存规则
    def save_warning_rules(rules_by_level: Dict[int, List[dict]], out_dir: str):
        for level, rules in rules_by_level.items():
            if not rules:
                continue
            label = warning_labels.get(level, f"{level}级")
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
    
    # 返回结果用于生成报告
    return {
        "df": df,
        "feature_cols": feature_cols,
        "rules_by_level": all_rules,
        "feature_scores": feature_scores,
        "warning_labels": warning_labels,
        "stats": {
            "total_samples": len(df),
            "valid_samples": mask_valid.sum(),
            "warning_distribution": dict(warning_counts),
        }
    }


def generate_report(results: dict, out_dir: str, max_depth: int = 3, min_samples_leaf: int = 5):
    """生成详细分析报告"""
    report_lines = []
    
    report_lines.append("# 荔枝霜疫霉预警等级阈值模型分析报告\n\n")
    report_lines.append("## 数据说明\n\n")
    report_lines.append("- **数据文件**: 样本数据.xlsx（包含19-25年采集的数据）\n")
    report_lines.append("- **目标变量**: 预警列（0-3级预警等级）\n")
    report_lines.append("- **样本筛选**: 排除预警为\"未定义\"的样本（当天无数据采集）\n")
    report_lines.append(f"- **有效样本数**: {results['stats']['valid_samples']} 条\n\n")
    
    # 预警等级分布
    report_lines.append("### 预警等级分布\n\n")
    warning_labels = results["warning_labels"]
    warning_dist = results["stats"]["warning_distribution"]
    for level, count in sorted(warning_dist.items()):
        label = warning_labels.get(int(level), f"{level}级")
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
    
    report_lines.append("**皮尔逊相关（点二列相关）**：对每个预警等级（1,2,3）分别进行二分类分析，衡量特征与该预警等级的相关强度。\n\n")
    
    report_lines.append("**灰色关联分析 GRA（分辨系数 ρ=0.5）**：衡量特征与目标序列\"形状接近度\"，对非线性/不同量纲更稳健。\n\n")
    
    report_lines.append(f"**决策树多分类（深度 max_depth={max_depth}, min_samples_leaf={min_samples_leaf}）**：使用预警等级（0-3级）作为目标变量，从多气象因子中提取每个预警等级的阈值组合规则。\n\n")
    
    # 5. 参考文献
    report_lines.append("## 5. 参考文献\n\n")
    
    report_lines.append("[1]吕安瑞,严梦荧,张妍,等.基于土壤中霜疫霉活力变化防治荔枝霜疫病初探[C]//中国植物病理学会.中国植物病理学会2024年学术年会论文集.华南农业大学园艺学院华南农业大学植物保护学院;,2024:167.DOI:10.26914/c.cnkihy.2024.022408.\n\n")
    
    report_lines.append("结果表明，荔枝霜疫霉可在果园土壤中存活１６个月，成功越夏、越冬，成为荔枝霜疫病初侵染源；土壤中荔枝霜疫霉存在２个活跃期，第一个是冬末初春（１２月中旬至次年开花前），此时土壤中荔枝霜疫霉活力逐渐增强，处于催醒状态，作为初侵染源具有侵染能力；第二个活跃期３月下旬至４月上中旬，为荔枝开花后至第二次生理落果期，此时土壤中复苏的荔枝霜疫霉在雨水的作用下，侵染落地花果，以多种生物学形态习居在土壤中；健康荔枝花果落地后染病快速，花落地后３ｈ即可发病，２４ｈ后感染率超５０％；幼果发病略慢，接触土壤４５ｈ感染率大于５０％。落花和落果是土壤中霜疫霉初次侵染和再侵染的寄主，导致荔枝霜疫霉辗转传播侵染。\n\n")
    
    # 保存报告
    report_path = os.path.join(out_dir, "1104.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    
    print(f"\n已生成详细报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="荔枝霜疫霉感染率阈值模型分析（单次分析）")
    parser.add_argument(
        "--data",
        default="样本数据.xlsx",
        help="包含预警和气象因子的Excel文件路径"
    )
    parser.add_argument("--out", default="analysis_output", help="输出目录")
    parser.add_argument("--max-depth", type=int, default=4, help="决策树最大深度")
    parser.add_argument("--min-samples-leaf", type=int, default=3, help="决策树最小叶子节点样本数")
    args = parser.parse_args()
    
    print("=" * 60)
    print("荔枝霜疫霉感染率阈值模型分析")
    print("=" * 60)
    
    results = run_analysis(args.data, args.out, args.max_depth, args.min_samples_leaf)
    generate_report(results, args.out, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    
    print("\n" + "=" * 60)
    print(f"分析完成！结果已保存到: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()

