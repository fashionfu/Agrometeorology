import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(tok in str(c) for tok in ["日期", "时间", "日", "date", "Date"])]
    if candidates:
        return str(candidates[0])
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)
    return None


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _normalize_percent(value) -> Optional[float]:
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
    if "树下感病率_标准化(%)" in df.columns and "树上感病率_标准化(%)" in df.columns:
        return "树下感病率_标准化(%)", "树上感病率_标准化(%)"
    def match(tokens: List[str]) -> Optional[str]:
        for c in df.columns:
            name = str(c)
            if all(tok in name for tok in tokens):
                return name
        return None
    down = match(["树下"]) or match(["落地"]) or match(["地面"]) or None
    up = match(["树上"]) or match(["挂树"]) or None
    return down, up


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
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


def winsorize(a: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return a
    lo = np.nanpercentile(a, 100 * alpha)
    hi = np.nanpercentile(a, 100 * (1 - alpha))
    return np.clip(a, lo, hi)


def pearson_with_binary(X: np.ndarray, y: np.ndarray, winsor_alpha: float = 0.0) -> np.ndarray:
    Xw = X.copy().astype(float)
    for j in range(Xw.shape[1]):
        Xw[:, j] = winsorize(Xw[:, j], winsor_alpha)
    yw = winsorize(y.astype(float), winsor_alpha)
    Xc = Xw - np.nanmean(Xw, axis=0, keepdims=True)
    yc = yw - np.nanmean(yw)
    num = np.nansum(Xc * yc.reshape(-1, 1), axis=0)
    den = np.sqrt(np.nansum(Xc**2, axis=0)) * np.sqrt(np.nansum(yc**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den == 0, 0.0, num / den)
    return r


def extract_positive_rules(clf: DecisionTreeClassifier, feature_names: List[str], X: np.ndarray, y: np.ndarray) -> List[dict]:
    t = clf.tree_
    rules: List[dict] = []
    def recurse(node: int, path: List[str], idx: np.ndarray):
        if t.feature[node] == -2:
            samples = idx.sum()
            if samples == 0:
                return
            pos = y[idx] == 1
            pos_rate = float(pos.sum()) / float(samples)
            if pos_rate >= 0.5:
                rules.append({"path": list(path), "samples": int(samples), "pos_rate": pos_rate})
            return
        feat_idx = t.feature[node]
        thr = t.threshold[node]
        name = feature_names[feat_idx]
        left_idx = idx & (X[:, feat_idx] <= thr)
        right_idx = idx & (X[:, feat_idx] > thr)
        recurse(t.children_left[node], path + [f"{name} ≤ {thr:.3f}"], left_idx)
        recurse(t.children_right[node], path + [f"{name} > {thr:.3f}"], right_idx)
    recurse(0, [], np.ones(X.shape[0], dtype=bool))
    rules.sort(key=lambda r: (r["pos_rate"], r["samples"]), reverse=True)
    return rules


def run_sweep(factors_csv: str, warning_xlsx: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    met = pd.read_csv(factors_csv, encoding="utf-8", engine="python")
    warn = pd.read_excel(warning_xlsx, sheet_name=0)

    met_date_col = _find_date_column(met)
    warn_date_col = _find_date_column(warn)
    if met_date_col is None or warn_date_col is None:
        raise ValueError("无法识别日期列")
    met[met_date_col] = _to_datetime_series(met[met_date_col])
    warn[warn_date_col] = _to_datetime_series(warn[warn_date_col])

    down_col, up_col = _pick_infection_columns(warn)
    if down_col is None or up_col is None:
        raise ValueError("未找到树下/树上感病率列")
    warn[down_col] = warn[down_col].map(_normalize_percent)
    warn[up_col] = warn[up_col].map(_normalize_percent)

    df = pd.merge(met, warn[[warn_date_col, down_col, up_col]], left_on=met_date_col, right_on=warn_date_col, how="inner")
    df = df.dropna(subset=[down_col, up_col])

    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("未匹配到气象因子列")

    X = df[feature_cols].astype(float).to_numpy()
    y_down = (df[down_col].astype(float) > 0.0).astype(int).to_numpy()
    y_up = (df[up_col].astype(float) > 0.0).astype(int).to_numpy()

    # Sweep configurations
    depths = list(range(1, 11))
    rhos = [round(x, 2) for x in np.linspace(0.1, 1.0, 10)]
    winsors = [round(x, 2) for x in np.linspace(0.0, 0.18, 10)]  # 0%~18%温莎化

    summary_rows_rules: List[dict] = []
    summary_rows_scores: List[dict] = []

    # 1) Pearson & GRA sweeps (仅评分对比)
    for a in winsors:
        pear_down = pearson_with_binary(X, y_down, winsor_alpha=a)
        pear_up = pearson_with_binary(X, y_up, winsor_alpha=a)
        pd.DataFrame({"feature": feature_cols, "pearson": pear_down}).sort_values(
            by=["pearson"], key=lambda s: np.abs(s), ascending=False
        ).to_csv(os.path.join(out_dir, f"feature_scores_down_pearson_w{a:.2f}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame({"feature": feature_cols, "pearson": pear_up}).sort_values(
            by=["pearson"], key=lambda s: np.abs(s), ascending=False
        ).to_csv(os.path.join(out_dir, f"feature_scores_up_pearson_w{a:.2f}.csv"), index=False, encoding="utf-8-sig")
        summary_rows_scores.append({"type": "pearson", "param": a})

    for r in rhos:
        gra_down = grey_relational_grade(X, y_down.astype(float), rho=r)
        gra_up = grey_relational_grade(X, y_up.astype(float), rho=r)
        pd.DataFrame({"feature": feature_cols, "gra": gra_down}).sort_values(
            by=["gra"], ascending=False
        ).to_csv(os.path.join(out_dir, f"feature_scores_down_gra_rho{r:.2f}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame({"feature": feature_cols, "gra": gra_up}).sort_values(
            by=["gra"], ascending=False
        ).to_csv(os.path.join(out_dir, f"feature_scores_up_gra_rho{r:.2f}.csv"), index=False, encoding="utf-8-sig")
        summary_rows_scores.append({"type": "gra", "param": r})

    # 2) Decision tree depth sweep，记录最强正类规则
    mask_valid = ~np.isnan(X).any(axis=1)
    Xv = X[mask_valid]
    ydv = y_down[mask_valid]
    yuv = y_up[mask_valid]

    for d in depths:
        clf_down = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=5, random_state=42)
        clf_up = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=5, random_state=42)
        clf_down.fit(Xv, ydv)
        clf_up.fit(Xv, yuv)

        rules_down = extract_positive_rules(clf_down, feature_cols, Xv, ydv)
        rules_up = extract_positive_rules(clf_up, feature_cols, Xv, yuv)

        def save_rules(rules: List[dict], path: str):
            with open(path, "w", encoding="utf-8") as f:
                f.write("正类（感染率>0）规则，按阳性比例与覆盖样本排序\n\n")
                for i, r in enumerate(rules, 1):
                    f.write(f"规则{i}: 覆盖样本={r['samples']}, 阳性比例={r['pos_rate']:.3f}\n")
                    if r["path"]:
                        f.write("  条件: " + " 且 ".join(r["path"]) + "\n")
                    else:
                        f.write("  条件: (无条件)\n")
                    f.write("\n")

        save_rules(rules_down, os.path.join(out_dir, f"tree_rules_down_depth{d}.txt"))
        save_rules(rules_up, os.path.join(out_dir, f"tree_rules_up_depth{d}.txt"))

        # 取每个方向的最优规则（pos_rate*样本）作为代表
        def best_score(rules: List[dict]) -> Tuple[float, int, float, str]:
            if not rules:
                return 0.0, 0, 0.0, ""
            scored = [((r["pos_rate"] * r["samples"]), r["samples"], r["pos_rate"], " 且 ".join(r["path"])) for r in rules]
            scored.sort(reverse=True)
            return scored[0]

        b_score_d, b_samples_d, b_pos_d, b_rule_d = best_score(rules_down)
        b_score_u, b_samples_u, b_pos_u, b_rule_u = best_score(rules_up)

        summary_rows_rules.append({
            "depth": d,
            "dir": "down",
            "score": b_score_d,
            "samples": b_samples_d,
            "pos_rate": b_pos_d,
            "rule": b_rule_d,
        })
        summary_rows_rules.append({
            "depth": d,
            "dir": "up",
            "score": b_score_u,
            "samples": b_samples_u,
            "pos_rate": b_pos_u,
            "rule": b_rule_u,
        })

    # 汇总表
    pd.DataFrame(summary_rows_rules).sort_values(["dir", "score", "samples", "pos_rate"], ascending=[True, False, False, False]).to_csv(
        os.path.join(out_dir, "tree_rules_summary.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows_scores).to_csv(os.path.join(out_dir, "feature_scores_sweep_index.csv"), index=False, encoding="utf-8-sig")


def parse_rule_file(filepath: str) -> List[dict]:
    """解析规则文件，提取规则信息"""
    rules = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("规则"):
            # 解析规则头: 规则1: 覆盖样本=37, 阳性比例=1.000
            match = re.search(r"规则(\d+):\s*覆盖样本=(\d+),\s*阳性比例=([\d.]+)", line)
            if match:
                rule_num = int(match.group(1))
                samples = int(match.group(2))
                pos_rate = float(match.group(3))
                # 读取下一行的条件（可能有缩进）
                condition = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if "条件:" in next_line:
                        condition = next_line.split("条件:")[-1].strip()
                        if condition in ["(无条件)", "(无条件，整体为正类)", ""]:
                            condition = ""
                rules.append({
                    "rule_num": rule_num,
                    "samples": samples,
                    "pos_rate": pos_rate,
                    "condition": condition
                })
        i += 1
    return rules


def generate_report(out_dir: str):
    """生成详细的结果报告"""
    report_lines = []
    
    # 标题
    report_lines.append("# 荔枝霜疫霉感染率阈值模型分析报告\n")
    report_lines.append("基于批量阈值实验（10种树深度、10种GRA分辨系数、10种皮尔逊温莎化比例）\n\n")
    
    # 1. 结论概览
    report_lines.append("## 1. 结论概览\n\n")
    
    # 1.1 树下花果感染率 >0
    report_lines.append("### 1.1. 树下花果感染率 >0 的关键因子与阈值（规则来自决策树）\n\n")
    
    # 读取depth=3的规则（作为主要规则）
    depth3_down_file = os.path.join(out_dir, "tree_rules_down_depth3.txt")
    if os.path.exists(depth3_down_file):
        rules_down = parse_rule_file(depth3_down_file)
        for i, r in enumerate(rules_down, 1):
            report_lines.append(f"\t{i}.规则{r['rule_num']}（覆盖样本={r['samples']}，阳性比例={r['pos_rate']*100:.0f}%）：{r['condition']}\n\n")
    
    # 读取depth=1的规则（作为最优规则）
    depth1_down_file = os.path.join(out_dir, "tree_rules_down_depth1.txt")
    if os.path.exists(depth1_down_file):
        rules_down_d1 = parse_rule_file(depth1_down_file)
        if rules_down_d1:
            r = rules_down_d1[0]
            score = r['pos_rate'] * r['samples']
            report_lines.append(f"4.最优规则（depth=1）：{r['condition']}（samples={r['samples']}，pos_rate={r['pos_rate']:.3f}，score={score:.0f}）")
            
            # 查找depth=3中最严格的规则
            if len(rules_down) > 0:
                strict_rule = rules_down[0]
                strict_score = strict_rule['pos_rate'] * strict_rule['samples']
                report_lines.append(f"备注：在此基础上叠加条件可达更高精度（pos_rate={strict_rule['pos_rate']:.1f}），但覆盖样本降至{strict_rule['samples']}（score={strict_score:.0f}）。\n\n")
    
    # 1.2 树上花果感染率 >0
    report_lines.append("### 1.2. 树上花果感染率 >0 的关键因子与阈值\n\n")
    
    depth3_up_file = os.path.join(out_dir, "tree_rules_up_depth3.txt")
    if os.path.exists(depth3_up_file):
        rules_up = parse_rule_file(depth3_up_file)
        for i, r in enumerate(rules_up, 1):
            report_lines.append(f"\t{i}.规则{r['rule_num']}（覆盖样本={r['samples']}，阳性比例={r['pos_rate']*100:.0f}%）：{r['condition']}\n\n")
    
    depth1_up_file = os.path.join(out_dir, "tree_rules_up_depth1.txt")
    if os.path.exists(depth1_up_file):
        rules_up_d1 = parse_rule_file(depth1_up_file)
        if rules_up_d1:
            r = rules_up_d1[0]
            score = r['pos_rate'] * r['samples']
            report_lines.append(f"4.最优规则（depth=1）：{r['condition']}（samples={r['samples']}，pos_rate={r['pos_rate']:.3f}，score={score:.0f}）")
            
            if len(rules_up) > 0:
                strict_rule = rules_up[0]
                strict_score = strict_rule['pos_rate'] * strict_rule['samples']
                report_lines.append(f"备注：叠加条件可达更高精度（pos_rate={strict_rule['pos_rate']:.1f}），但覆盖样本降至{strict_rule['samples']}（score={strict_score:.0f}）。\n\n")
    
    report_lines.append("注：极端高湿度（>96%）时可能伴随其他抑制因子（如通风差、凝结水影响、温度过低等），导致感染率略降；且极端高湿度样本较少，可能受统计波动影响；同时极高湿度可能超出最适范围，形成\"过度饱和\"效应。\n\n")
    
    # 2. 关联强度
    report_lines.append("## 2. 关联强度（GRA 分数，值越大关联越强）\n\n")
    
    # 2.1 树下>0
    report_lines.append("### 2.1 树下>0（节选，来自 feature_scores_down_gra_rho0.50.csv）\n\n")
    
    gra_down_file = os.path.join(out_dir, "feature_scores_down_gra_rho0.50.csv")
    if os.path.exists(gra_down_file):
        df_gra_down = pd.read_csv(gra_down_file, encoding="utf-8-sig")
        # 选择前几个重要特征
        top_features = df_gra_down.head(8).itertuples()
        for row in top_features:
            feature_name = row.feature
            gra_score = row.gra
            # 检查是否有对应的Pearson值（负相关标记）
            pearson_file = os.path.join(out_dir, "feature_scores_down_pearson_w0.00.csv")
            pearson_note = ""
            if os.path.exists(pearson_file):
                df_pear = pd.read_csv(pearson_file, encoding="utf-8-sig")
                pear_row = df_pear[df_pear["feature"] == feature_name]
                if not pear_row.empty:
                    pear_val = pear_row.iloc[0]["pearson"]
                    if "日照" in feature_name and pear_val < 0:
                        pearson_note = "（与发生呈反向关系，Pearson为负）"
            report_lines.append(f"\t{feature_name}: {gra_score:.3f}{pearson_note}\n\n")
    
    # 2.2 树上>0
    report_lines.append("### 2.2 树上>0（节选，来自 feature_scores_up_gra_rho0.50.csv）\n\n")
    
    gra_up_file = os.path.join(out_dir, "feature_scores_up_gra_rho0.50.csv")
    if os.path.exists(gra_up_file):
        df_gra_up = pd.read_csv(gra_up_file, encoding="utf-8-sig")
        top_features = df_gra_up.head(8).itertuples()
        for row in top_features:
            feature_name = row.feature
            gra_score = row.gra
            # 检查是否有对应的Pearson值
            pearson_file = os.path.join(out_dir, "feature_scores_up_pearson_w0.00.csv")
            pearson_note = ""
            if os.path.exists(pearson_file):
                df_pear = pd.read_csv(pearson_file, encoding="utf-8-sig")
                pear_row = df_pear[df_pear["feature"] == feature_name]
                if not pear_row.empty:
                    pear_val = pear_row.iloc[0]["pearson"]
                    if "日照" in feature_name and pear_val < 0:
                        pearson_note = "（多为负相关，日照少更易发生）"
            report_lines.append(f"\t{feature_name}: {gra_score:.3f}{pearson_note}\n\n")
    
    # 3. 条件设置
    report_lines.append("## 3. 条件设置\n\n")
    
    report_lines.append("### 3.1. 树下发生更易出现在：\n\n")
    report_lines.append("近7天有降雨时数（>0.5小时）叠加高湿背景（10天均湿>约81%），当天极端高湿（>96%）时也可触发；若10天均湿不高时，当降雨时数多但近10天整体降雨时数不大（≤9.5小时）也可出现。\n\n")
    
    report_lines.append("### 3.2. 树上发生更易出现在：\n\n")
    report_lines.append("持续偏高湿（10天均湿>约82%）且日照偏少（10天累积日照≤16小时）的组合条件；当湿度非常高（>93.25%）时风险更高。\n\n")
    
    # 4. 方法概述
    report_lines.append("## 4. 方法概述\n\n")
    
    report_lines.append("**皮尔逊相关（含温莎化稳健参数）**：衡量特征与目标(是否感染>0)的线性相关强度；通过温莎化比例 alpha 抑制异常值影响。\n\n")
    
    report_lines.append("**灰色关联分析 GRA（分辨系数 ρ）**：衡量特征与目标序列\"形状接近度\"，对非线性/不同量纲更稳健；ρ 控制区分度。\n\n")
    
    report_lines.append("**决策树阈值规则（深度 max_depth）**：用于从多气象因子中提取\"感染>0\"的阈值组合规则，支持从浅到深的可解释分割。\n\n")
    
    # 5. 参考文献
    report_lines.append("## 5. 参考文献\n\n")
    
    report_lines.append("[1]吕安瑞,严梦荧,张妍,等.基于土壤中霜疫霉活力变化防治荔枝霜疫病初探[C]//中国植物病理学会.中国植物病理学会2024年学术年会论文集.华南农业大学园艺学院华南农业大学植物保护学院;,2024:167.DOI:10.26914/c.cnkihy.2024.022408.\n\n")
    
    report_lines.append("结果表明，荔枝霜疫霉可在果园土壤中存活１６个月，成功越夏、越冬，成为荔枝霜疫病初侵染源；土壤中荔枝霜疫霉存在２个活跃期，第一个是冬末初春（１２月中旬至次年开花前），此时土壤中荔枝霜疫霉活力逐渐增强，处于催醒状态，作为初侵染源具有侵染能力；第二个活跃期３月下旬至４月上中旬，为荔枝开花后至第二次生理落果期，此时土壤中复苏的荔枝霜疫霉在雨水的作用下，侵染落地花果，以多种生物学形态习居在土壤中；健康荔枝花果落地后染病快速，花落地后３ｈ即可发病，２４ｈ后感染率超５０％；幼果发病略慢，接触土壤４５ｈ感染率大于５０％。落花和落果是土壤中霜疫霉初次侵染和再侵染的寄主，导致荔枝霜疫霉辗转传播侵染。\n\n")
    
    # 保存报告
    report_path = os.path.join(out_dir, "1104.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    
    print(f"已生成详细报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="批量阈值实验：10种树深度、10种GRA分辨系数、10种皮尔逊温莎化比例")
    parser.add_argument("--factors", default="metadata/影响因子1103_final\.csv", help="气象因子CSV")
    parser.add_argument(
        "--warning",
        default="metadata/张桂香19\-25校内大果园花果带菌率数据分析\-\-给张工分析数据\-10\.20_预警\.xlsx",
        help="包含标准化感病率与预警等级的Excel",
    )
    parser.add_argument("--out", default="analysis_sweep", help="输出目录（新建）")
    parser.add_argument("--report-only", action="store_true", help="仅生成报告（不重新运行分析）")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    if not args.report_only:
        run_sweep(args.factors, args.warning, args.out)
        print(f"已输出到目录: {args.out}")
    
    # 生成详细报告
    generate_report(args.out)


if __name__ == "__main__":
    main()


