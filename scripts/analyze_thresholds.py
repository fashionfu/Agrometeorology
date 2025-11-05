import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# ---------------------- Utilities ----------------------


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns if any(tok in str(c) for tok in ["日期", "时间", "日", "date", "Date"])
    ]
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
    # Prefer our generated columns if present
    if "树下感病率_标准化(%)" in df.columns and "树上感病率_标准化(%)" in df.columns:
        return "树下感病率_标准化(%)", "树上感病率_标准化(%)"

    # Otherwise, guess by tokens
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
    # Expected weather factor keywords
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
    # de-dup and preserve order
    out: List[str] = []
    for c in features:
        if c not in out:
            out.append(c)
    return out


# ---------------------- GRA (Grey Relational Analysis) ----------------------


def grey_relational_grade(X: np.ndarray, y: np.ndarray, rho: float = 0.5) -> np.ndarray:
    """
    Compute Grey Relational Grade for each column in X relative to y.
    Steps:
      1) Min-max normalize each series to [0,1].
      2) Compute absolute difference Δ_i(k) = |x_i(k) - y(k)|
      3) ξ_i(k) = (Δ_min + ρΔ_max) / (Δ_i(k) + ρΔ_max)
      4) Grade γ_i = mean_k ξ_i(k)
    """
    def minmax(a: np.ndarray) -> np.ndarray:
        a = a.astype(float)
        amin, amax = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
            return np.zeros_like(a, dtype=float)
        return (a - amin) / (amax - amin)

    y_norm = minmax(y)
    grades = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        xi_norm = minmax(xi)
        delta = np.abs(xi_norm - y_norm)
        delta_min = np.nanmin(delta)
        delta_max = np.nanmax(delta)
        denom = delta + rho * delta_max
        rel = (delta_min + rho * delta_max) / np.where(denom == 0, np.finfo(float).eps, denom)
        grades.append(np.nanmean(rel))
    return np.array(grades)


# ---------------------- Decision Tree Rule Extraction ----------------------


@dataclass
class Rule:
    path: List[str]
    samples: int
    pos_rate: float


def extract_positive_rules(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    positive_label: int = 1,
) -> List[Rule]:
    t = clf.tree_
    rules: List[Rule] = []

    def recurse(node: int, path: List[str], idx: np.ndarray):
        if t.feature[node] == -2:
            # leaf
            samples = idx.sum()
            if samples == 0:
                return
            pos = y[idx] == positive_label
            pos_rate = float(pos.sum()) / float(samples)
            if pos_rate >= 0.5:  # consider as positive leaf
                rules.append(Rule(path=list(path), samples=int(samples), pos_rate=pos_rate))
            return

        feat_idx = t.feature[node]
        threshold = t.threshold[node]
        name = feature_names[feat_idx]

        left_idx = idx & (X[:, feat_idx] <= threshold)
        right_idx = idx & (X[:, feat_idx] > threshold)

        recurse(node=t.children_left[node], path=path + [f"{name} ≤ {threshold:.3f}"], idx=left_idx)
        recurse(node=t.children_right[node], path=path + [f"{name} > {threshold:.3f}"], idx=right_idx)

    recurse(0, [], np.ones(X.shape[0], dtype=bool))
    # sort by positive rate then samples
    rules.sort(key=lambda r: (r.pos_rate, r.samples), reverse=True)
    return rules


# ---------------------- Main Pipeline ----------------------


def run_pipeline(
    factors_csv: str,
    warning_xlsx: str,
    output_dir: str,
    max_depth: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    met = pd.read_csv(factors_csv, encoding="utf-8", engine="python")
    warn = pd.read_excel(warning_xlsx, sheet_name=0)

    # Date alignment
    met_date_col = _find_date_column(met)
    warn_date_col = _find_date_column(warn)
    if met_date_col is None or warn_date_col is None:
        raise ValueError("无法识别日期列，请检查两份文件的日期列名称（含‘日期/时间/日’等关键词）")

    met[met_date_col] = _to_datetime_series(met[met_date_col])
    warn[warn_date_col] = _to_datetime_series(warn[warn_date_col])

    # Infection columns
    down_col, up_col = _pick_infection_columns(warn)
    if down_col is None or up_col is None:
        raise ValueError("未找到‘树下/树上’感病率列，请确认列名或先用生成脚本产生标准化列。")

    # Standardize percent
    warn[down_col] = warn[down_col].map(_normalize_percent)
    warn[up_col] = warn[up_col].map(_normalize_percent)

    # Join
    df = pd.merge(met, warn[[warn_date_col, down_col, up_col]], left_on=met_date_col, right_on=warn_date_col, how="inner")
    df = df.dropna(subset=[down_col, up_col])

    # Features
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("未匹配到气象因子列，请检查列名（温度/湿度/降雨/降雨时数/累计日照等关键词）。")

    # Prepare matrices
    X = df[feature_cols].astype(float).to_numpy()
    y_down = (df[down_col].astype(float) > 0.0).astype(int).to_numpy()
    y_up = (df[up_col].astype(float) > 0.0).astype(int).to_numpy()

    # (1) Pearson (point-biserial) and GRA
    def pearson_with_binary(X_: np.ndarray, y_: np.ndarray) -> np.ndarray:
        # Center both
        Xc = X_ - np.nanmean(X_, axis=0, keepdims=True)
        yc = y_ - np.nanmean(y_)
        num = np.nansum(Xc * yc.reshape(-1, 1), axis=0)
        den = np.sqrt(np.nansum(Xc**2, axis=0)) * np.sqrt(np.nansum(yc**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den == 0, 0.0, num / den)
        return r

    pear_down = pearson_with_binary(X, y_down)
    pear_up = pearson_with_binary(X, y_up)

    gra_down = grey_relational_grade(X, y_down.astype(float))
    gra_up = grey_relational_grade(X, y_up.astype(float))

    pd.DataFrame({
        "feature": feature_cols,
        "pearson": pear_down,
        "gra": gra_down,
    }).sort_values(by=["pearson"], key=lambda s: np.abs(s), ascending=False).to_csv(
        os.path.join(output_dir, "feature_scores_down.csv"), index=False, encoding="utf-8-sig"
    )

    pd.DataFrame({
        "feature": feature_cols,
        "pearson": pear_up,
        "gra": gra_up,
    }).sort_values(by=["pearson"], key=lambda s: np.abs(s), ascending=False).to_csv(
        os.path.join(output_dir, "feature_scores_up.csv"), index=False, encoding="utf-8-sig"
    )

    # (2) Decision Trees for thresholds
    def fit_tree(y_: np.ndarray) -> DecisionTreeClassifier:
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_leaf=5, random_state=42)
        # Drop rows with any NaNs
        mask = ~np.isnan(X).any(axis=1)
        clf.fit(X[mask], y_[mask])
        return clf

    clf_down = fit_tree(y_down)
    clf_up = fit_tree(y_up)

    rules_down = extract_positive_rules(clf_down, feature_cols, X, y_down)
    rules_up = extract_positive_rules(clf_up, feature_cols, X, y_up)

    def save_rules(rules: List[Rule], path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("正类（感染率>0）规则，按阳性比例与覆盖样本排序\n\n")
            for i, r in enumerate(rules, 1):
                f.write(f"规则{i}: 覆盖样本={r.samples}, 阳性比例={r.pos_rate:.3f}\n")
                if r.path:
                    f.write("  条件: " + " 且 ".join(r.path) + "\n")
                else:
                    f.write("  条件: (无条件，整体为正类)\n")
                f.write("\n")

    save_rules(rules_down, os.path.join(output_dir, "tree_rules_down.txt"))
    save_rules(rules_up, os.path.join(output_dir, "tree_rules_up.txt"))

    # Also export feature importances for reference
    pd.DataFrame({
        "feature": feature_cols,
        "importance_down": clf_down.feature_importances_,
        "importance_up": clf_up.feature_importances_,
    }).to_csv(os.path.join(output_dir, "tree_feature_importances.csv"), index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="气象因子阈值分析：皮尔逊/灰色关联 + 决策树阈值")
    parser.add_argument("--factors", default="metadata/影响因子1103_final.csv", help="气象因子CSV文件路径")
    parser.add_argument(
        "--warning",
        default="metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx",
        help="包含标准化感病率与预警等级的Excel",
    )
    parser.add_argument("--out", default="analysis_outputs", help="输出目录")
    parser.add_argument("--max_depth", type=int, default=3, help="决策树最大深度")
    args = parser.parse_args()

    run_pipeline(args.factors, args.warning, args.out, max_depth=args.max_depth)
    print(f"已输出到目录: {args.out}")


if __name__ == "__main__":
    main()


