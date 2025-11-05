import argparse
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd


def _normalize_rate(value) -> Optional[float]:
    """
    Convert various percentage formats to a float in [0, 100].
    Accepts:
      - numeric in [0, 1] (interpreted as fraction) -> multiply by 100
      - numeric in [0, 100] -> kept as is
      - strings like "12%", "12 %", "0.12", "0.12%"
    Returns None if not parsable.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        # Keep the original for % detection
        has_percent = "%" in s
        # Remove non-numeric except dot, minus
        s_num = re.sub(r"[^0-9.\-]", "", s)
        if s_num in ("", ".", "-", "-."):
            return None
        try:
            num = float(s_num)
        except ValueError:
            return None
        if has_percent:
            return max(0.0, min(100.0, num))
        # No explicit %: decide by range
        if 0 <= num <= 1:
            return num * 100.0
        return max(0.0, min(100.0, num))
    # Numeric types
    try:
        num = float(value)
    except Exception:
        return None
    if 0 <= num <= 1:
        return num * 100.0
    return max(0.0, min(100.0, num))


def _pick_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try to identify columns for date, down (树下/落地) rate, and up (树上) rate.
    Heuristic by Chinese keywords.
    """
    columns = list(df.columns)
    col_str = [str(c) for c in columns]

    # date-like: prefer columns containing any of these tokens
    date_tokens = ["日期", "时间", "日", "统计日期", "观测日期"]
    rate_tokens = ["感病率", "感染率", "病率", "带菌率", "阳性率"]

    down_tokens = ["树下", "落地", "地面", "掉落"]
    up_tokens = ["树上", "在树", "挂树", "挂果"]

    date_col = None
    for c in col_str:
        if any(tok in c for tok in date_tokens):
            date_col = c
            break
    if date_col is None:
        # Fallback: first column if dtype datetime-like or looks like date
        for c in columns:
            series = df[c]
            if pd.api.types.is_datetime64_any_dtype(series):
                date_col = str(c)
                break

    # rate cols: match down/up with rate tokens present somewhere
    def match_rate(target_tokens):
        candidates = []
        for c in col_str:
            if any(tok in c for tok in target_tokens) and any(rt in c for rt in rate_tokens):
                candidates.append(c)
        # If not found, allow looser: target token only
        if not candidates:
            for c in col_str:
                if any(tok in c for tok in target_tokens):
                    candidates.append(c)
        return candidates[0] if candidates else None

    down_col = match_rate(down_tokens)
    up_col = match_rate(up_tokens)

    return date_col, down_col, up_col


def _classify_warning(down_rate: Optional[float], up_rate: Optional[float]) -> str:
    """Apply the provided warning rules and return 等级 string."""
    # Treat None as missing; if missing, return 未定义 以免误判
    if down_rate is None or up_rate is None:
        return "未定义"

    # Standardize rounding a little to avoid float noise
    d = round(down_rate, 6)
    u = round(up_rate, 6)

    # (1) both 0 -> 0
    if d == 0 and u == 0:
        return "0"
    # (2) down <= 10%, up == 0 -> 轻度
    if d <= 10 and u == 0:
        return "轻度（不易发生）"
    # (3) 10% < down <= 40%, up == 0 -> 中度
    if 10 < d <= 40 and u == 0:
        return "中度（较易发生）"
    # (4) down > 40%, up > 0 -> 重度
    if d > 40 and u > 0:
        return "重度（易发生）"

    # Not explicitly covered by the rules
    return "未定义"


def process_file(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_预警.xlsx"

    # Read all sheets, preserve them by adding warning result columns where possible
    xls = pd.read_excel(input_path, sheet_name=None)
    out_sheets: Dict[str, pd.DataFrame] = {}

    for sheet_name, df in xls.items():
        if df.empty:
            out_sheets[sheet_name] = df
            continue

        # Try to pick columns
        date_col, down_col, up_col = _pick_columns(df)

        # Prepare a working copy to avoid modifying original dtypes too much
        work = df.copy()

        # Normalize rates if columns exist
        down_rate_series = None
        up_rate_series = None

        if down_col in work.columns:
            down_rate_series = work[down_col].map(_normalize_rate)
        if up_col in work.columns:
            up_rate_series = work[up_col].map(_normalize_rate)

        # Build result columns
        if down_rate_series is not None:
            work["树下感病率_标准化(%)"] = down_rate_series
        if up_rate_series is not None:
            work["树上感病率_标准化(%)"] = up_rate_series

        # Warning level
        if down_rate_series is not None and up_rate_series is not None:
            work["预警等级"] = [
                _classify_warning(d, u) for d, u in zip(down_rate_series, up_rate_series)
            ]
        else:
            work["预警等级"] = "未定义"

        # Reorder: date (if any) first, then original columns, then computed
        preferred_order = []
        if date_col in work.columns:
            preferred_order.append(date_col)
        # Keep original order
        original_cols = [c for c in df.columns if c != date_col]
        computed_cols = [c for c in ["树下感病率_标准化(%)", "树上感病率_标准化(%)", "预警等级"] if c in work.columns]
        new_cols = preferred_order + original_cols + [c for c in computed_cols if c not in preferred_order + original_cols]
        work = work[new_cols]

        out_sheets[sheet_name] = work

    # Write output
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in out_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="根据规则生成荔枝霜疫预警表（_预警.xlsx）")
    parser.add_argument("input", help="输入 Excel 文件路径")
    args = parser.parse_args()

    output = process_file(args.input)
    print(f"已生成: {output}")


if __name__ == "__main__":
    main()


