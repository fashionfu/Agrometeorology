#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
气象因子数据质量检验脚本
检验影响因子1103_final.xlsx的数据质量，包括：
- 基本信息（行数、列数、列名）
- 日期列检查
- 缺失值检查
- 数据类型检查
- 异常值检查
- 重复值检查
- 特征列识别
- 数值范围检查
"""
import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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


def check_basic_info(df: pd.DataFrame) -> Dict:
    """检查基本信息"""
    info = {
        "总行数": len(df),
        "总列数": len(df.columns),
        "列名列表": list(df.columns),
        "数据形状": df.shape,
    }
    return info


def check_date_column(df: pd.DataFrame) -> Dict:
    """检查日期列"""
    date_col = _find_date_column(df)
    result = {
        "日期列名称": date_col,
        "是否存在": date_col is not None,
        "日期值数量": 0,
        "缺失日期数量": 0,
        "日期范围": None,
        "日期格式问题": [],
    }
    
    if date_col:
        date_series = _to_datetime_series(df[date_col])
        result["日期值数量"] = date_series.notna().sum()
        result["缺失日期数量"] = date_series.isna().sum()
        valid_dates = date_series.dropna()
        if len(valid_dates) > 0:
            result["日期范围"] = (valid_dates.min(), valid_dates.max())
        
        # 检查格式问题
        invalid_dates = df[date_col][date_series.isna()]
        if len(invalid_dates) > 0:
            result["日期格式问题"] = invalid_dates.head(10).tolist()
    
    return result


def check_missing_values(df: pd.DataFrame) -> Dict:
    """检查缺失值"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    result = {
        "总缺失值数量": missing.sum(),
        "有缺失值的列数": (missing > 0).sum(),
        "缺失值详情": {},
        "缺失率最高的5列": [],
    }
    
    for col in df.columns:
        if missing[col] > 0:
            result["缺失值详情"][col] = {
                "缺失数量": int(missing[col]),
                "缺失率(%)": float(missing_pct[col]),
            }
    
    # 缺失率最高的5列
    if missing.sum() > 0:
        missing_sorted = missing_pct[missing_pct > 0].sort_values(ascending=False)
        result["缺失率最高的5列"] = [
            {"列名": col, "缺失率(%)": float(missing_sorted[col])}
            for col in missing_sorted.head(5).index
        ]
    
    return result


def check_data_types(df: pd.DataFrame) -> Dict:
    """检查数据类型"""
    result = {
        "各列数据类型": {},
        "数值型列数": 0,
        "文本型列数": 0,
        "日期型列数": 0,
        "类型建议": [],
    }
    
    numeric_cols = []
    text_cols = []
    date_cols = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        result["各列数据类型"][col] = dtype
        
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        else:
            text_cols.append(col)
    
    result["数值型列数"] = len(numeric_cols)
    result["文本型列数"] = len(text_cols)
    result["日期型列数"] = len(date_cols)
    
    # 检查可能是数值但被识别为文本的列
    for col in text_cols[:10]:  # 只检查前10个文本列
        sample = df[col].dropna().head(10)
        if len(sample) > 0:
            try:
                pd.to_numeric(sample, errors="raise")
                result["类型建议"].append(f"{col}: 可能是数值型，但被识别为文本型")
            except:
                pass
    
    return result


def check_duplicates(df: pd.DataFrame, date_col: Optional[str] = None) -> Dict:
    """检查重复值"""
    result = {
        "完全重复行数": df.duplicated().sum(),
        "日期重复": None,
    }
    
    if date_col and date_col in df.columns:
        date_duplicates = df[date_col].duplicated().sum()
        result["日期重复"] = {
            "重复日期数量": int(date_duplicates),
            "重复日期列表": df[date_col][df[date_col].duplicated(keep=False)].unique().tolist()[:10],
        }
    
    return result


def check_outliers(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """检查异常值"""
    result = {
        "异常值详情": {},
        "负值检查": {},
        "合理范围检查": {},
    }
    
    # 定义合理范围
    reasonable_ranges = {
        "温度": (-50, 50),
        "湿度": (0, 100),
        "相对湿度": (0, 100),
        "降雨": (0, 1000),
        "降雨量": (0, 1000),
        "降水量": (0, 1000),
        "降雨时数": (0, 24),
        "日照": (0, 24),
        "日照时数": (0, 24),
    }
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        try:
            col_data = pd.to_numeric(df[col], errors="coerce")
            valid_data = col_data.dropna()
            
            if len(valid_data) == 0:
                continue
            
            # 负值检查
            negatives = (valid_data < 0).sum()
            if negatives > 0:
                result["负值检查"][col] = {
                    "负值数量": int(negatives),
                    "负值比例(%)": float(negatives / len(valid_data) * 100),
                    "最小负值": float(valid_data[valid_data < 0].min()) if negatives > 0 else None,
                }
            
            # 合理范围检查
            col_lower = None
            col_upper = None
            for key, (lower, upper) in reasonable_ranges.items():
                if key in col:
                    col_lower = lower
                    col_upper = upper
                    break
            
            if col_lower is not None and col_upper is not None:
                outliers_low = (valid_data < col_lower).sum()
                outliers_high = (valid_data > col_upper).sum()
                if outliers_low > 0 or outliers_high > 0:
                    result["合理范围检查"][col] = {
                        "建议范围": f"[{col_lower}, {col_upper}]",
                        "低于下限数量": int(outliers_low),
                        "高于上限数量": int(outliers_high),
                        "超出范围的值": {
                            "最小值": float(valid_data.min()),
                            "最大值": float(valid_data.max()),
                        },
                    }
            
            # 统计信息
            result["异常值详情"][col] = {
                "有效值数量": int(len(valid_data)),
                "缺失值数量": int(col_data.isna().sum()),
                "最小值": float(valid_data.min()),
                "最大值": float(valid_data.max()),
                "平均值": float(valid_data.mean()),
                "中位数": float(valid_data.median()),
                "标准差": float(valid_data.std()),
            }
            
        except Exception as e:
            result["异常值详情"][col] = {"错误": str(e)}
    
    return result


def check_features(df: pd.DataFrame) -> Dict:
    """检查特征列识别"""
    feature_cols = _select_feature_columns(df)
    result = {
        "识别到的特征列数": len(feature_cols),
        "特征列列表": feature_cols,
        "未识别的列": [col for col in df.columns if col not in feature_cols],
    }
    return result


def generate_report(check_results: Dict, output_path: str):
    """生成检验报告"""
    lines = []
    
    lines.append("# 气象因子数据质量检验报告\n\n")
    lines.append("本文档对 `影响因子1103_final.xlsx` 进行全面的数据质量检验。\n\n")
    lines.append("---\n\n")
    
    # 1. 基本信息
    lines.append("## 1. 基本信息\n\n")
    basic = check_results["basic_info"]
    lines.append(f"- **总行数**: {basic['总行数']}\n")
    lines.append(f"- **总列数**: {basic['总列数']}\n")
    lines.append(f"- **数据形状**: {basic['数据形状'][0]} 行 × {basic['数据形状'][1]} 列\n\n")
    
    lines.append("### 列名列表\n\n")
    for i, col in enumerate(basic['列名列表'], 1):
        lines.append(f"{i}. {col}\n")
    lines.append("\n")
    
    # 2. 日期列检查
    lines.append("## 2. 日期列检查\n\n")
    date_check = check_results["date_check"]
    if date_check["是否存在"]:
        lines.append(f"- **日期列名称**: {date_check['日期列名称']}\n")
        lines.append(f"- **有效日期数量**: {date_check['日期值数量']}\n")
        lines.append(f"- **缺失日期数量**: {date_check['缺失日期数量']}\n")
        if date_check['日期范围']:
            lines.append(f"- **日期范围**: {date_check['日期范围'][0]} 至 {date_check['日期范围'][1]}\n")
        if date_check['日期格式问题']:
            lines.append(f"- **日期格式问题**: 发现 {len(date_check['日期格式问题'])} 条无法解析的日期\n")
            lines.append("  示例值:\n")
            for val in date_check['日期格式问题'][:5]:
                lines.append(f"  - {val}\n")
    else:
        lines.append("⚠️ **警告**: 未找到日期列！\n")
    lines.append("\n")
    
    # 3. 缺失值检查
    lines.append("## 3. 缺失值检查\n\n")
    missing = check_results["missing_values"]
    lines.append(f"- **总缺失值数量**: {missing['总缺失值数量']}\n")
    lines.append(f"- **有缺失值的列数**: {missing['有缺失值的列数']}\n\n")
    
    if missing['缺失值详情']:
        lines.append("### 缺失值详情\n\n")
        lines.append("| 列名 | 缺失数量 | 缺失率(%) |\n")
        lines.append("|------|---------|----------|\n")
        for col, info in list(missing['缺失值详情'].items())[:20]:  # 只显示前20列
            lines.append(f"| {col} | {info['缺失数量']} | {info['缺失率(%)']} |\n")
        lines.append("\n")
        
        if missing['缺失率最高的5列']:
            lines.append("### 缺失率最高的5列\n\n")
            for item in missing['缺失率最高的5列']:
                lines.append(f"- **{item['列名']}**: {item['缺失率(%)']}%\n")
    else:
        lines.append("✅ **良好**: 未发现缺失值\n")
    lines.append("\n")
    
    # 4. 数据类型检查
    lines.append("## 4. 数据类型检查\n\n")
    dtype_check = check_results["data_types"]
    lines.append(f"- **数值型列数**: {dtype_check['数值型列数']}\n")
    lines.append(f"- **文本型列数**: {dtype_check['文本型列数']}\n")
    lines.append(f"- **日期型列数**: {dtype_check['日期型列数']}\n\n")
    
    if dtype_check['类型建议']:
        lines.append("### 类型建议\n\n")
        for suggestion in dtype_check['类型建议']:
            lines.append(f"- ⚠️ {suggestion}\n")
        lines.append("\n")
    
    # 5. 重复值检查
    lines.append("## 5. 重复值检查\n\n")
    dup_check = check_results["duplicates"]
    lines.append(f"- **完全重复行数**: {dup_check['完全重复行数']}\n")
    if dup_check['日期重复']:
        lines.append(f"- **日期重复数量**: {dup_check['日期重复']['重复日期数量']}\n")
        if dup_check['日期重复']['重复日期列表']:
            lines.append("  重复日期示例:\n")
            for date in dup_check['日期重复']['重复日期列表'][:5]:
                lines.append(f"  - {date}\n")
    lines.append("\n")
    
    # 6. 特征列识别
    lines.append("## 6. 特征列识别\n\n")
    feature_check = check_results["features"]
    lines.append(f"- **识别到的特征列数**: {feature_check['识别到的特征列数']}\n")
    lines.append(f"- **特征列列表**:\n")
    for col in feature_check['特征列列表']:
        lines.append(f"  - {col}\n")
    lines.append("\n")
    
    if feature_check['未识别的列']:
        lines.append("### 未识别的列（非气象因子）\n\n")
        for col in feature_check['未识别的列']:
            lines.append(f"  - {col}\n")
        lines.append("\n")
    
    # 7. 异常值检查
    lines.append("## 7. 异常值检查\n\n")
    outlier_check = check_results["outliers"]
    
    if outlier_check['负值检查']:
        lines.append("### 负值检查\n\n")
        lines.append("| 列名 | 负值数量 | 负值比例(%) | 最小负值 |\n")
        lines.append("|------|---------|------------|---------|\n")
        for col, info in outlier_check['负值检查'].items():
            lines.append(f"| {col} | {info['负值数量']} | {info['负值比例(%)']:.2f} | {info['最小负值']} |\n")
        lines.append("\n")
    else:
        lines.append("✅ **良好**: 未发现负值\n\n")
    
    if outlier_check['合理范围检查']:
        lines.append("### 合理范围检查\n\n")
        for col, info in outlier_check['合理范围检查'].items():
            lines.append(f"- **{col}**:\n")
            lines.append(f"  - 建议范围: {info['建议范围']}\n")
            lines.append(f"  - 低于下限数量: {info['低于下限数量']}\n")
            lines.append(f"  - 高于上限数量: {info['高于上限数量']}\n")
            lines.append(f"  - 实际范围: [{info['超出范围的值']['最小值']:.2f}, {info['超出范围的值']['最大值']:.2f}]\n")
        lines.append("\n")
    
    # 8. 特征统计信息
    lines.append("## 8. 特征统计信息\n\n")
    if outlier_check['异常值详情']:
        lines.append("| 列名 | 有效值数量 | 缺失值数量 | 最小值 | 最大值 | 平均值 | 中位数 | 标准差 |\n")
        lines.append("|------|-----------|-----------|--------|--------|--------|--------|--------|\n")
        for col, stats in list(outlier_check['异常值详情'].items())[:20]:  # 只显示前20列
            if '错误' not in stats:
                lines.append(f"| {col} | {stats['有效值数量']} | {stats['缺失值数量']} | "
                           f"{stats['最小值']:.2f} | {stats['最大值']:.2f} | "
                           f"{stats['平均值']:.2f} | {stats['中位数']:.2f} | {stats['标准差']:.2f} |\n")
        lines.append("\n")
    
    # 9. 总结与建议
    lines.append("## 9. 总结与建议\n\n")
    
    issues = []
    if not date_check["是否存在"]:
        issues.append("⚠️ 未找到日期列，可能影响数据合并")
    if missing['总缺失值数量'] > 0:
        issues.append(f"⚠️ 发现 {missing['总缺失值数量']} 个缺失值，需要处理")
    if dup_check['完全重复行数'] > 0:
        issues.append(f"⚠️ 发现 {dup_check['完全重复行数']} 行完全重复的数据")
    if outlier_check['负值检查']:
        issues.append(f"⚠️ 发现 {len(outlier_check['负值检查'])} 列存在负值")
    if outlier_check['合理范围检查']:
        issues.append(f"⚠️ 发现 {len(outlier_check['合理范围检查'])} 列存在超出合理范围的值")
    
    if issues:
        lines.append("### 发现的问题\n\n")
        for issue in issues:
            lines.append(f"- {issue}\n")
        lines.append("\n")
    else:
        lines.append("✅ **数据质量良好**，未发现明显问题。\n\n")
    
    lines.append("### 建议\n\n")
    lines.append("1. 如有缺失值，建议使用插值或删除缺失值过多的行/列\n")
    lines.append("2. 如有异常值（负值、超出合理范围），建议检查数据来源或进行修正\n")
    lines.append("3. 如有重复行，建议删除重复数据\n")
    lines.append("4. 确保日期列格式正确，便于与感染率数据合并\n")
    lines.append("5. 确保特征列名称包含关键词（温度、湿度、降雨、日照等）\n\n")
    
    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"已生成检验报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="气象因子数据质量检验")
    parser.add_argument(
        "--input",
        default="metadata/影响因子1103_final\.xlsx",
        help="输入Excel文件路径"
    )
    parser.add_argument(
        "--output",
        default="analysis_1104/检验气象因子.md",
        help="输出报告路径（Markdown格式）"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("气象因子数据质量检验")
    print("=" * 60)
    
    # 读取数据
    print(f"\n正在读取数据文件: {args.input}")
    try:
        df = pd.read_excel(args.input, sheet_name=0)
        print(f"✅ 数据读取成功: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 数据读取失败: {e}")
        return
    
    # 执行各项检查
    print("\n正在执行数据质量检验...")
    check_results = {}
    
    print("  1. 检查基本信息...")
    check_results["basic_info"] = check_basic_info(df)
    
    print("  2. 检查日期列...")
    check_results["date_check"] = check_date_column(df)
    
    print("  3. 检查缺失值...")
    check_results["missing_values"] = check_missing_values(df)
    
    print("  4. 检查数据类型...")
    check_results["data_types"] = check_data_types(df)
    
    print("  5. 检查重复值...")
    check_results["duplicates"] = check_duplicates(df, check_results["date_check"]["日期列名称"])
    
    print("  6. 检查特征列识别...")
    check_results["features"] = check_features(df)
    
    print("  7. 检查异常值...")
    check_results["outliers"] = check_outliers(df, check_results["features"]["特征列列表"])
    
    # 生成报告
    print("\n正在生成检验报告...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(check_results, args.output)
    
    print("\n" + "=" * 60)
    print(f"检验完成！报告已保存到: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

