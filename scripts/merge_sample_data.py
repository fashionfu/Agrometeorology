#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并感染率数据和气象因子数据，生成样本数据.xlsx
"""

import os
import pandas as pd
from pathlib import Path


def find_date_column(df: pd.DataFrame) -> str:
    """查找日期列"""
    date_tokens = ["日期", "时间", "日", "date", "Date", "统计日期", "观测日期"]
    for col in df.columns:
        col_str = str(col)
        if any(tok in col_str for tok in date_tokens):
            return col
    # 如果没有找到，检查是否有datetime类型的列
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    raise ValueError("无法找到日期列，请检查数据文件")


def to_datetime_series(s: pd.Series) -> pd.Series:
    """转换为日期时间序列并标准化"""
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def merge_data(
    infection_file: str,
    weather_file: str,
    output_file: str = "metadata/样本数据\.xlsx"
):
    """
    合并感染率数据和气象因子数据
    
    参数:
        infection_file: 感染率数据Excel文件路径
        weather_file: 气象因子数据文件路径（支持.xlsx或.csv）
        output_file: 输出文件名
    """
    print(f"正在读取感染率数据: {infection_file}")
    # 读取感染率数据
    infection_df = pd.read_excel(infection_file, sheet_name=0)
    
    print(f"正在读取气象因子数据: {weather_file}")
    # 读取气象因子数据
    if weather_file.endswith('.xlsx'):
        weather_df = pd.read_excel(weather_file, sheet_name=0)
    elif weather_file.endswith('.csv'):
        weather_df = pd.read_csv(weather_file, encoding='utf-8', engine='python')
    else:
        raise ValueError(f"不支持的文件格式: {weather_file}，请使用.xlsx或.csv文件")
    
    print("正在查找日期列...")
    # 查找日期列
    infection_date_col = find_date_column(infection_df)
    weather_date_col = find_date_column(weather_df)
    
    print(f"感染率数据日期列: {infection_date_col}")
    print(f"气象因子数据日期列: {weather_date_col}")
    
    # 标准化日期
    infection_df[infection_date_col] = to_datetime_series(infection_df[infection_date_col])
    weather_df[weather_date_col] = to_datetime_series(weather_df[weather_date_col])
    
    # 合并数据（内连接，只保留有匹配的日期）
    print("正在合并数据...")
    merged_df = pd.merge(
        infection_df,
        weather_df,
        left_on=infection_date_col,
        right_on=weather_date_col,
        how="inner",
        suffixes=("", "_气象")
    )
    
    # 删除重复的日期列（如果有）
    if f"{weather_date_col}_气象" in merged_df.columns:
        merged_df = merged_df.drop(columns=[f"{weather_date_col}_气象"])
    
    print(f"合并完成，共 {len(merged_df)} 条记录")
    
    # 保存结果
    print(f"正在保存到: {output_file}")
    merged_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"✓ 已成功保存: {output_file}")
    
    return merged_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="合并感染率数据和气象因子数据")
    parser.add_argument(
        "--infection",
        default="metadata/张桂香19\-25校内大果园花果带菌率数据分析\-\-给张工分析数据\-10\.20\.xlsx",
        help="感染率数据Excel文件路径"
    )
    parser.add_argument(
        "--weather",
        default="metadata/影响因子1103_final\.xlsx",
        help="气象因子数据文件路径（支持.xlsx或.csv）"
    )
    parser.add_argument(
        "--output",
        default="metadata/样本数据\.xlsx",
        help="输出文件名"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.infection):
        print(f"错误: 找不到感染率数据文件: {args.infection}")
        return
    
    if not os.path.exists(args.weather):
        # 尝试.csv文件
        weather_csv = args.weather.replace('.xlsx', '.csv')
        if os.path.exists(weather_csv):
            args.weather = weather_csv
            print(f"使用CSV文件: {args.weather}")
        else:
            print(f"错误: 找不到气象因子数据文件: {args.weather} 或 {weather_csv}")
            return
    
    try:
        merge_data(args.infection, args.weather, args.output)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
