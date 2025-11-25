#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
填充影响因子1103_final.xlsx中的"7天日均降雨量"列

功能：
1. 从影响因子1103_final.xlsx读取2019年至今的每日数据
2. 使用"当天雨量"列计算每行的7天日均降雨量（过去7天包括当天的平均日降雨量）
3. 填充到"7天日均降雨量"列
4. 保存更新后的文件
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

# 导入日期处理函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_thresholds_single import _find_date_column, _to_datetime_series


def find_daily_rainfall_column(df: pd.DataFrame) -> str:
    """查找当天雨量列"""
    keywords_list = [
        ["当天", "雨量"],
        ["当天", "降雨", "量"],
        ["日", "雨量"],
        ["日", "降雨", "量"],
        ["当天雨量"],
        ["当天降雨量"],
    ]
    
    for keywords in keywords_list:
        for col in df.columns:
            col_str = str(col)
            if all(kw in col_str for kw in keywords):
                return col
    
    # 如果找不到，列出所有可能相关的列
    candidates = []
    for col in df.columns:
        col_str = str(col)
        if ("当天" in col_str or "日" in col_str) and ("雨" in col_str or "降雨" in col_str):
            candidates.append(col)
    
    if candidates:
        print(f"未找到精确匹配的列，但找到以下候选列：{candidates}")
        return candidates[0]
    
    return None


def find_7day_daily_avg_rainfall_column(df: pd.DataFrame) -> str:
    """查找7天日均降雨量列"""
    keywords_list = [
        ["7", "日均", "降雨", "量"],
        ["7", "日均", "雨量"],
        ["7天", "日均", "降雨", "量"],
        ["7天", "日均", "雨量"],
    ]
    
    for keywords in keywords_list:
        for col in df.columns:
            col_str = str(col)
            if all(kw in col_str for kw in keywords):
                return col
    
    # 如果找不到，列出所有可能相关的列
    candidates = []
    for col in df.columns:
        col_str = str(col)
        if ("7" in col_str or "七" in col_str) and ("日均" in col_str) and ("雨" in col_str or "降雨" in col_str):
            candidates.append(col)
    
    if candidates:
        print(f"未找到精确匹配的列，但找到以下候选列：{candidates}")
        return candidates[0]
    
    return None


def calculate_7day_daily_avg_rainfall(df: pd.DataFrame, date_col: str, daily_rainfall_col: str) -> pd.Series:
    """
    计算每行的7天日均降雨量（过去7天包括当天的平均日降雨量）
    
    参数:
        df: 包含每日数据的DataFrame
        date_col: 日期列名
        daily_rainfall_col: 当天雨量列名
    
    返回:
        包含每行7天日均降雨量的Series
    """
    # 确保日期列是datetime类型
    df = df.copy()
    df[date_col] = _to_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col])
    
    # 确保当天雨量列是数值类型
    df[daily_rainfall_col] = pd.to_numeric(df[daily_rainfall_col], errors='coerce')
    
    # 按日期排序
    df = df.sort_values(by=date_col)
    
    # 重置索引以便后续操作
    df = df.reset_index(drop=True)
    
    # 计算7天滚动平均（过去7天包括当天的平均）
    # 使用rolling window=7，min_periods=1表示即使不足7天也计算平均值
    rolling_7day_avg = df[daily_rainfall_col].rolling(window=7, min_periods=1).mean()
    
    return rolling_7day_avg


def main():
    parser = argparse.ArgumentParser(
        description="填充影响因子1103_final.xlsx中的7天日均降雨量列"
    )
    
    # 使用脚本所在目录为基准，构造健壮默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_factor_file = os.path.join(project_root, "影响因子1103_final.xlsx")
    default_output_file = os.path.join(project_root, "影响因子1103_final_更新.xlsx")
    
    parser.add_argument(
        "--factor-file",
        default=default_factor_file,
        help="影响因子数据文件路径（包含2019年至今的每日数据）"
    )
    parser.add_argument(
        "--output-file",
        default=default_output_file,
        help="输出文件路径（更新后的影响因子文件）"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="是否创建备份文件"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("填充7天日均降雨量列")
    print("=" * 80)
    
    # 1. 读取影响因子数据文件
    print(f"\n1. 读取影响因子数据文件: {args.factor_file}")
    if not os.path.exists(args.factor_file):
        raise FileNotFoundError(f"文件不存在: {args.factor_file}")
    
    df = pd.read_excel(args.factor_file, sheet_name=0)
    print(f"   形状: {df.shape}")
    print(f"   列名: {list(df.columns)[:15]}...")
    
    # 识别日期列
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError("无法识别影响因子文件中的日期列")
    print(f"   日期列: {date_col}")
    
    # 识别当天雨量列
    daily_rainfall_col = find_daily_rainfall_column(df)
    if daily_rainfall_col is None:
        raise ValueError("无法找到当天雨量列，请检查影响因子文件")
    print(f"   当天雨量列: {daily_rainfall_col}")
    
    # 识别7天日均降雨量列
    col_7day_daily_avg = find_7day_daily_avg_rainfall_column(df)
    if col_7day_daily_avg is None:
        raise ValueError("无法找到7天日均降雨量列，请检查影响因子文件")
    print(f"   7天日均降雨量列: {col_7day_daily_avg}")
    
    # 检查7天日均降雨量列是否为空
    non_null_count = df[col_7day_daily_avg].notna().sum()
    if non_null_count > 0:
        print(f"   警告: 7天日均降雨量列已有 {non_null_count} 个非空值，将全部覆盖")
    
    # 筛选2019年至今的数据
    df[date_col] = _to_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col])
    df = df[df[date_col].dt.year >= 2019]
    print(f"   2019年至今的数据: {len(df)} 条")
    
    # 2. 创建备份（如果需要）
    if args.backup:
        backup_file = args.factor_file.replace(".xlsx", "_备份.xlsx")
        print(f"\n2. 创建备份文件: {backup_file}")
        pd.read_excel(args.factor_file, sheet_name=0).to_excel(backup_file, index=False, engine='openpyxl')
        print(f"   备份完成")
    
    # 3. 计算每行的7天日均降雨量
    print(f"\n3. 计算每行的7天日均降雨量（过去7天包括当天的平均日降雨量）...")
    
    # 保存原始索引以便后续恢复
    original_index = df.index.copy()
    
    # 计算7天日均降雨量
    df_7day_avg = calculate_7day_daily_avg_rainfall(df, date_col, daily_rainfall_col)
    
    # 将计算结果填充到7天日均降雨量列
    df.loc[df_7day_avg.index, col_7day_daily_avg] = df_7day_avg.values
    
    # 统计填充情况
    filled_count = df[col_7day_daily_avg].notna().sum()
    print(f"   计算完成，共填充 {filled_count} 行的数据")
    
    # 显示一些示例数据
    print(f"\n   示例数据（前10行）:")
    sample_df = df[[date_col, daily_rainfall_col, col_7day_daily_avg]].head(10)
    for idx, row in sample_df.iterrows():
        date_str = str(row[date_col])[:10] if pd.notna(row[date_col]) else "N/A"
        daily_rain = row[daily_rainfall_col] if pd.notna(row[daily_rainfall_col]) else "N/A"
        avg_7day = row[col_7day_daily_avg] if pd.notna(row[col_7day_daily_avg]) else "N/A"
        print(f"      {date_str}: 当天雨量={daily_rain}, 7天日均降雨量={avg_7day:.4f}" if isinstance(avg_7day, (int, float)) else f"      {date_str}: 当天雨量={daily_rain}, 7天日均降雨量={avg_7day}")
    
    # 4. 保存更新后的文件
    print(f"\n4. 保存更新后的文件: {args.output_file}")
    
    # 读取原始文件的所有sheet（如果有多个sheet）
    try:
        excel_file = pd.ExcelFile(args.factor_file)
        sheet_names = excel_file.sheet_names
        
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            # 更新第一个sheet（包含我们修改的数据）
            df.to_excel(writer, sheet_name=sheet_names[0], index=False)
            
            # 复制其他sheet（如果有）
            for sheet_name in sheet_names[1:]:
                sheet_df = pd.read_excel(args.factor_file, sheet_name=sheet_name)
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"   保存完成（共 {len(sheet_names)} 个sheet）")
    except Exception as e:
        # 如果读取多个sheet失败，只保存第一个sheet
        print(f"   警告: 无法读取多个sheet，只保存第一个sheet: {str(e)}")
        df.to_excel(args.output_file, index=False, engine='openpyxl')
        print(f"   保存完成")
    
    # 5. 显示更新统计
    print(f"\n" + "=" * 80)
    print("更新统计")
    print("=" * 80)
    print(f"原始文件: {args.factor_file}")
    print(f"输出文件: {args.output_file}")
    print(f"更新的列: {col_7day_daily_avg}")
    print(f"填充行数: {filled_count} / {len(df)}")
    print(f"日期范围: {df[date_col].min()} 至 {df[date_col].max()}")
    print(f"\n✓ 处理完成！")


if __name__ == "__main__":
    main()
