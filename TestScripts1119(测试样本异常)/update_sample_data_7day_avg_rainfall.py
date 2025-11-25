#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将影响因子1103_final_更新.xlsx中的7天日均降雨量数据更新到样本数据.xlsx

功能：
1. 从影响因子1103_final_更新.xlsx读取7天日均降雨量数据
2. 在样本数据.xlsx中找到"7天日均降雨量"列
3. 根据日期匹配，填充样本数据中的7天日均降雨量列
4. 保存更新后的样本数据.xlsx
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

# 导入日期处理函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_thresholds_single import _find_date_column, _to_datetime_series


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


def main():
    parser = argparse.ArgumentParser(
        description="将影响因子文件中的7天日均降雨量数据更新到样本数据文件中"
    )
    
    # 使用脚本所在目录为基准，构造健壮默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_factor_file = os.path.join(project_root, "影响因子1103_final_更新.xlsx")
    default_sample_file = os.path.join(project_root, "样本数据.xlsx")
    default_output_file = os.path.join(project_root, "样本数据_更新.xlsx")
    
    parser.add_argument(
        "--factor-file",
        default=default_factor_file,
        help="影响因子数据文件路径（包含7天日均降雨量数据）"
    )
    parser.add_argument(
        "--sample-file",
        default=default_sample_file,
        help="样本数据文件路径（需要更新的文件）"
    )
    parser.add_argument(
        "--output-file",
        default=default_output_file,
        help="输出文件路径（更新后的样本数据）"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="是否创建备份文件"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("更新样本数据中的7天日均降雨量列")
    print("=" * 80)
    
    # 1. 读取影响因子数据文件
    print(f"\n1. 读取影响因子数据文件: {args.factor_file}")
    if not os.path.exists(args.factor_file):
        raise FileNotFoundError(f"文件不存在: {args.factor_file}")
    
    df_factor = pd.read_excel(args.factor_file, sheet_name=0)
    print(f"   形状: {df_factor.shape}")
    print(f"   列名: {list(df_factor.columns)[:15]}...")
    
    # 识别日期列
    date_col_factor = _find_date_column(df_factor)
    if date_col_factor is None:
        raise ValueError("无法识别影响因子文件中的日期列")
    print(f"   日期列: {date_col_factor}")
    
    # 识别7天日均降雨量列
    col_7day_daily_avg_factor = find_7day_daily_avg_rainfall_column(df_factor)
    if col_7day_daily_avg_factor is None:
        raise ValueError("无法找到影响因子文件中的7天日均降雨量列，请检查文件")
    print(f"   7天日均降雨量列: {col_7day_daily_avg_factor}")
    
    # 标准化日期并筛选有效数据
    df_factor[date_col_factor] = _to_datetime_series(df_factor[date_col_factor])
    df_factor = df_factor.dropna(subset=[date_col_factor])
    
    # 确保7天日均降雨量列是数值类型
    df_factor[col_7day_daily_avg_factor] = pd.to_numeric(df_factor[col_7day_daily_avg_factor], errors='coerce')
    
    # 创建日期到7天日均降雨量的映射（标准化日期索引）
    df_factor = df_factor.sort_values(by=date_col_factor)
    df_factor['date_normalized'] = df_factor[date_col_factor].apply(
        lambda x: pd.Timestamp(x).normalize() if pd.notna(x) else x
    )
    
    # 创建映射字典（日期 -> 7天日均降雨量）
    date_to_avg_rainfall = {}
    for idx, row in df_factor.iterrows():
        date_norm = row['date_normalized']
        avg_rainfall = row[col_7day_daily_avg_factor]
        if pd.notna(date_norm) and pd.notna(avg_rainfall):
            date_to_avg_rainfall[date_norm] = avg_rainfall
    
    print(f"   有效数据: {len(date_to_avg_rainfall)} 条")
    print(f"   日期范围: {min(date_to_avg_rainfall.keys())} 至 {max(date_to_avg_rainfall.keys())}")
    
    # 2. 读取样本数据文件
    print(f"\n2. 读取样本数据文件: {args.sample_file}")
    if not os.path.exists(args.sample_file):
        raise FileNotFoundError(f"文件不存在: {args.sample_file}")
    
    df_sample = pd.read_excel(args.sample_file, sheet_name=0)
    print(f"   形状: {df_sample.shape}")
    print(f"   列名: {list(df_sample.columns)[:15]}...")
    
    # 识别日期列
    date_col_sample = _find_date_column(df_sample)
    if date_col_sample is None:
        raise ValueError("无法识别样本数据文件中的日期列")
    print(f"   日期列: {date_col_sample}")
    
    # 识别7天日均降雨量列
    col_7day_daily_avg_sample = find_7day_daily_avg_rainfall_column(df_sample)
    if col_7day_daily_avg_sample is None:
        raise ValueError("无法找到样本数据文件中的7天日均降雨量列，请检查文件")
    print(f"   7天日均降雨量列: {col_7day_daily_avg_sample}")
    
    # 检查列中已有的非空值
    existing_non_null = df_sample[col_7day_daily_avg_sample].notna().sum()
    if existing_non_null > 0:
        print(f"   注意: 7天日均降雨量列已有 {existing_non_null} 个非空值，将被覆盖")
    
    # 3. 创建备份（如果需要）
    if args.backup:
        backup_file = args.sample_file.replace(".xlsx", "_备份.xlsx")
        print(f"\n3. 创建备份文件: {backup_file}")
        df_sample.to_excel(backup_file, index=False, engine='openpyxl')
        print(f"   备份完成")
    
    # 4. 根据日期匹配，更新7天日均降雨量列
    print(f"\n4. 根据日期匹配，更新7天日均降雨量列...")
    
    # 标准化样本数据的日期
    df_sample[date_col_sample] = _to_datetime_series(df_sample[date_col_sample])
    
    # 匹配日期并更新值
    updated_count = 0
    missing_count = 0
    missing_dates = []
    kept_existing_count = 0
    
    for idx, row in df_sample.iterrows():
        date = row[date_col_sample]
        if pd.isna(date):
            missing_count += 1
            continue
        
        # 标准化日期（只保留日期部分，忽略时间）
        date_normalized = pd.Timestamp(date).normalize()
        
        # 查找匹配的日期
        if date_normalized in date_to_avg_rainfall:
            new_value = date_to_avg_rainfall[date_normalized]
            df_sample.at[idx, col_7day_daily_avg_sample] = new_value
            updated_count += 1
        else:
            # 如果当前值已存在且不为空，保留原值
            current_value = row[col_7day_daily_avg_sample]
            if pd.notna(current_value):
                kept_existing_count += 1
            else:
                missing_count += 1
                missing_dates.append(str(date))
    
    print(f"   更新成功: {updated_count} 条")
    if kept_existing_count > 0:
        print(f"   保留原值: {kept_existing_count} 条（未找到匹配日期但已有值）")
    if missing_count > 0:
        print(f"   未找到匹配日期: {missing_count} 条（保留原值或为空）")
        if len(missing_dates) <= 10:
            print(f"   未匹配的日期: {', '.join(missing_dates)}")
        else:
            print(f"   未匹配的日期（前10个）: {', '.join(missing_dates[:10])}...")
    
    # 5. 保存更新后的文件
    print(f"\n5. 保存更新后的文件: {args.output_file}")
    
    # 读取原始文件的所有sheet（如果有多个sheet）
    try:
        excel_file = pd.ExcelFile(args.sample_file)
        sheet_names = excel_file.sheet_names
        
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            # 更新第一个sheet（包含我们修改的数据）
            df_sample.to_excel(writer, sheet_name=sheet_names[0], index=False)
            
            # 复制其他sheet（如果有）
            for sheet_name in sheet_names[1:]:
                sheet_df = pd.read_excel(args.sample_file, sheet_name=sheet_name)
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"   保存完成（共 {len(sheet_names)} 个sheet）")
    except Exception as e:
        # 如果读取多个sheet失败，只保存第一个sheet
        print(f"   警告: 无法读取多个sheet，只保存第一个sheet: {str(e)}")
        df_sample.to_excel(args.output_file, index=False, engine='openpyxl')
        print(f"   保存完成")
    
    # 6. 显示更新统计
    print(f"\n" + "=" * 80)
    print("更新统计")
    print("=" * 80)
    print(f"影响因子文件: {args.factor_file}")
    print(f"样本数据文件: {args.sample_file}")
    print(f"输出文件: {args.output_file}")
    print(f"更新的列: {col_7day_daily_avg_sample}")
    print(f"更新行数: {updated_count} / {len(df_sample)}")
    if kept_existing_count > 0:
        print(f"保留原值行数: {kept_existing_count}")
    if missing_count > 0:
        print(f"未匹配行数: {missing_count}")
    print(f"\n✓ 处理完成！")


if __name__ == "__main__":
    main()

