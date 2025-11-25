#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证并修复样本数据_更新.xlsx中的数据

功能：
1. 读取样本数据_更新.xlsx
2. 读取影响因子1103_final_更新.xlsx作为验证数据源
3. 根据日期匹配，验证样本数据中的各个气象因子列
4. 如果发现不一致或缺失，用影响因子文件中的数据替换
5. 保存验证和修复后的文件
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

# 导入日期处理函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_thresholds_single import _find_date_column, _to_datetime_series


def find_matching_columns(df1: pd.DataFrame, df2: pd.DataFrame, exclude_cols: list = None) -> dict:
    """
    查找两个DataFrame中匹配的列名
    
    参数:
        df1: 第一个DataFrame（样本数据）
        df2: 第二个DataFrame（影响因子数据）
        exclude_cols: 需要排除的列名列表（如日期列、预警列等）
    
    返回:
        字典，键为df1中的列名，值为df2中匹配的列名
    """
    if exclude_cols is None:
        exclude_cols = []
    
    matching = {}
    df1_cols = [col for col in df1.columns if col not in exclude_cols]
    df2_cols = list(df2.columns)
    
    for col1 in df1_cols:
        col1_str = str(col1)
        # 精确匹配
        if col1 in df2_cols:
            matching[col1] = col1
        else:
            # 尝试模糊匹配（去除空格、大小写等）
            col1_normalized = col1_str.strip().replace(' ', '').replace('　', '')
            for col2 in df2_cols:
                col2_str = str(col2)
                col2_normalized = col2_str.strip().replace(' ', '').replace('　', '')
                if col1_normalized == col2_normalized:
                    matching[col1] = col2
                    break
    
    return matching


def validate_and_fix_column(
    df_sample: pd.DataFrame,
    df_factor: pd.DataFrame,
    date_col_sample: str,
    date_col_factor: str,
    col_sample: str,
    col_factor: str,
    tolerance: float = 1e-6
) -> dict:
    """
    验证并修复单个列的数据
    
    参数:
        df_sample: 样本数据DataFrame
        df_factor: 影响因子数据DataFrame
        date_col_sample: 样本数据的日期列名
        date_col_factor: 影响因子数据的日期列名
        col_sample: 样本数据中要验证的列名
        col_factor: 影响因子数据中对应的列名
        tolerance: 数值比较的容差
    
    返回:
        统计字典：{
            'total_rows': 总行数,
            'matched_rows': 匹配的行数,
            'fixed_rows': 修复的行数,
            'missing_in_factor': 影响因子中缺失的行数,
            'missing_in_sample': 样本中缺失的行数,
            'mismatched_values': 不匹配的值数量
        }
    """
    stats = {
        'total_rows': len(df_sample),
        'matched_rows': 0,
        'fixed_rows': 0,
        'missing_in_factor': 0,
        'missing_in_sample': 0,
        'mismatched_values': 0
    }
    
    # 创建影响因子数据的日期到值的映射
    factor_map = {}
    for idx, row in df_factor.iterrows():
        date = row[date_col_factor]
        if pd.notna(date):
            date_norm = pd.Timestamp(date).normalize()
            value = row[col_factor]
            if pd.notna(value):
                factor_map[date_norm] = value
    
    # 验证和修复样本数据
    for idx, row in df_sample.iterrows():
        date = row[date_col_sample]
        if pd.isna(date):
            continue
        
        date_norm = pd.Timestamp(date).normalize()
        sample_value = row[col_sample]
        
        if date_norm in factor_map:
            factor_value = factor_map[date_norm]
            stats['matched_rows'] += 1
            
            # 检查是否需要修复
            if pd.isna(sample_value):
                # 样本中缺失，用影响因子的值填充
                df_sample.at[idx, col_sample] = factor_value
                stats['fixed_rows'] += 1
                stats['missing_in_sample'] += 1
            else:
                # 比较数值是否一致
                try:
                    sample_num = float(sample_value)
                    factor_num = float(factor_value)
                    if abs(sample_num - factor_num) > tolerance:
                        # 值不匹配，用影响因子的值替换
                        df_sample.at[idx, col_sample] = factor_value
                        stats['fixed_rows'] += 1
                        stats['mismatched_values'] += 1
                except (ValueError, TypeError):
                    # 非数值类型，直接比较字符串
                    if str(sample_value).strip() != str(factor_value).strip():
                        df_sample.at[idx, col_sample] = factor_value
                        stats['fixed_rows'] += 1
                        stats['mismatched_values'] += 1
        else:
            # 影响因子中找不到对应日期
            stats['missing_in_factor'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="验证并修复样本数据_更新.xlsx中的数据"
    )
    
    # 使用脚本所在目录为基准，构造健壮默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_sample_file = os.path.join(project_root, "样本数据_更新.xlsx")
    default_factor_file = os.path.join(project_root, "影响因子1103_final_更新.xlsx")
    default_output_file = os.path.join(project_root, "样本数据_验证修复.xlsx")
    
    parser.add_argument(
        "--sample-file",
        default=default_sample_file,
        help="样本数据文件路径（需要验证的文件）"
    )
    parser.add_argument(
        "--factor-file",
        default=default_factor_file,
        help="影响因子数据文件路径（验证数据源）"
    )
    parser.add_argument(
        "--output-file",
        default=default_output_file,
        help="输出文件路径（验证修复后的样本数据）"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="是否创建备份文件"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="数值比较的容差（默认1e-6）"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("验证并修复样本数据")
    print("=" * 80)
    
    # 1. 读取样本数据文件
    print(f"\n1. 读取样本数据文件: {args.sample_file}")
    if not os.path.exists(args.sample_file):
        raise FileNotFoundError(f"文件不存在: {args.sample_file}")
    
    df_sample = pd.read_excel(args.sample_file, sheet_name=0)
    print(f"   形状: {df_sample.shape}")
    print(f"   列数: {len(df_sample.columns)}")
    
    # 识别日期列
    date_col_sample = _find_date_column(df_sample)
    if date_col_sample is None:
        raise ValueError("无法识别样本数据文件中的日期列")
    print(f"   日期列: {date_col_sample}")
    
    # 标准化日期
    df_sample[date_col_sample] = _to_datetime_series(df_sample[date_col_sample])
    
    # 2. 读取影响因子数据文件
    print(f"\n2. 读取影响因子数据文件: {args.factor_file}")
    if not os.path.exists(args.factor_file):
        raise FileNotFoundError(f"文件不存在: {args.factor_file}")
    
    df_factor = pd.read_excel(args.factor_file, sheet_name=0)
    print(f"   形状: {df_factor.shape}")
    print(f"   列数: {len(df_factor.columns)}")
    
    # 识别日期列
    date_col_factor = _find_date_column(df_factor)
    if date_col_factor is None:
        raise ValueError("无法识别影响因子文件中的日期列")
    print(f"   日期列: {date_col_factor}")
    
    # 标准化日期
    df_factor[date_col_factor] = _to_datetime_series(df_factor[date_col_factor])
    df_factor = df_factor.dropna(subset=[date_col_factor])
    
    print(f"   影响因子数据日期范围: {df_factor[date_col_factor].min()} 至 {df_factor[date_col_factor].max()}")
    
    # 3. 查找匹配的列
    print(f"\n3. 查找匹配的列...")
    
    # 排除的列（不需要验证的列）
    exclude_cols = [
        date_col_sample,
        'Unnamed: 0',
        '采样日期',
        '开花天数',
        '树上花果感染率%',
        '树下感染率%',
        '预警',
        'Unnamed: 5',
    ]
    
    matching_cols = find_matching_columns(df_sample, df_factor, exclude_cols)
    print(f"   找到 {len(matching_cols)} 个匹配的列")
    
    if len(matching_cols) == 0:
        print("   警告: 未找到匹配的列，请检查列名")
        print(f"   样本数据列名: {list(df_sample.columns)[:20]}")
        print(f"   影响因子列名: {list(df_factor.columns)[:20]}")
    else:
        print(f"   匹配的列（前10个）: {list(matching_cols.items())[:10]}")
    
    # 4. 创建备份（如果需要）
    if args.backup:
        backup_file = args.sample_file.replace(".xlsx", "_验证前备份.xlsx")
        print(f"\n4. 创建备份文件: {backup_file}")
        df_sample.to_excel(backup_file, index=False, engine='openpyxl')
        print(f"   备份完成")
    
    # 5. 验证和修复每个匹配的列
    print(f"\n5. 验证和修复数据...")
    
    all_stats = {}
    total_fixed = 0
    
    for col_sample, col_factor in matching_cols.items():
        if col_factor not in df_factor.columns:
            print(f"   跳过 {col_sample}（影响因子文件中不存在 {col_factor}）")
            continue
        
        print(f"\n   验证列: {col_sample} <-> {col_factor}")
        stats = validate_and_fix_column(
            df_sample,
            df_factor,
            date_col_sample,
            date_col_factor,
            col_sample,
            col_factor,
            tolerance=args.tolerance
        )
        
        all_stats[col_sample] = stats
        total_fixed += stats['fixed_rows']
        
        if stats['fixed_rows'] > 0:
            print(f"     ✓ 修复 {stats['fixed_rows']} 行")
            if stats['missing_in_sample'] > 0:
                print(f"       - 填充缺失值: {stats['missing_in_sample']} 行")
            if stats['mismatched_values'] > 0:
                print(f"       - 修正不匹配值: {stats['mismatched_values']} 行")
        else:
            print(f"     ✓ 无问题")
        
        if stats['missing_in_factor'] > 0:
            print(f"     ⚠ 影响因子中缺失: {stats['missing_in_factor']} 行")
    
    # 6. 保存验证修复后的文件
    print(f"\n6. 保存验证修复后的文件: {args.output_file}")
    
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
    
    # 7. 显示验证统计
    print(f"\n" + "=" * 80)
    print("验证修复统计")
    print("=" * 80)
    print(f"样本数据文件: {args.sample_file}")
    print(f"影响因子文件: {args.factor_file}")
    print(f"输出文件: {args.output_file}")
    print(f"验证的列数: {len(matching_cols)}")
    print(f"总修复行数: {total_fixed}")
    print(f"\n详细统计:")
    
    # 按修复数量排序
    sorted_stats = sorted(all_stats.items(), key=lambda x: x[1]['fixed_rows'], reverse=True)
    
    for col_name, stats in sorted_stats[:20]:  # 只显示前20个
        if stats['fixed_rows'] > 0:
            print(f"  {col_name}:")
            print(f"    总行数: {stats['total_rows']}, 匹配行数: {stats['matched_rows']}")
            print(f"    修复行数: {stats['fixed_rows']} (缺失: {stats['missing_in_sample']}, 不匹配: {stats['mismatched_values']})")
            if stats['missing_in_factor'] > 0:
                print(f"    影响因子中缺失: {stats['missing_in_factor']} 行")
    
    if len(sorted_stats) > 20:
        print(f"  ... 还有 {len(sorted_stats) - 20} 个列")
    
    print(f"\n✓ 验证修复完成！")


if __name__ == "__main__":
    main()

