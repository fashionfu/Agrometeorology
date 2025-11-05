#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
气候数据处理脚本
根据原始数据计算3天、5天、7天和10天的平均值和累积值
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

def read_excel_data(file_path):
    """读取Excel数据"""
    try:
        print("正在读取原始数据...")
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"成功读取数据，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"读取文件出错: {e}")
        sys.exit(1)

def process_climate_data(df):
    """
    处理气候数据，计算不同时间窗口的统计值
    
    参数:
        df: 原始数据DataFrame
    返回:
        处理后的DataFrame
    """
    print("\n原始数据列名:")
    print(df.columns.tolist())
    print("\n前5行数据:")
    print(df.head())
    print("\n数据形状:", df.shape)
    
    # 检查是否有足够的数据
    if df.shape[1] < 6:
        print(f"错误：数据只有 {df.shape[1]} 列，需要至少6列")
        sys.exit(1)
    
    # 直接使用列索引
    all_cols = df.columns.tolist()
    col_mapping = {
        'date': all_cols[0],
        'temp': all_cols[1],
        'rain': all_cols[2],
        'humidity': all_cols[3],
        'rain_hours': all_cols[4],
        'sunshine': all_cols[5]
    }
    
    print("\n使用列索引映射:")
    for key, val in col_mapping.items():
        print(f"  {key}: {val}")
    
    # 提取数据
    result_df = pd.DataFrame()
    result_df['日期'] = df[col_mapping['date']]
    
    # 确保日期是datetime类型
    result_df['日期'] = pd.to_datetime(result_df['日期'], errors='coerce')
    
    # 提取其他字段并转为数值型
    try:
        temp = pd.to_numeric(df[col_mapping['temp']], errors='coerce')
        rain = pd.to_numeric(df[col_mapping['rain']], errors='coerce')
        humidity = pd.to_numeric(df[col_mapping['humidity']], errors='coerce')
        rain_hours = pd.to_numeric(df[col_mapping['rain_hours']], errors='coerce')
        sunshine = pd.to_numeric(df[col_mapping['sunshine']], errors='coerce')
    except Exception as e:
        print(f"数据转换错误: {e}")
        sys.exit(1)
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  总记录数: {len(df)} 行")
    print(f"  日期范围: {result_df['日期'].min()} 到 {result_df['日期'].max()}")
    print(f"  平均气温范围: {temp.min():.2f} 到 {temp.max():.2f}")
    print(f"  雨量范围: {rain.min():.2f} 到 {rain.max():.2f}")
    
    # 检查缺失值
    missing = {
        '平均气温': temp.isna().sum(),
        '雨量': rain.isna().sum(),
        '相对湿度': humidity.isna().sum(),
        '降雨时数': rain_hours.isna().sum(),
        '日照时数': sunshine.isna().sum()
    }
    if any(missing.values()):
        print(f"\n缺失值统计:")
        for key, val in missing.items():
            if val > 0:
                print(f"  {key}: {val} 个缺失值")
    
    # 计算不同时间窗口的统计值
    windows = [3, 5, 7, 10]
    
    total_rows = len(temp)
    
    for window in windows:
        print(f"\n计算 {window} 天统计值...")
        
        # 平均气温
        temp_mean = []
        # 累积雨量
        rain_sum = []
        # 平均相对湿度
        humidity_mean = []
        # 累积降雨时数
        rain_hours_sum = []
        # 累积日照时数
        sunshine_sum = []
        
        for i in range(total_rows):
            # 计算从当前日期往前推window天的数据
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            # 提取窗口内的数据
            temp_window = temp.iloc[start_idx:end_idx]
            rain_window = rain.iloc[start_idx:end_idx]
            humidity_window = humidity.iloc[start_idx:end_idx]
            rain_hours_window = rain_hours.iloc[start_idx:end_idx]
            sunshine_window = sunshine.iloc[start_idx:end_idx]
            
            # 计算平均值或累积值
            temp_mean.append(temp_window.mean())
            rain_sum.append(rain_window.sum())
            humidity_mean.append(humidity_window.mean())
            rain_hours_sum.append(rain_hours_window.sum())
            sunshine_sum.append(sunshine_window.sum())
        
        # 添加到结果DataFrame
        result_df[f'{window}天平均气温'] = temp_mean
        result_df[f'{window}天累积雨量'] = rain_sum
        result_df[f'{window}天平均相对湿度'] = humidity_mean
        result_df[f'{window}天累积降雨时数'] = rain_hours_sum
        result_df[f'{window}天累积日照时数'] = sunshine_sum
    
    return result_df

def main():
    """主函数"""
    # 输入文件
    input_file = 'G1099.xlsx'
    
    # 读取数据
    df = read_excel_data(input_file)
    
    # 处理数据
    result_df = process_climate_data(df)
    
    # 保存结果
    output_file = '影响因子1103.xlsx'
    print(f"\n正在保存结果到 {output_file}...")
    result_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"数据已成功保存到 {output_file}")
    
    print("\n结果数据预览（前10行）:")
    print(result_df.head(10))
    
    # 同时保存为CSV
    csv_file = '影响因子1103.csv'
    print(f"\n正在保存CSV文件 {csv_file}...")
    result_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"CSV文件已成功保存")

if __name__ == '__main__':
    main()
