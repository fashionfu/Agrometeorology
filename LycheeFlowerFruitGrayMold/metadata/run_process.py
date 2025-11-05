"""
直接运行的气候数据处理脚本
简化版本，确保能正常运行
"""

import pandas as pd
import os

print("=" * 60)
print("气候数据处理程序")
print("=" * 60)

# 检查输入文件是否存在
input_file = 'G1099.xlsx'
if not os.path.exists(input_file):
    print(f"错误：找不到输入文件 {input_file}")
    print("请确保 G1099.xlsx 文件在当前目录中")
    input("按回车键退出...")
    exit(1)

print(f"\n正在读取: {input_file}")

try:
    # 读取数据
    df = pd.read_excel(input_file)
    print(f"✓ 成功读取数据，共 {len(df)} 行")
    
    # 显示数据信息
    print(f"\n数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"\n列名:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")
    
    # 检查列数
    if df.shape[1] < 6:
        print(f"\n错误：数据只有 {df.shape[1]} 列，需要至少6列")
        input("按回车键退出...")
        exit(1)
    
    # 提取数据
    print("\n正在处理数据...")
    result_df = pd.DataFrame()
    result_df['日期'] = df.iloc[:, 0]
    
    # 转换为数值型
    temp = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    rain = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    humidity = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    rain_hours = pd.to_numeric(df.iloc[:, 4], errors='coerce')
    sunshine = pd.to_numeric(df.iloc[:, 5], errors='coerce')
    
    # 添加当天数据列
    result_df['当天平均气温'] = temp
    result_df['当天雨量'] = rain
    result_df['当天相对湿度'] = humidity
    result_df['当天降雨时数'] = rain_hours
    result_df['当天日照时数'] = sunshine
    print("  ✓ 已添加当天数据列")
    
    total_rows = len(temp)
    print(f"  数据行数: {total_rows}")
    
    # 计算不同时间窗口
    windows = [3, 5, 7, 10]
    
    for window in windows:
        print(f"  正在计算 {window} 天统计值...")
        
        temp_mean = []
        rain_sum = []
        humidity_mean = []
        rain_hours_sum = []
        sunshine_sum = []
        
        for i in range(total_rows):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            temp_mean.append(temp.iloc[start_idx:end_idx].mean())
            rain_sum.append(rain.iloc[start_idx:end_idx].sum())
            humidity_mean.append(humidity.iloc[start_idx:end_idx].mean())
            rain_hours_sum.append(rain_hours.iloc[start_idx:end_idx].sum())
            sunshine_sum.append(sunshine.iloc[start_idx:end_idx].sum())
        
        result_df[f'{window}天平均气温'] = temp_mean
        result_df[f'{window}天累积雨量'] = rain_sum
        result_df[f'{window}天平均相对湿度'] = humidity_mean
        result_df[f'{window}天累积降雨时数'] = rain_hours_sum
        result_df[f'{window}天累积日照时数'] = sunshine_sum
    
    # 保存结果（带final后缀）
    print("\n正在保存结果...")
    
    output_xlsx = 'metadata/影响因子1103_final\.xlsx'
    result_df.to_excel(output_xlsx, index=False)
    print(f"✓ {output_xlsx} 已生成")
    
    output_csv = 'metadata/影响因子1103_final\.csv'
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ {output_csv} 已生成")
    
    # 显示结果预览
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n结果预览（前5行）:")
    print(result_df.head().to_string())
    
    print(f"\n输出列数: {len(result_df.columns)} 列")
    print(f"输出行数: {len(result_df)} 行")
    
except Exception as e:
    print(f"\n错误: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n按回车键退出...")
input()

