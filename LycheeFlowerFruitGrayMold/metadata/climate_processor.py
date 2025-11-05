import pandas as pd
import numpy as np

# 读取数据
print("Reading data...")
df = pd.read_excel('G1099.xlsx')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 获取数据列
result_df = pd.DataFrame()
result_df['日期'] = df.iloc[:, 0]

# 提取各个气候数据列
temp = pd.to_numeric(df.iloc[:, 1], errors='coerce')
rain = pd.to_numeric(df.iloc[:, 2], errors='coerce')
humidity = pd.to_numeric(df.iloc[:, 3], errors='coerce')
rain_hours = pd.to_numeric(df.iloc[:, 4], errors='coerce')
sunshine = pd.to_numeric(df.iloc[:, 5], errors='coerce')

# 计算不同时间窗口
windows = [3, 5, 7, 10]
total_rows = len(temp)

for window in windows:
    print(f"Processing {window}-day statistics...")
    
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

# 保存结果
print("Saving results...")
result_df.to_excel('影响因子1103.xlsx', index=False)
result_df.to_csv('影响因子1103.csv', index=False, encoding='utf-8-sig')

print("Done! Files saved:")
print("  - 影响因子1103.xlsx")
print("  - 影响因子1103.csv")
print(f"Total rows: {len(result_df)}")

