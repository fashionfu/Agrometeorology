#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试关键词匹配是否正确"""
import pandas as pd

def _select_feature_columns(df: pd.DataFrame):
    """选择气象因子列"""
    patterns = [
        ("日均温度", ["温度", "日均温度", "Tmean", "日平均气温", "平均温度", "平均气温", "气温"]),
        ("日均湿度", ["湿度", "日均湿度", "RH", "平均相对湿度"]),
        ("日均降雨量", ["降雨", "降雨量", "日降雨", "日均降雨", "降水", "降水量", "雨量", "Rain", "Precip"]),
        ("累计降雨量", ["累计降雨量", "累计降水", "累积降雨", "累积降水", "累积雨量", "RainCum", "PrecipCum"]),
        ("累计降雨时数", ["降雨时数", "降水时数", "累计降雨时数", "累计降水时数", "RainHours", "PrecipHours"]),
        ("累计日照", ["日照", "累计日照", "日照时数", "Sunshine", "Solar", "Radiation"]),
    ]
    features = []
    for _, keys in patterns:
        for c in df.columns:
            name = str(c)
            if any(k in name for k in keys):
                features.append(name)
    out = []
    for c in features:
        if c not in out:
            out.append(c)
    return out

# 读取数据
df = pd.read_excel("metadata/影响因子1103_final\.xlsx", sheet_name=0)
print(f"总列数: {len(df.columns)}")
print(f"所有列名: {list(df.columns)}")
print("\n" + "="*60)
print("特征列识别测试")
print("="*60)

feature_cols = _select_feature_columns(df)
print(f"\n识别到的特征列数: {len(feature_cols)}")
print("\n特征列列表:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

unidentified = [col for col in df.columns if col not in feature_cols and col != "日期"]
print(f"\n未识别的列（排除日期列）: {len(unidentified)}")
for i, col in enumerate(unidentified, 1):
    print(f"  {i}. {col}")

# 检查应该被识别的列
expected_cols = [
    "当天平均气温", "3天平均气温", "5天平均气温", "7天平均气温", "10天平均气温",
    "当天雨量", "3天累积雨量", "5天累积雨量", "7天累积雨量", "10天累积雨量"
]

print("\n" + "="*60)
print("验证应该被识别的列")
print("="*60)
missing = []
for col in expected_cols:
    if col in df.columns:
        if col in feature_cols:
            print(f"✅ {col} - 已识别")
        else:
            print(f"❌ {col} - 未识别")
            missing.append(col)
    else:
        print(f"⚠️  {col} - 列不存在")

if missing:
    print(f"\n⚠️  警告: {len(missing)} 个应该被识别的列未识别")
else:
    print("\n✅ 所有应该被识别的列都已正确识别！")


