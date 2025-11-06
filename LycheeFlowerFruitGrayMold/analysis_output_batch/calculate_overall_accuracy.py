#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算包含未覆盖样本的整体准确率
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from analyze_thresholds_single import run_analysis

def calculate_overall_accuracy(data_xlsx, out_dir, max_depth, min_samples_leaf):
    """计算包含所有样本的整体准确率"""
    # 读取数据
    df = pd.read_excel(data_xlsx, sheet_name=0)
    
    # 识别日期列和预警列
    date_col = None
    for c in df.columns:
        if any(tok in str(c) for tok in ["日期", "时间", "日", "date", "Date"]):
            date_col = str(c)
            break
    
    warning_col = None
    exact_match = [c for c in df.columns if str(c).strip() == "预警"]
    if exact_match:
        warning_col = str(exact_match[0])
    
    if not date_col or not warning_col:
        return None
    
    # 排除未定义的样本
    df = df[df[warning_col].astype(str).str.strip() != "未定义"]
    
    # 转换预警值
    def normalize_warning_level(value):
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s.lower() == "nan":
            return None
        if s.isdigit():
            val = int(s)
            if 0 <= val <= 3:
                return val
        if "重度" in s or "3" in s or "易发生" in s:
            return 3
        if "中度" in s or "2" in s or "较易" in s:
            return 2
        if "轻度" in s or "1" in s or "不易" in s:
            return 1
        if "0" in s or "不发生" in s or "无" in s:
            return 0
        return None
    
    df["预警数值"] = df[warning_col].map(normalize_warning_level)
    df = df.dropna(subset=["预警数值"])
    
    # 选择特征列
    def select_feature_columns(df):
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
    
    feature_cols = select_feature_columns(df)
    
    # 准备特征和标签
    X = df[feature_cols].astype(float).to_numpy()
    y_true = df["预警数值"].astype(int).to_numpy()
    
    # 删除特征中有NaN的行
    mask_valid = ~np.isnan(X).any(axis=1)
    X = X[mask_valid]
    y_true = y_true[mask_valid]
    
    # 训练决策树模型
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X, y_true)
    
    # 预测所有样本
    y_pred = clf.predict(X)
    
    # 计算整体准确率
    overall_accuracy = (y_pred == y_true).mean()
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': len(y_true),
        'correct_predictions': (y_pred == y_true).sum(),
        'wrong_predictions': (y_pred != y_true).sum()
    }

def main():
    data_xlsx = os.path.join(os.path.dirname(os.path.dirname(__file__)), '样本数据.xlsx')
    base_out_dir = os.path.dirname(__file__)
    
    # 读取汇总数据
    summary_file = os.path.join(base_out_dir, 'experiments_summary.csv')
    df_summary = pd.read_csv(summary_file, encoding='utf-8-sig')
    
    # 对每个实验计算整体准确率
    results = []
    for idx, row in df_summary.iterrows():
        if row['综合评分'] == '失败':
            continue
        
        param_name = row['参数名称']
        max_depth = int(row['max_depth'])
        min_samples_leaf = int(row['min_samples_leaf'])
        
        # 找到对应的输出目录
        out_dir = os.path.join(base_out_dir, f'analysis_{param_name}')
        
        print(f"计算 {param_name} 的整体准确率...")
        try:
            acc_result = calculate_overall_accuracy(
                data_xlsx, out_dir, max_depth, min_samples_leaf
            )
            
            if acc_result:
                results.append({
                    '参数名称': param_name,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    '整体准确率(含未覆盖)': acc_result['overall_accuracy'],
                    '总样本数': acc_result['total_samples'],
                    '正确预测数': acc_result['correct_predictions'],
                    '错误预测数': acc_result['wrong_predictions']
                })
                print(f"  ✓ 整体准确率: {acc_result['overall_accuracy']*100:.1f}%")
        except Exception as e:
            print(f"  ✗ 计算失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 合并结果
    df_results = pd.DataFrame(results)
    df_merged = df_summary.merge(df_results, on=['参数名称', 'max_depth', 'min_samples_leaf'], how='left')
    
    # 按综合评分排序
    df_merged['综合评分_数值'] = pd.to_numeric(df_merged['综合评分'], errors='coerce')
    df_merged = df_merged.sort_values('综合评分_数值', ascending=False)
    
    # 保存结果
    output_file = os.path.join(base_out_dir, 'experiments_summary_with_overall_accuracy.csv')
    df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    # 输出前三种方案
    print("\n" + "=" * 80)
    print("前三种最优方案（考虑未覆盖样本的整体准确率）")
    print("=" * 80)
    print()
    print("| 方案 | 参数组合 | max_depth | min_samples_leaf | 综合评分 | 整体准确率(含未覆盖) |")
    print("|------|---------|-----------|------------------|----------|---------------------|")
    
    for i, (idx, row) in enumerate(df_merged.head(3).iterrows(), 1):
        overall_acc = row['整体准确率(含未覆盖)']
        if pd.isna(overall_acc):
            acc_str = "N/A"
        else:
            acc_str = f"{overall_acc*100:.1f}%"
        
        score = row['综合评分']
        print(f"| 方案{i} | {row['参数名称']} | {row['max_depth']} | {row['min_samples_leaf']} | {score} | {acc_str} |")

if __name__ == '__main__':
    main()

