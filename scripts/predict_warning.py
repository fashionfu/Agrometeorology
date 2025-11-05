#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据当天气象数据预测荔枝霜疫霉预警等级
"""

import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

# 导入分析脚本中的工具函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.analyze_thresholds_1104 import (
        _find_date_column,
        _to_datetime_series,
        _normalize_percent,
        _pick_infection_columns,
        _select_feature_columns,
        _classify_warning_level_from_tree  # 这个函数需要检查是否存在
    )
except ImportError:
    # 如果导入失败，定义必要的函数
    def _find_date_column(df):
        candidates = [c for c in df.columns if any(tok in str(c) for tok in ["日期", "时间", "日", "date", "Date"])]
        if candidates:
            return str(candidates[0])
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return str(c)
        return None
    
    def _to_datetime_series(s):
        return pd.to_datetime(s, errors="coerce").dt.normalize()
    
    def _normalize_percent(value):
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return None
            has_percent = "%" in s
            s_num = re.sub(r"[^0-9.\-]", "", s)
            if s_num in ("", ".", "-", "-."):
                return None
            try:
                num = float(s_num)
            except ValueError:
                return None
            if has_percent:
                return max(0.0, min(100.0, num))
            if 0 <= num <= 1:
                return num * 100.0
            return max(0.0, min(100.0, num))
        try:
            num = float(value)
        except Exception:
            return None
        if 0 <= num <= 1:
            return num * 100.0
        return max(0.0, min(100.0, num))
    
    def _select_feature_columns(df):
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
                    features.append(c)
        out = []
        for c in features:
            if c not in out:
                out.append(c)
        return out


def load_or_train_model(
    factors_csv: str,
    warning_xlsx: str,
    model_path: str = None,
    max_depth: int = 4,
    min_samples_leaf: int = 2
):
    """
    加载或训练决策树模型
    
    参数:
        factors_csv: 气象因子CSV文件路径
        warning_xlsx: 预警数据Excel文件路径
        model_path: 模型保存路径（如果存在则加载，否则训练并保存）
        max_depth: 决策树最大深度
        min_samples_leaf: 叶子节点最小样本数
    """
    # 如果模型文件存在，尝试加载
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                print(f"✓ 已加载模型: {model_path}")
                return model_data
        except Exception as e:
            print(f"警告: 无法加载模型文件 {model_path}: {e}")
            print("将重新训练模型...")
    
    # 重新训练模型
    print("正在读取训练数据...")
    met = pd.read_csv(factors_csv, encoding="utf-8", engine="python")
    warn = pd.read_excel(warning_xlsx, sheet_name=0)
    
    # 日期对齐
    met_date_col = _find_date_column(met)
    warn_date_col = _find_date_column(warn)
    if met_date_col is None or warn_date_col is None:
        raise ValueError("无法识别日期列")
    
    met[met_date_col] = _to_datetime_series(met[met_date_col])
    warn[warn_date_col] = _to_datetime_series(warn[warn_date_col])
    
    # 选择特征列
    feature_cols = _select_feature_columns(met)
    if not feature_cols:
        raise ValueError("未找到气象因子列")
    
    # 准备预警等级
    if "预警等级" not in warn.columns:
        # 如果没有预警等级列，需要根据感病率生成
        down_col, up_col = _pick_infection_columns(warn)
        if down_col and up_col:
            warn["预警等级"] = [
                _classify_warning_level(d, u) 
                for d, u in zip(
                    warn[down_col].map(_normalize_percent),
                    warn[up_col].map(_normalize_percent)
                )
            ]
        else:
            raise ValueError("未找到感病率列或预警等级列")
    
    # 合并数据
    df = pd.merge(met, warn[[warn_date_col, "预警等级"]], left_on=met_date_col, right_on=warn_date_col, how="inner")
    df = df[df["预警等级"] != "未定义"]  # 排除未定义的样本
    
    # 准备特征和目标
    X = df[feature_cols].astype(float).values
    # 将预警等级转换为数字：0, 1, 2, 3
    level_map = {"0": 0, "轻度": 1, "中度": 2, "重度": 3}
    y = df["预警等级"].map(lambda x: level_map.get(str(x).split("（")[0], -1))
    df = df[y >= 0]  # 只保留有效等级
    X = df[feature_cols].astype(float).values
    y = df["预警等级"].map(lambda x: level_map.get(str(x).split("（")[0], -1)).values
    
    # 删除包含NaN的行
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    print(f"训练样本数: {len(X)}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征名称: {feature_cols}")
    
    # 训练模型
    print(f"正在训练模型 (max_depth={max_depth}, min_samples_leaf={min_samples_leaf})...")
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X, y)
    
    # 保存模型
    model_data = {
        'model': clf,
        'feature_cols': feature_cols,
        'level_map': {0: "0级（不发生）", 1: "1级（轻度）", 2: "2级（中度）", 3: "3级（重度）"},
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf
    }
    
    if model_path:
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ 模型已保存: {model_path}")
    
    return model_data


def _classify_warning_level(down_rate, up_rate):
    """根据感病率分类预警等级"""
    if down_rate is None or up_rate is None:
        return "未定义"
    
    d = round(down_rate, 6)
    u = round(up_rate, 6)
    
    if d == 0 and u == 0:
        return "0"
    if d <= 10 and u == 0:
        return "轻度"
    if 10 < d <= 40 and u == 0:
        return "中度"
    if d > 40 and u > 0:
        return "重度"
    return "未定义"


def predict_warning_level(model_data, weather_data: dict):
    """
    根据气象数据预测预警等级
    
    参数:
        model_data: 模型数据字典
        weather_data: 包含气象因子的字典，键为特征名称
    
    返回:
        预警等级 (0, 1, 2, 3)
    """
    clf = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # 准备特征向量
    X = []
    for col in feature_cols:
        if col in weather_data:
            X.append(float(weather_data[col]))
        else:
            raise ValueError(f"缺少必需的气象因子: {col}")
    
    X = np.array(X).reshape(1, -1)
    
    # 预测
    predicted = clf.predict(X)[0]
    level_name = model_data['level_map'][predicted]
    
    return predicted, level_name


def predict_from_file(model_data, weather_file: str, output_file: str = None):
    """
    从文件读取气象数据并进行批量预测
    
    参数:
        model_data: 模型数据字典
        weather_file: 气象数据文件路径（支持.xlsx或.csv）
        output_file: 输出文件路径（可选）
    """
    print(f"正在读取气象数据: {weather_file}")
    
    # 读取数据
    if weather_file.endswith('.xlsx'):
        df = pd.read_excel(weather_file, sheet_name=0)
    elif weather_file.endswith('.csv'):
        df = pd.read_csv(weather_file, encoding='utf-8', engine='python')
    else:
        raise ValueError(f"不支持的文件格式: {weather_file}")
    
    feature_cols = model_data['feature_cols']
    date_col = _find_date_column(df)
    
    # 检查必需的列
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需的气象因子列: {missing_cols}")
    
    # 进行预测
    predictions_level = []
    predictions_name = []
    for idx, row in df.iterrows():
        weather_dict = {col: row[col] for col in feature_cols}
        try:
            level, level_name = predict_warning_level(model_data, weather_dict)
            predictions_level.append(level)
            predictions_name.append(level_name)
        except Exception as e:
            predictions_level.append(-1)
            predictions_name.append(f"错误: {str(e)}")
    
    # 创建结果DataFrame（复制原始数据）
    result_df = df.copy()
    result_df['预测预警等级'] = predictions_level
    result_df['预警等级说明'] = predictions_name
    
    # 保存结果
    if output_file:
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✓ 预测结果已保存: {output_file}")
    else:
        print("\n预测结果:")
        print(result_df.to_string())
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="根据气象数据预测荔枝霜疫霉预警等级")
    parser.add_argument(
        "--factors",
        default="metadata/影响因子1103_final.csv",
        help="气象因子CSV文件路径（用于训练模型）"
    )
    parser.add_argument(
        "--warning",
        default="metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx",
        help="预警数据Excel文件路径（用于训练模型）"
    )
    parser.add_argument(
        "--model",
        default="models/warning_model.pkl",
        help="模型文件路径（如果存在则加载，否则训练并保存）"
    )
    parser.add_argument(
        "--predict",
        help="要预测的气象数据文件路径（.xlsx或.csv）"
    )
    parser.add_argument(
        "--output",
        help="预测结果输出文件路径（.xlsx）"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="决策树最大深度（仅训练时使用）"
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="叶子节点最小样本数（仅训练时使用）"
    )
    
    args = parser.parse_args()
    
    # 加载或训练模型
    model_data = load_or_train_model(
        args.factors,
        args.warning,
        args.model,
        args.max_depth,
        args.min_samples_leaf
    )
    
    # 如果有预测文件，进行预测
    if args.predict:
        predict_from_file(model_data, args.predict, args.output)
    else:
        print("\n使用方法:")
        print("  python scripts/predict_warning.py --predict <气象数据文件> --output <输出文件>")


if __name__ == "__main__":
    main()
