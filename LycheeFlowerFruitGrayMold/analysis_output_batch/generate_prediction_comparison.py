#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成决策树预测结果与实际观测的对比Excel文件
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from analyze_thresholds_single import (
    _find_date_column, _to_datetime_series, _normalize_warning_level,
    _find_warning_column, _select_feature_columns
)

def generate_prediction_comparison(
    data_xlsx: str,
    output_excel: str,
    max_depth: int = 7,
    min_samples_leaf: int = 2
):
    """生成预测结果对比Excel文件"""
    
    print("=" * 80)
    print("生成决策树预测结果与实际观测对比表")
    print("=" * 80)
    print(f"\n数据文件: {data_xlsx}")
    print(f"输出文件: {output_excel}")
    print(f"模型参数: max_depth={max_depth}, min_samples_leaf={min_samples_leaf}\n")
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(data_xlsx, sheet_name=0)
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 识别日期列
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError("无法识别日期列")
    print(f"日期列: {date_col}")
    
    # 标准化日期
    df[date_col] = _to_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col])
    
    # 识别预警列
    warning_col = _find_warning_column(df)
    if not warning_col:
        raise ValueError("未找到预警列")
    print(f"预警列: {warning_col}")
    
    # 排除预警为"未定义"的样本
    before_count = len(df)
    df = df[df[warning_col].astype(str).str.strip() != "未定义"]
    after_count = len(df)
    print(f"排除'未定义'样本: {before_count} -> {after_count} (排除 {before_count - after_count} 条)")
    
    # 转换预警值为数值
    df["实际预警等级"] = df[warning_col].map(_normalize_warning_level)
    before_drop = len(df)
    df = df.dropna(subset=["实际预警等级"])
    after_drop = len(df)
    print(f"有效数据: {before_drop} -> {after_drop} 条（删除 {before_drop - after_drop} 条）")
    
    # 选择特征列
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("未匹配到气象因子列")
    print(f"识别到 {len(feature_cols)} 个气象因子特征")
    
    # 准备特征和标签
    X = df[feature_cols].astype(float).to_numpy()
    y_true = df["实际预警等级"].astype(int).to_numpy()
    
    # 删除特征中有NaN的行
    mask_valid = ~np.isnan(X).any(axis=1)
    X_valid = X[mask_valid]
    y_true_valid = y_true[mask_valid]
    df_valid = df[mask_valid].copy()
    
    print(f"有效样本（无缺失特征）: {mask_valid.sum()} / {len(mask_valid)}")
    
    # 训练决策树模型
    print(f"\n训练决策树模型 (max_depth={max_depth}, min_samples_leaf={min_samples_leaf})...")
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X_valid, y_true_valid)
    
    # 预测所有样本
    print("正在预测所有样本...")
    y_pred = clf.predict(X_valid)
    
    # 计算准确率
    accuracy = (y_pred == y_true_valid).mean()
    print(f"整体准确率: {accuracy*100:.2f}%")
    print(f"正确预测: {(y_pred == y_true_valid).sum()} / {len(y_pred)}")
    print(f"错误预测: {(y_pred != y_true_valid).sum()} / {len(y_pred)}")
    
    # 预警等级标签
    warning_labels = {
        0: "0级（不发生）",
        1: "1级（轻度）",
        2: "2级（中度）",
        3: "3级（重度）"
    }
    
    # 构建对比数据框
    print("\n正在构建对比数据...")
    comparison_data = {
        "日期": df_valid[date_col].values,
        "实际预警等级": y_true_valid,
        "实际预警等级描述": [warning_labels.get(int(level), f"{level}级") for level in y_true_valid],
        "预测预警等级": y_pred,
        "预测预警等级描述": [warning_labels.get(int(level), f"{level}级") for level in y_pred],
        "是否匹配": ["是" if pred == true else "否" for pred, true in zip(y_pred, y_true_valid)],
        "预测准确性": ["正确" if pred == true else "错误" for pred, true in zip(y_pred, y_true_valid)]
    }
    
    # 添加关键气象因子（最重要的几个）
    important_features = [
        "10天累积雨量",
        "当天相对湿度",
        "10天累积日照时数",
        "3天累积降雨时数",
        "5天累积日照时数",
        "5天累积雨量",
        "当天平均气温"
    ]
    
    for feat_name in important_features:
        # 查找匹配的特征列
        matching_cols = [col for col in feature_cols if feat_name in col]
        if matching_cols:
            comparison_data[feat_name] = df_valid[matching_cols[0]].values
        else:
            comparison_data[feat_name] = [None] * len(df_valid)
    
    # 创建DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # 重新排列列的顺序
    column_order = [
        "日期",
        "实际预警等级",
        "实际预警等级描述",
        "预测预警等级",
        "预测预警等级描述",
        "是否匹配",
        "预测准确性"
    ] + important_features
    
    df_comparison = df_comparison[column_order]
    
    # 按日期排序
    df_comparison = df_comparison.sort_values("日期")
    
    # 保存为Excel文件
    print(f"\n正在保存到Excel文件: {output_excel}")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 主对比表
        df_comparison.to_excel(writer, sheet_name='预测对比表', index=False)
        
        # 统计汇总表
        summary_data = []
        for level in [0, 1, 2, 3]:
            label = warning_labels.get(level, f"{level}级")
            actual_count = (y_true_valid == level).sum()
            pred_count = (y_pred == level).sum()
            correct_count = ((y_true_valid == level) & (y_pred == level)).sum()
            accuracy_level = correct_count / actual_count if actual_count > 0 else 0.0
            
            summary_data.append({
                "预警等级": label,
                "实际样本数": actual_count,
                "预测样本数": pred_count,
                "正确预测数": correct_count,
                "准确率": f"{accuracy_level*100:.2f}%"
            })
        
        # 添加总体统计
        summary_data.append({
            "预警等级": "总计",
            "实际样本数": len(y_true_valid),
            "预测样本数": len(y_pred),
            "正确预测数": (y_pred == y_true_valid).sum(),
            "准确率": f"{accuracy*100:.2f}%"
        })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='统计汇总', index=False)
        
        # 错误预测详情表
        error_mask = y_pred != y_true_valid
        if error_mask.sum() > 0:
            df_errors = df_comparison[error_mask].copy()
            df_errors.to_excel(writer, sheet_name='错误预测详情', index=False)
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_valid, y_pred, labels=[0, 1, 2, 3])
        cm_df = pd.DataFrame(
            cm,
            index=[f"实际{label}" for label in warning_labels.values()],
            columns=[f"预测{label}" for label in warning_labels.values()]
        )
        cm_df.to_excel(writer, sheet_name='混淆矩阵')
    
    print(f"✓ Excel文件已保存: {output_excel}")
    print(f"\n文件包含以下工作表:")
    print(f"  1. 预测对比表 - 所有样本的详细对比")
    print(f"  2. 统计汇总 - 各预警等级的统计信息")
    if error_mask.sum() > 0:
        print(f"  3. 错误预测详情 - 预测错误的样本详情")
    print(f"  4. 混淆矩阵 - 预测结果的混淆矩阵")
    
    return df_comparison


def main():
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(project_root, "样本数据.xlsx")
    output_dir = os.path.dirname(__file__)
    
    # 使用最优模型参数（depth7_leaf2）
    output_file = os.path.join(output_dir, "决策树预测结果对比表_depth7_leaf2.xlsx")
    
    try:
        df_comparison = generate_prediction_comparison(
            data_xlsx=data_file,
            output_excel=output_file,
            max_depth=7,
            min_samples_leaf=2
        )
        
        print("\n" + "=" * 80)
        print("生成完成！")
        print("=" * 80)
        print(f"\n输出文件: {output_file}")
        print(f"总样本数: {len(df_comparison)}")
        print(f"正确预测: {(df_comparison['是否匹配'] == '是').sum()}")
        print(f"错误预测: {(df_comparison['是否匹配'] == '否').sum()}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

