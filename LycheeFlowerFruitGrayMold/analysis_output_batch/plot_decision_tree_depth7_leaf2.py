#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制深度7、叶子节点数2的决策树可视化图
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
from analyze_thresholds_single import (
    _find_date_column, _to_datetime_series, _normalize_warning_level,
    _find_warning_column, _select_feature_columns, run_analysis
)

def plot_decision_tree(data_xlsx, out_dir, max_depth=7, min_samples_leaf=2):
    """绘制决策树可视化图"""
    
    print("正在读取数据...")
    # 读取数据
    df = pd.read_excel(data_xlsx, sheet_name=0)
    
    # 识别日期列
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError("无法识别日期列")
    
    # 标准化日期
    df[date_col] = _to_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col])
    
    # 识别预警列
    warning_col = _find_warning_column(df)
    if not warning_col:
        raise ValueError("未找到预警列")
    
    # 排除预警为"未定义"的样本
    df = df[df[warning_col].astype(str).str.strip() != "未定义"]
    
    # 转换预警值为数值
    df["预警数值"] = df[warning_col].map(_normalize_warning_level)
    df = df.dropna(subset=["预警数值"])
    
    # 选择特征列
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("未匹配到气象因子列")
    
    print(f"识别到 {len(feature_cols)} 个气象因子特征")
    
    # 准备特征和标签
    X = df[feature_cols].astype(float).to_numpy()
    y_warning = df["预警数值"].astype(int).to_numpy()
    
    # 删除特征中有NaN的行
    mask_valid = ~np.isnan(X).any(axis=1)
    Xv = X[mask_valid]
    yw = y_warning[mask_valid]
    
    print(f"有效样本数: {len(Xv)}")
    print(f"预警等级分布: {np.bincount(yw)}")
    
    # 训练决策树
    print(f"\n训练决策树: max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    clf.fit(Xv, yw)
    
    # 预警等级标签
    warning_labels = {0: "0级(不发生)", 1: "1级(轻度)", 2: "2级(中度)", 3: "3级(重度)"}
    class_names = [warning_labels.get(i, str(i)) for i in range(4)]
    
    # 绘制决策树
    print("正在绘制决策树...")
    fig, ax = plt.subplots(figsize=(30, 20), dpi=150)
    
    plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    
    plt.title(
        f'荔枝霜疫霉预警等级决策树\n(max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, 样本数={len(Xv)})',
        fontsize=16,
        pad=20
    )
    
    # 保存图片
    output_file = os.path.join(out_dir, f"决策树_depth{max_depth}_leaf{min_samples_leaf}.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n决策树图已保存: {output_file}")
    
    # 同时保存为PDF格式（更清晰）
    output_file_pdf = os.path.join(out_dir, f"决策树_depth{max_depth}_leaf{min_samples_leaf}.pdf")
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"决策树图已保存(PDF): {output_file_pdf}")
    
    plt.close()
    
    # 输出决策树文本表示
    from sklearn.tree import export_text
    tree_rules = export_text(
        clf,
        feature_names=feature_cols,
        max_depth=10,
        spacing=2
    )
    
    text_file = os.path.join(out_dir, f"决策树_depth{max_depth}_leaf{min_samples_leaf}.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"决策树文本表示 (max_depth={max_depth}, min_samples_leaf={min_samples_leaf})\n")
        f.write("=" * 80 + "\n\n")
        f.write(tree_rules)
    print(f"决策树文本表示已保存: {text_file}")
    
    return output_file, output_file_pdf, text_file


def main():
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(project_root, "样本数据.xlsx")
    out_dir = os.path.dirname(__file__)  # analysis_output_batch目录
    
    print("=" * 80)
    print("绘制决策树可视化图 (depth7_leaf2)")
    print("=" * 80)
    print(f"\n数据文件: {data_file}")
    print(f"输出目录: {out_dir}\n")
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 {data_file}")
        return
    
    try:
        plot_decision_tree(
            data_xlsx=data_file,
            out_dir=out_dir,
            max_depth=7,
            min_samples_leaf=2
        )
        print("\n" + "=" * 80)
        print("决策树可视化完成！")
        print("=" * 80)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

