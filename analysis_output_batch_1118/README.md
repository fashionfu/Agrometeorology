# 批量实验汇总报告（强制使用'7天累计雨量'作为根节点）

## 实验参数组合

测试了10种不同的决策树参数组合，所有实验都强制使用'7天累计雨量'作为决策树的根节点：

| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 强制根节点特征 | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本总数 | 总体准确率 | 综合评分 |
|---------|---------|-----------|------------------|--------------|-----------|-----------|----------|------------|-----------|----------|
| 1 | depth3_leaf5 | 3 | 5 | 7天累积雨量 | 5 | 10天累积雨量 | 3 | 36 | 0.7778 | 28.00 |
| 2 | depth4_leaf3 | 4 | 3 | 7天累积雨量 | 6 | 10天累积雨量 | 4 | 34 | 0.9118 | 31.00 |
| 3 | depth4_leaf2 | 4 | 2 | 7天累积雨量 | 7 | 10天累积雨量 | 6 | 37 | 0.9189 | 34.00 |
| 4 | depth5_leaf3 | 5 | 3 | 7天累积雨量 | 7 | 10天累积雨量 | 4 | 30 | 1.0000 | 30.00 |
| 5 | depth5_leaf2 | 5 | 2 | 7天累积雨量 | 9 | 10天累积雨量 | 7 | 35 | 1.0000 | 35.00 |
| 6 | depth6_leaf3 | 6 | 3 | 7天累积雨量 | 7 | 10天累积雨量 | 4 | 28 | 1.0000 | 28.00 |
| 7 | depth6_leaf2 | 6 | 2 | 7天累积雨量 | 9 | 10天累积雨量 | 7 | 32 | 1.0000 | 32.00 |
| 8 | depth7_leaf3 | 7 | 3 | 7天累积雨量 | 7 | 10天累积雨量 | 4 | 26 | 1.0000 | 26.00 |
| 9 | depth7_leaf2 | 7 | 2 | 7天累积雨量 | 10 | 10天累积雨量 | 7 | 29 | 1.0000 | 29.00 |
| 10 | depth5_leaf5 | 5 | 5 | 7天累积雨量 | 8 | 10天累积雨量 | 3 | 28 | 1.0000 | 28.00 |

## 最优解法分析

### Top 3 最优解法（按综合评分排序）

综合评分 = 总体准确率 × 覆盖样本总数，综合考虑准确率和覆盖范围。

**注意**: 所有实验都强制使用'7天累计雨量'作为决策树的根节点。

#### 第1名：depth5_leaf2 (max_depth=5, min_samples_leaf=2)

- **综合评分**: 35.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 35
- **规则总数**: 7
- **使用特征数**: 9
- **强制根节点特征**: 7天累积雨量
- **最重要特征**: 10天累积雨量 (重要性=0.2849)
- **输出目录**: `F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉\analysis_output_batch_1118\analysis_depth5_leaf2`

#### 第2名：depth4_leaf2 (max_depth=4, min_samples_leaf=2)

- **综合评分**: 34.00
- **总体准确率**: 0.9189
- **覆盖样本总数**: 37
- **规则总数**: 6
- **使用特征数**: 7
- **强制根节点特征**: 7天累积雨量
- **最重要特征**: 10天累积雨量 (重要性=0.3335)
- **输出目录**: `F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉\analysis_output_batch_1118\analysis_depth4_leaf2`

#### 第3名：depth6_leaf2 (max_depth=6, min_samples_leaf=2)

- **综合评分**: 32.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 32
- **规则总数**: 7
- **使用特征数**: 9
- **强制根节点特征**: 7天累积雨量
- **最重要特征**: 10天累积雨量 (重要性=0.2725)
- **输出目录**: `F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉\analysis_output_batch_1118\analysis_depth6_leaf2`

### 详细对比分析

| 排名 | 参数名称 | max_depth | min_samples_leaf | 总体准确率 | 覆盖样本数 | 规则数 | 综合评分 |
|------|---------|-----------|------------------|-----------|-----------|--------|----------|
| 1 | depth5_leaf2 | 5 | 2 | 1.0000 | 35 | 7 | 35.00 |
| 2 | depth4_leaf2 | 4 | 2 | 0.9189 | 37 | 6 | 34.00 |
| 3 | depth6_leaf2 | 6 | 2 | 1.0000 | 32 | 7 | 32.00 |

### 最优解法特点分析

**最优解法（depth5_leaf2）的特点：**

1. **参数设置**: max_depth=5, min_samples_leaf=2
2. **强制根节点**: 所有实验都强制使用'7天累积雨量'作为决策树的根节点
3. **性能表现**: 在10个有效实验中，综合评分最高（35.00）
4. **准确率**: 1.0000，意味着该模型在覆盖的样本上预测准确率较高
5. **覆盖范围**: 覆盖了35个样本，说明规则具有良好的泛化能力
6. **特征选择**: 使用了9个特征，其中最重要特征是10天累积雨量
7. **规则数量**: 共7条规则，规则数量适中，既不过于复杂也不过于简单

**建议使用该参数组合进行最终的预警模型构建。**

**重要说明**: 本批次实验强制使用'7天累计雨量'作为决策树的根节点，所有规则都包含该特征作为第一个判断条件。


## 详细结果

每个实验的详细结果保存在对应的子文件夹中：

- **depth3_leaf5** (depth=3, leaf=5): `analysis_depth3_leaf5/`
  - 特征重要性: `analysis_depth3_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf5/1104.md`

- **depth4_leaf3** (depth=4, leaf=3): `analysis_depth4_leaf3/`
  - 特征重要性: `analysis_depth4_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf3/1104.md`

- **depth4_leaf2** (depth=4, leaf=2): `analysis_depth4_leaf2/`
  - 特征重要性: `analysis_depth4_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf2/1104.md`

- **depth5_leaf3** (depth=5, leaf=3): `analysis_depth5_leaf3/`
  - 特征重要性: `analysis_depth5_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf3/1104.md`

- **depth5_leaf2** (depth=5, leaf=2): `analysis_depth5_leaf2/`
  - 特征重要性: `analysis_depth5_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf2/1104.md`

- **depth6_leaf3** (depth=6, leaf=3): `analysis_depth6_leaf3/`
  - 特征重要性: `analysis_depth6_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf3/1104.md`

- **depth6_leaf2** (depth=6, leaf=2): `analysis_depth6_leaf2/`
  - 特征重要性: `analysis_depth6_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf2/1104.md`

- **depth7_leaf3** (depth=7, leaf=3): `analysis_depth7_leaf3/`
  - 特征重要性: `analysis_depth7_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf3/1104.md`

- **depth7_leaf2** (depth=7, leaf=2): `analysis_depth7_leaf2/`
  - 特征重要性: `analysis_depth7_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf2/1104.md`

- **depth5_leaf5** (depth=5, leaf=5): `analysis_depth5_leaf5/`
  - 特征重要性: `analysis_depth5_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf5/1104.md`

