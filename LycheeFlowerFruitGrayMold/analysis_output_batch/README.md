# 批量实验汇总报告

## 实验参数组合

测试了10种不同的决策树参数组合：

| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本总数 | 总体准确率 | 综合评分 |
|---------|---------|-----------|------------------|-----------|-----------|----------|------------|-----------|----------|
| 1 | depth3_leaf5 | 3 | 5 | 5 | 10天累积雨量 | 3 | 23 | 0.9565 | 22.00 |
| 2 | depth4_leaf3 | 4 | 3 | 6 | 10天累积雨量 | 4 | 29 | 0.9310 | 27.00 |
| 3 | depth4_leaf2 | 4 | 2 | 7 | 10天累积雨量 | 5 | 32 | 0.9375 | 30.00 |
| 4 | depth5_leaf3 | 5 | 3 | 7 | 10天累积雨量 | 5 | 32 | 0.9688 | 31.00 |
| 5 | depth5_leaf2 | 5 | 2 | 9 | 10天累积雨量 | 6 | 32 | 1.0000 | 32.00 |
| 6 | depth6_leaf3 | 6 | 3 | 7 | 10天累积雨量 | 5 | 29 | 1.0000 | 29.00 |
| 7 | depth6_leaf2 | 6 | 2 | 9 | 10天累积雨量 | 7 | 36 | 1.0000 | 36.00 |
| 8 | depth7_leaf3 | 7 | 3 | 7 | 10天累积雨量 | 5 | 29 | 1.0000 | 29.00 |
| 9 | depth7_leaf2 | 7 | 2 | 10 | 10天累积雨量 | 8 | 38 | 1.0000 | 38.00 |
| 10 | depth5_leaf5 | 5 | 5 | 8 | 10天累积雨量 | 5 | 32 | 0.9062 | 29.00 |

## 最优解法分析

### Top 3 最优解法（按综合评分排序）

综合评分 = 总体准确率 × 覆盖样本总数，综合考虑准确率和覆盖范围。

#### 第1名：depth7_leaf2 (max_depth=7, min_samples_leaf=2)

- **综合评分**: 38.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 38
- **规则总数**: 8
- **使用特征数**: 10
- **最重要特征**: 10天累积雨量 (重要性=0.2692)
- **输出目录**: `E:/1106lizhi/analysis_output_batch\analysis_depth7_leaf2`

#### 第2名：depth6_leaf2 (max_depth=6, min_samples_leaf=2)

- **综合评分**: 36.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 36
- **规则总数**: 7
- **使用特征数**: 9
- **最重要特征**: 10天累积雨量 (重要性=0.2725)
- **输出目录**: `E:/1106lizhi/analysis_output_batch\analysis_depth6_leaf2`

#### 第3名：depth5_leaf2 (max_depth=5, min_samples_leaf=2)

- **综合评分**: 32.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 32
- **规则总数**: 6
- **使用特征数**: 9
- **最重要特征**: 10天累积雨量 (重要性=0.2849)
- **输出目录**: `E:/1106lizhi/analysis_output_batch\analysis_depth5_leaf2`

### 详细对比分析

| 排名 | 参数名称 | max_depth | min_samples_leaf | 总体准确率 | 覆盖样本数 | 规则数 | 综合评分 |
|------|---------|-----------|------------------|-----------|-----------|--------|----------|
| 1 | depth7_leaf2 | 7 | 2 | 1.0000 | 38 | 8 | 38.00 |
| 2 | depth6_leaf2 | 6 | 2 | 1.0000 | 36 | 7 | 36.00 |
| 3 | depth5_leaf2 | 5 | 2 | 1.0000 | 32 | 6 | 32.00 |

### 最优解法特点分析

**最优解法（depth7_leaf2）的特点：**

1. **参数设置**: max_depth=7, min_samples_leaf=2
2. **性能表现**: 在10个有效实验中，综合评分最高（38.00）
3. **准确率**: 1.0000，意味着该模型在覆盖的样本上预测准确率较高
4. **覆盖范围**: 覆盖了38个样本，说明规则具有良好的泛化能力
5. **特征选择**: 使用了10个特征，其中最重要特征是10天累积雨量
6. **规则数量**: 共8条规则，规则数量适中，既不过于复杂也不过于简单

**建议使用该参数组合进行最终的预警模型构建。**


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

