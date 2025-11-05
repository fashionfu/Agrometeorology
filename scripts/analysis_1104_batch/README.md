# 批量实验汇总报告

## 实验参数组合

测试了10种不同的决策树参数组合：

| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 使用特征数 | 最重要特征 | 规则总数 |
|---------|---------|-----------|------------------|-----------|-----------|----------|
| 1 | depth3_leaf5 | 3 | 5 | 失败 | 失败 | 失败 |
| 2 | depth4_leaf3 | 4 | 3 | 失败 | 失败 | 失败 |
| 3 | depth4_leaf2 | 4 | 2 | 失败 | 失败 | 失败 |
| 4 | depth5_leaf3 | 5 | 3 | 失败 | 失败 | 失败 |
| 5 | depth5_leaf2 | 5 | 2 | 失败 | 失败 | 失败 |
| 6 | depth6_leaf3 | 6 | 3 | 失败 | 失败 | 失败 |
| 7 | depth6_leaf2 | 6 | 2 | 失败 | 失败 | 失败 |
| 8 | depth7_leaf3 | 7 | 3 | 失败 | 失败 | 失败 |
| 9 | depth7_leaf2 | 7 | 2 | 失败 | 失败 | 失败 |
| 10 | depth5_leaf5 | 5 | 5 | 失败 | 失败 | 失败 |

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

