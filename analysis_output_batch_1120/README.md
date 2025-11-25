# 批量实验汇总报告（基于新预警标准）

## 实验参数组合

测试了100种不同的决策树参数组合：

- **特征选择**: 只使用7天的特定特征（平均气温、日均降雨量、平均相对湿度、累积降雨时数、累积日照时数）
- **预警标准**: 直接使用数据文件中的'预警'列（0-3级），排除'未定义'的样本
- **根节点**: 强制使用'7天日均降雨量'作为决策树的根节点（第一个判断条件）
- **深度1节点**: 强制使用'7天平均气温'作为区分节点
- **其他子节点**: 使用所有特征（包括7天平均相对湿度、7天累计降雨时数、7天累计日照时数等）进行详细划分

| 实验编号 | 参数名称 | max_depth | min_samples_leaf | 训练样本数 | 验证样本数 | 验证准确率 | 使用特征数 | 最重要特征 | 规则总数 | 覆盖样本总数 | 总体准确率 | 综合评分 |
|---------|---------|-----------|------------------|-----------|-----------|-----------|-----------|-----------|----------|------------|-----------|----------|
| 1 | depth2_leaf1 | 2 | 1 | 44 | 13 | 0.5385 | 2 | 7天平均气温 | 0 | 0 | 0.0000 | 0.00 |
| 2 | depth2_leaf2 | 2 | 2 | 44 | 13 | 0.5385 | 2 | 7天平均气温 | 0 | 0 | 0.0000 | 0.00 |
| 3 | depth2_leaf3 | 2 | 3 | 44 | 13 | 0.5385 | 2 | 7天平均气温 | 0 | 0 | 0.0000 | 0.00 |
| 4 | depth2_leaf4 | 2 | 4 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 5 | depth2_leaf5 | 2 | 5 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 6 | depth2_leaf6 | 2 | 6 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 7 | depth2_leaf7 | 2 | 7 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 8 | depth2_leaf8 | 2 | 8 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 9 | depth2_leaf9 | 2 | 9 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 10 | depth2_leaf10 | 2 | 10 | 44 | 13 | 0.5385 | 2 | 7天日均降雨量 | 0 | 0 | 0.0000 | 0.00 |
| 11 | depth3_leaf1 | 3 | 1 | 44 | 13 | 0.6923 | 3 | 7天平均气温 | 3 | 30 | 0.6000 | 18.00 |
| 12 | depth3_leaf2 | 3 | 2 | 44 | 13 | 0.5385 | 3 | 7天平均气温 | 2 | 29 | 0.5862 | 17.00 |
| 13 | depth3_leaf3 | 3 | 3 | 44 | 13 | 0.5385 | 3 | 7天平均气温 | 1 | 26 | 0.5385 | 14.00 |
| 14 | depth3_leaf4 | 3 | 4 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 15 | depth3_leaf5 | 3 | 5 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 16 | depth3_leaf6 | 3 | 6 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 17 | depth3_leaf7 | 3 | 7 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 18 | depth3_leaf8 | 3 | 8 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 19 | depth3_leaf9 | 3 | 9 | 44 | 13 | 0.6923 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 20 | depth3_leaf10 | 3 | 10 | 44 | 13 | 0.6923 | 3 | 7天日均降雨量 | 1 | 26 | 0.5385 | 14.00 |
| 21 | depth4_leaf1 | 4 | 1 | 44 | 13 | 0.6923 | 3 | 7天日均降雨量 | 6 | 35 | 0.7429 | 26.00 |
| 22 | depth4_leaf2 | 4 | 2 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 4 | 34 | 0.7353 | 25.00 |
| 23 | depth4_leaf3 | 4 | 3 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 3 | 31 | 0.7097 | 22.00 |
| 24 | depth4_leaf4 | 4 | 4 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 2 | 28 | 0.6786 | 19.00 |
| 25 | depth4_leaf5 | 4 | 5 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 23 | 0.6522 | 15.00 |
| 26 | depth4_leaf6 | 4 | 6 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 23 | 0.6522 | 15.00 |
| 27 | depth4_leaf7 | 4 | 7 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 23 | 0.6522 | 15.00 |
| 28 | depth4_leaf8 | 4 | 8 | 44 | 13 | 0.5385 | 3 | 7天日均降雨量 | 1 | 23 | 0.6522 | 15.00 |
| 29 | depth4_leaf9 | 4 | 9 | 44 | 13 | 0.6923 | 3 | 7天日均降雨量 | 1 | 23 | 0.6522 | 15.00 |
| 30 | depth4_leaf10 | 4 | 10 | 44 | 13 | 0.6923 | 3 | 7天日均降雨量 | 1 | 23 | 0.6522 | 15.00 |
| 31 | depth5_leaf1 | 5 | 1 | 44 | 13 | 0.6923 | 5 | 7天日均降雨量 | 9 | 24 | 0.9583 | 23.00 |
| 32 | depth5_leaf2 | 5 | 2 | 44 | 13 | 0.5385 | 5 | 7天日均降雨量 | 4 | 22 | 0.9545 | 21.00 |
| 33 | depth5_leaf3 | 5 | 3 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 3 | 20 | 0.9000 | 18.00 |
| 34 | depth5_leaf4 | 5 | 4 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 2 | 17 | 0.8824 | 15.00 |
| 35 | depth5_leaf5 | 5 | 5 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 36 | depth5_leaf6 | 5 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 37 | depth5_leaf7 | 5 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 38 | depth5_leaf8 | 5 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 39 | depth5_leaf9 | 5 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 40 | depth5_leaf10 | 5 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 41 | depth6_leaf1 | 6 | 1 | 44 | 13 | 0.3077 | 5 | 7天平均气温 | 11 | 20 | 1.0000 | 20.00 |
| 42 | depth6_leaf2 | 6 | 2 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 4 | 17 | 1.0000 | 17.00 |
| 43 | depth6_leaf3 | 6 | 3 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 3 | 16 | 0.9375 | 15.00 |
| 44 | depth6_leaf4 | 6 | 4 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 2 | 13 | 0.9231 | 12.00 |
| 45 | depth6_leaf5 | 6 | 5 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 46 | depth6_leaf6 | 6 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 47 | depth6_leaf7 | 6 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 48 | depth6_leaf8 | 6 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 49 | depth6_leaf9 | 6 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 50 | depth6_leaf10 | 6 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 51 | depth7_leaf1 | 7 | 1 | 44 | 13 | 0.5385 | 5 | 7天平均气温 | 18 | 32 | 1.0000 | 32.00 |
| 52 | depth7_leaf2 | 7 | 2 | 44 | 13 | 0.3846 | 5 | 7天平均气温 | 8 | 27 | 1.0000 | 27.00 |
| 53 | depth7_leaf3 | 7 | 3 | 44 | 13 | 0.3846 | 5 | 7天日均降雨量 | 4 | 18 | 0.9444 | 17.00 |
| 54 | depth7_leaf4 | 7 | 4 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 2 | 12 | 0.9167 | 11.00 |
| 55 | depth7_leaf5 | 7 | 5 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 56 | depth7_leaf6 | 7 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 57 | depth7_leaf7 | 7 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 58 | depth7_leaf8 | 7 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 59 | depth7_leaf9 | 7 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 60 | depth7_leaf10 | 7 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 61 | depth8_leaf1 | 8 | 1 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 26 | 35 | 1.0000 | 35.00 |
| 62 | depth8_leaf2 | 8 | 2 | 44 | 13 | 0.3846 | 5 | 7天平均气温 | 8 | 26 | 1.0000 | 26.00 |
| 63 | depth8_leaf3 | 8 | 3 | 44 | 13 | 0.3846 | 5 | 7天日均降雨量 | 4 | 17 | 0.9412 | 16.00 |
| 64 | depth8_leaf4 | 8 | 4 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 2 | 12 | 0.9167 | 11.00 |
| 65 | depth8_leaf5 | 8 | 5 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 66 | depth8_leaf6 | 8 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 67 | depth8_leaf7 | 8 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 68 | depth8_leaf8 | 8 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 69 | depth8_leaf9 | 8 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 70 | depth8_leaf10 | 8 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 71 | depth9_leaf1 | 9 | 1 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 31 | 35 | 1.0000 | 35.00 |
| 72 | depth9_leaf2 | 9 | 2 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 8 | 25 | 1.0000 | 25.00 |
| 73 | depth9_leaf3 | 9 | 3 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 4 | 16 | 0.9375 | 15.00 |
| 74 | depth9_leaf4 | 9 | 4 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 2 | 12 | 0.9167 | 11.00 |
| 75 | depth9_leaf5 | 9 | 5 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 76 | depth9_leaf6 | 9 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 77 | depth9_leaf7 | 9 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 78 | depth9_leaf8 | 9 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 79 | depth9_leaf9 | 9 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 80 | depth9_leaf10 | 9 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 81 | depth10_leaf1 | 10 | 1 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 32 | 35 | 1.0000 | 35.00 |
| 82 | depth10_leaf2 | 10 | 2 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 8 | 24 | 1.0000 | 24.00 |
| 83 | depth10_leaf3 | 10 | 3 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 4 | 16 | 0.9375 | 15.00 |
| 84 | depth10_leaf4 | 10 | 4 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 2 | 12 | 0.9167 | 11.00 |
| 85 | depth10_leaf5 | 10 | 5 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 86 | depth10_leaf6 | 10 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 87 | depth10_leaf7 | 10 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 88 | depth10_leaf8 | 10 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 89 | depth10_leaf9 | 10 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 90 | depth10_leaf10 | 10 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 91 | depth11_leaf1 | 11 | 1 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 33 | 35 | 1.0000 | 35.00 |
| 92 | depth11_leaf2 | 11 | 2 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 8 | 23 | 1.0000 | 23.00 |
| 93 | depth11_leaf3 | 11 | 3 | 44 | 13 | 0.4615 | 5 | 7天平均气温 | 4 | 16 | 0.9375 | 15.00 |
| 94 | depth11_leaf4 | 11 | 4 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 2 | 12 | 0.9167 | 11.00 |
| 95 | depth11_leaf5 | 11 | 5 | 44 | 13 | 0.1538 | 5 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 96 | depth11_leaf6 | 11 | 6 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 8 | 1.0000 | 8.00 |
| 97 | depth11_leaf7 | 11 | 7 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 98 | depth11_leaf8 | 11 | 8 | 44 | 13 | 0.5385 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 99 | depth11_leaf9 | 11 | 9 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |
| 100 | depth11_leaf10 | 11 | 10 | 44 | 13 | 0.6923 | 4 | 7天日均降雨量 | 1 | 12 | 0.9167 | 11.00 |

## 最优解法分析

### Top 3 最优解法（按综合评分排序）

综合评分 = 总体准确率 × 覆盖样本总数，综合考虑准确率和覆盖范围。

#### 第1名：depth8_leaf1 (max_depth=8, min_samples_leaf=1)

- **综合评分**: 35.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 35
- **规则总数**: 26
- **使用特征数**: 5
- **最重要特征**: 7天平均气温 (重要性=2.7727)
- **训练 / 验证样本**: 44 / 13
- **验证准确率(2025)**: 0.4615
- **输出目录**: `F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉\analysis_output_batch_1120\analysis_depth8_leaf1`

#### 第2名：depth9_leaf1 (max_depth=9, min_samples_leaf=1)

- **综合评分**: 35.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 35
- **规则总数**: 31
- **使用特征数**: 5
- **最重要特征**: 7天平均气温 (重要性=3.0909)
- **训练 / 验证样本**: 44 / 13
- **验证准确率(2025)**: 0.4615
- **输出目录**: `F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉\analysis_output_batch_1120\analysis_depth9_leaf1`

#### 第3名：depth10_leaf1 (max_depth=10, min_samples_leaf=1)

- **综合评分**: 35.00
- **总体准确率**: 1.0000
- **覆盖样本总数**: 35
- **规则总数**: 32
- **使用特征数**: 5
- **最重要特征**: 7天平均气温 (重要性=3.2045)
- **训练 / 验证样本**: 44 / 13
- **验证准确率(2025)**: 0.4615
- **输出目录**: `F:\02_MeteorologyWork\02_正式\2025-11月正式工作\荔枝花果霜疫霉\analysis_output_batch_1120\analysis_depth10_leaf1`

### 详细对比分析

| 排名 | 参数名称 | max_depth | min_samples_leaf | 总体准确率 | 覆盖样本数 | 验证准确率 | 验证样本数 | 规则数 | 综合评分 |
|------|---------|-----------|------------------|-----------|-----------|-----------|-----------|--------|----------|
| 1 | depth8_leaf1 | 8 | 1 | 1.0000 | 35 | 0.4615 | 13 | 26 | 35.00 |
| 2 | depth9_leaf1 | 9 | 1 | 1.0000 | 35 | 0.4615 | 13 | 31 | 35.00 |
| 3 | depth10_leaf1 | 10 | 1 | 1.0000 | 35 | 0.4615 | 13 | 32 | 35.00 |

### 最优解法特点分析

**最优解法（depth8_leaf1）的特点：**

1. **参数设置**: max_depth=8, min_samples_leaf=1
2. **性能表现**: 在100个有效实验中，综合评分最高（35.00）
3. **准确率**: 1.0000，意味着该模型在覆盖的样本上预测准确率较高
4. **覆盖范围**: 覆盖了35个样本，说明规则具有良好的泛化能力
5. **验证表现**: 2025年验证样本 13 条，准确率 0.4615
6. **特征选择**: 使用了5个特征，其中最重要特征是7天平均气温
7. **规则数量**: 共26条规则，规则数量适中，既不过于复杂也不过于简单

**建议使用该参数组合进行最终的预警模型构建。**


## 详细结果

每个实验的详细结果保存在对应的子文件夹中：

- **depth2_leaf1** (depth=2, leaf=1): `analysis_depth2_leaf1/`
  - 特征重要性: `analysis_depth2_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf1/1120.md`

- **depth2_leaf2** (depth=2, leaf=2): `analysis_depth2_leaf2/`
  - 特征重要性: `analysis_depth2_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf2/1120.md`

- **depth2_leaf3** (depth=2, leaf=3): `analysis_depth2_leaf3/`
  - 特征重要性: `analysis_depth2_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf3/1120.md`

- **depth2_leaf4** (depth=2, leaf=4): `analysis_depth2_leaf4/`
  - 特征重要性: `analysis_depth2_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf4/1120.md`

- **depth2_leaf5** (depth=2, leaf=5): `analysis_depth2_leaf5/`
  - 特征重要性: `analysis_depth2_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf5/1120.md`

- **depth2_leaf6** (depth=2, leaf=6): `analysis_depth2_leaf6/`
  - 特征重要性: `analysis_depth2_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf6/1120.md`

- **depth2_leaf7** (depth=2, leaf=7): `analysis_depth2_leaf7/`
  - 特征重要性: `analysis_depth2_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf7/1120.md`

- **depth2_leaf8** (depth=2, leaf=8): `analysis_depth2_leaf8/`
  - 特征重要性: `analysis_depth2_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf8/1120.md`

- **depth2_leaf9** (depth=2, leaf=9): `analysis_depth2_leaf9/`
  - 特征重要性: `analysis_depth2_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf9/1120.md`

- **depth2_leaf10** (depth=2, leaf=10): `analysis_depth2_leaf10/`
  - 特征重要性: `analysis_depth2_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth2_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth2_leaf10/1120.md`

- **depth3_leaf1** (depth=3, leaf=1): `analysis_depth3_leaf1/`
  - 特征重要性: `analysis_depth3_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf1/1120.md`

- **depth3_leaf2** (depth=3, leaf=2): `analysis_depth3_leaf2/`
  - 特征重要性: `analysis_depth3_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf2/1120.md`

- **depth3_leaf3** (depth=3, leaf=3): `analysis_depth3_leaf3/`
  - 特征重要性: `analysis_depth3_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf3/1120.md`

- **depth3_leaf4** (depth=3, leaf=4): `analysis_depth3_leaf4/`
  - 特征重要性: `analysis_depth3_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf4/1120.md`

- **depth3_leaf5** (depth=3, leaf=5): `analysis_depth3_leaf5/`
  - 特征重要性: `analysis_depth3_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf5/1120.md`

- **depth3_leaf6** (depth=3, leaf=6): `analysis_depth3_leaf6/`
  - 特征重要性: `analysis_depth3_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf6/1120.md`

- **depth3_leaf7** (depth=3, leaf=7): `analysis_depth3_leaf7/`
  - 特征重要性: `analysis_depth3_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf7/1120.md`

- **depth3_leaf8** (depth=3, leaf=8): `analysis_depth3_leaf8/`
  - 特征重要性: `analysis_depth3_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf8/1120.md`

- **depth3_leaf9** (depth=3, leaf=9): `analysis_depth3_leaf9/`
  - 特征重要性: `analysis_depth3_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf9/1120.md`

- **depth3_leaf10** (depth=3, leaf=10): `analysis_depth3_leaf10/`
  - 特征重要性: `analysis_depth3_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth3_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth3_leaf10/1120.md`

- **depth4_leaf1** (depth=4, leaf=1): `analysis_depth4_leaf1/`
  - 特征重要性: `analysis_depth4_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf1/1120.md`

- **depth4_leaf2** (depth=4, leaf=2): `analysis_depth4_leaf2/`
  - 特征重要性: `analysis_depth4_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf2/1120.md`

- **depth4_leaf3** (depth=4, leaf=3): `analysis_depth4_leaf3/`
  - 特征重要性: `analysis_depth4_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf3/1120.md`

- **depth4_leaf4** (depth=4, leaf=4): `analysis_depth4_leaf4/`
  - 特征重要性: `analysis_depth4_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf4/1120.md`

- **depth4_leaf5** (depth=4, leaf=5): `analysis_depth4_leaf5/`
  - 特征重要性: `analysis_depth4_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf5/1120.md`

- **depth4_leaf6** (depth=4, leaf=6): `analysis_depth4_leaf6/`
  - 特征重要性: `analysis_depth4_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf6/1120.md`

- **depth4_leaf7** (depth=4, leaf=7): `analysis_depth4_leaf7/`
  - 特征重要性: `analysis_depth4_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf7/1120.md`

- **depth4_leaf8** (depth=4, leaf=8): `analysis_depth4_leaf8/`
  - 特征重要性: `analysis_depth4_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf8/1120.md`

- **depth4_leaf9** (depth=4, leaf=9): `analysis_depth4_leaf9/`
  - 特征重要性: `analysis_depth4_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf9/1120.md`

- **depth4_leaf10** (depth=4, leaf=10): `analysis_depth4_leaf10/`
  - 特征重要性: `analysis_depth4_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth4_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth4_leaf10/1120.md`

- **depth5_leaf1** (depth=5, leaf=1): `analysis_depth5_leaf1/`
  - 特征重要性: `analysis_depth5_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf1/1120.md`

- **depth5_leaf2** (depth=5, leaf=2): `analysis_depth5_leaf2/`
  - 特征重要性: `analysis_depth5_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf2/1120.md`

- **depth5_leaf3** (depth=5, leaf=3): `analysis_depth5_leaf3/`
  - 特征重要性: `analysis_depth5_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf3/1120.md`

- **depth5_leaf4** (depth=5, leaf=4): `analysis_depth5_leaf4/`
  - 特征重要性: `analysis_depth5_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf4/1120.md`

- **depth5_leaf5** (depth=5, leaf=5): `analysis_depth5_leaf5/`
  - 特征重要性: `analysis_depth5_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf5/1120.md`

- **depth5_leaf6** (depth=5, leaf=6): `analysis_depth5_leaf6/`
  - 特征重要性: `analysis_depth5_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf6/1120.md`

- **depth5_leaf7** (depth=5, leaf=7): `analysis_depth5_leaf7/`
  - 特征重要性: `analysis_depth5_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf7/1120.md`

- **depth5_leaf8** (depth=5, leaf=8): `analysis_depth5_leaf8/`
  - 特征重要性: `analysis_depth5_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf8/1120.md`

- **depth5_leaf9** (depth=5, leaf=9): `analysis_depth5_leaf9/`
  - 特征重要性: `analysis_depth5_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf9/1120.md`

- **depth5_leaf10** (depth=5, leaf=10): `analysis_depth5_leaf10/`
  - 特征重要性: `analysis_depth5_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth5_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth5_leaf10/1120.md`

- **depth6_leaf1** (depth=6, leaf=1): `analysis_depth6_leaf1/`
  - 特征重要性: `analysis_depth6_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf1/1120.md`

- **depth6_leaf2** (depth=6, leaf=2): `analysis_depth6_leaf2/`
  - 特征重要性: `analysis_depth6_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf2/1120.md`

- **depth6_leaf3** (depth=6, leaf=3): `analysis_depth6_leaf3/`
  - 特征重要性: `analysis_depth6_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf3/1120.md`

- **depth6_leaf4** (depth=6, leaf=4): `analysis_depth6_leaf4/`
  - 特征重要性: `analysis_depth6_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf4/1120.md`

- **depth6_leaf5** (depth=6, leaf=5): `analysis_depth6_leaf5/`
  - 特征重要性: `analysis_depth6_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf5/1120.md`

- **depth6_leaf6** (depth=6, leaf=6): `analysis_depth6_leaf6/`
  - 特征重要性: `analysis_depth6_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf6/1120.md`

- **depth6_leaf7** (depth=6, leaf=7): `analysis_depth6_leaf7/`
  - 特征重要性: `analysis_depth6_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf7/1120.md`

- **depth6_leaf8** (depth=6, leaf=8): `analysis_depth6_leaf8/`
  - 特征重要性: `analysis_depth6_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf8/1120.md`

- **depth6_leaf9** (depth=6, leaf=9): `analysis_depth6_leaf9/`
  - 特征重要性: `analysis_depth6_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf9/1120.md`

- **depth6_leaf10** (depth=6, leaf=10): `analysis_depth6_leaf10/`
  - 特征重要性: `analysis_depth6_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth6_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth6_leaf10/1120.md`

- **depth7_leaf1** (depth=7, leaf=1): `analysis_depth7_leaf1/`
  - 特征重要性: `analysis_depth7_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf1/1120.md`

- **depth7_leaf2** (depth=7, leaf=2): `analysis_depth7_leaf2/`
  - 特征重要性: `analysis_depth7_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf2/1120.md`

- **depth7_leaf3** (depth=7, leaf=3): `analysis_depth7_leaf3/`
  - 特征重要性: `analysis_depth7_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf3/1120.md`

- **depth7_leaf4** (depth=7, leaf=4): `analysis_depth7_leaf4/`
  - 特征重要性: `analysis_depth7_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf4/1120.md`

- **depth7_leaf5** (depth=7, leaf=5): `analysis_depth7_leaf5/`
  - 特征重要性: `analysis_depth7_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf5/1120.md`

- **depth7_leaf6** (depth=7, leaf=6): `analysis_depth7_leaf6/`
  - 特征重要性: `analysis_depth7_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf6/1120.md`

- **depth7_leaf7** (depth=7, leaf=7): `analysis_depth7_leaf7/`
  - 特征重要性: `analysis_depth7_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf7/1120.md`

- **depth7_leaf8** (depth=7, leaf=8): `analysis_depth7_leaf8/`
  - 特征重要性: `analysis_depth7_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf8/1120.md`

- **depth7_leaf9** (depth=7, leaf=9): `analysis_depth7_leaf9/`
  - 特征重要性: `analysis_depth7_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf9/1120.md`

- **depth7_leaf10** (depth=7, leaf=10): `analysis_depth7_leaf10/`
  - 特征重要性: `analysis_depth7_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth7_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth7_leaf10/1120.md`

- **depth8_leaf1** (depth=8, leaf=1): `analysis_depth8_leaf1/`
  - 特征重要性: `analysis_depth8_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf1/1120.md`

- **depth8_leaf2** (depth=8, leaf=2): `analysis_depth8_leaf2/`
  - 特征重要性: `analysis_depth8_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf2/1120.md`

- **depth8_leaf3** (depth=8, leaf=3): `analysis_depth8_leaf3/`
  - 特征重要性: `analysis_depth8_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf3/1120.md`

- **depth8_leaf4** (depth=8, leaf=4): `analysis_depth8_leaf4/`
  - 特征重要性: `analysis_depth8_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf4/1120.md`

- **depth8_leaf5** (depth=8, leaf=5): `analysis_depth8_leaf5/`
  - 特征重要性: `analysis_depth8_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf5/1120.md`

- **depth8_leaf6** (depth=8, leaf=6): `analysis_depth8_leaf6/`
  - 特征重要性: `analysis_depth8_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf6/1120.md`

- **depth8_leaf7** (depth=8, leaf=7): `analysis_depth8_leaf7/`
  - 特征重要性: `analysis_depth8_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf7/1120.md`

- **depth8_leaf8** (depth=8, leaf=8): `analysis_depth8_leaf8/`
  - 特征重要性: `analysis_depth8_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf8/1120.md`

- **depth8_leaf9** (depth=8, leaf=9): `analysis_depth8_leaf9/`
  - 特征重要性: `analysis_depth8_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf9/1120.md`

- **depth8_leaf10** (depth=8, leaf=10): `analysis_depth8_leaf10/`
  - 特征重要性: `analysis_depth8_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth8_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth8_leaf10/1120.md`

- **depth9_leaf1** (depth=9, leaf=1): `analysis_depth9_leaf1/`
  - 特征重要性: `analysis_depth9_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf1/1120.md`

- **depth9_leaf2** (depth=9, leaf=2): `analysis_depth9_leaf2/`
  - 特征重要性: `analysis_depth9_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf2/1120.md`

- **depth9_leaf3** (depth=9, leaf=3): `analysis_depth9_leaf3/`
  - 特征重要性: `analysis_depth9_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf3/1120.md`

- **depth9_leaf4** (depth=9, leaf=4): `analysis_depth9_leaf4/`
  - 特征重要性: `analysis_depth9_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf4/1120.md`

- **depth9_leaf5** (depth=9, leaf=5): `analysis_depth9_leaf5/`
  - 特征重要性: `analysis_depth9_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf5/1120.md`

- **depth9_leaf6** (depth=9, leaf=6): `analysis_depth9_leaf6/`
  - 特征重要性: `analysis_depth9_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf6/1120.md`

- **depth9_leaf7** (depth=9, leaf=7): `analysis_depth9_leaf7/`
  - 特征重要性: `analysis_depth9_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf7/1120.md`

- **depth9_leaf8** (depth=9, leaf=8): `analysis_depth9_leaf8/`
  - 特征重要性: `analysis_depth9_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf8/1120.md`

- **depth9_leaf9** (depth=9, leaf=9): `analysis_depth9_leaf9/`
  - 特征重要性: `analysis_depth9_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf9/1120.md`

- **depth9_leaf10** (depth=9, leaf=10): `analysis_depth9_leaf10/`
  - 特征重要性: `analysis_depth9_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth9_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth9_leaf10/1120.md`

- **depth10_leaf1** (depth=10, leaf=1): `analysis_depth10_leaf1/`
  - 特征重要性: `analysis_depth10_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf1/1120.md`

- **depth10_leaf2** (depth=10, leaf=2): `analysis_depth10_leaf2/`
  - 特征重要性: `analysis_depth10_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf2/1120.md`

- **depth10_leaf3** (depth=10, leaf=3): `analysis_depth10_leaf3/`
  - 特征重要性: `analysis_depth10_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf3/1120.md`

- **depth10_leaf4** (depth=10, leaf=4): `analysis_depth10_leaf4/`
  - 特征重要性: `analysis_depth10_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf4/1120.md`

- **depth10_leaf5** (depth=10, leaf=5): `analysis_depth10_leaf5/`
  - 特征重要性: `analysis_depth10_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf5/1120.md`

- **depth10_leaf6** (depth=10, leaf=6): `analysis_depth10_leaf6/`
  - 特征重要性: `analysis_depth10_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf6/1120.md`

- **depth10_leaf7** (depth=10, leaf=7): `analysis_depth10_leaf7/`
  - 特征重要性: `analysis_depth10_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf7/1120.md`

- **depth10_leaf8** (depth=10, leaf=8): `analysis_depth10_leaf8/`
  - 特征重要性: `analysis_depth10_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf8/1120.md`

- **depth10_leaf9** (depth=10, leaf=9): `analysis_depth10_leaf9/`
  - 特征重要性: `analysis_depth10_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf9/1120.md`

- **depth10_leaf10** (depth=10, leaf=10): `analysis_depth10_leaf10/`
  - 特征重要性: `analysis_depth10_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth10_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth10_leaf10/1120.md`

- **depth11_leaf1** (depth=11, leaf=1): `analysis_depth11_leaf1/`
  - 特征重要性: `analysis_depth11_leaf1/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf1/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf1/1120.md`

- **depth11_leaf2** (depth=11, leaf=2): `analysis_depth11_leaf2/`
  - 特征重要性: `analysis_depth11_leaf2/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf2/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf2/1120.md`

- **depth11_leaf3** (depth=11, leaf=3): `analysis_depth11_leaf3/`
  - 特征重要性: `analysis_depth11_leaf3/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf3/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf3/1120.md`

- **depth11_leaf4** (depth=11, leaf=4): `analysis_depth11_leaf4/`
  - 特征重要性: `analysis_depth11_leaf4/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf4/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf4/1120.md`

- **depth11_leaf5** (depth=11, leaf=5): `analysis_depth11_leaf5/`
  - 特征重要性: `analysis_depth11_leaf5/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf5/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf5/1120.md`

- **depth11_leaf6** (depth=11, leaf=6): `analysis_depth11_leaf6/`
  - 特征重要性: `analysis_depth11_leaf6/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf6/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf6/1120.md`

- **depth11_leaf7** (depth=11, leaf=7): `analysis_depth11_leaf7/`
  - 特征重要性: `analysis_depth11_leaf7/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf7/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf7/1120.md`

- **depth11_leaf8** (depth=11, leaf=8): `analysis_depth11_leaf8/`
  - 特征重要性: `analysis_depth11_leaf8/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf8/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf8/1120.md`

- **depth11_leaf9** (depth=11, leaf=9): `analysis_depth11_leaf9/`
  - 特征重要性: `analysis_depth11_leaf9/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf9/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf9/1120.md`

- **depth11_leaf10** (depth=11, leaf=10): `analysis_depth11_leaf10/`
  - 特征重要性: `analysis_depth11_leaf10/tree_feature_importances.csv`
  - 预警规则: `analysis_depth11_leaf10/tree_rules_warning_*.txt`
  - 详细报告: `analysis_depth11_leaf10/1120.md`

