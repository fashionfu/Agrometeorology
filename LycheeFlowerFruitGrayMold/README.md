# 荔枝霜疫霉预警等级阈值模型分析

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目基于机器学习决策树模型，分析荔枝霜疫霉预警等级与气象因子的关系，通过批量实验优化模型参数，提取可解释的预警规则，为荔枝霜疫霉的早期预警提供科学依据。

### 核心功能

- 🔍 **多维度特征分析**：结合皮尔逊相关分析和灰色关联分析（GRA）评估气象因子重要性
- 🌳 **决策树模型**：构建可解释的预警等级分类模型
- 📊 **批量参数优化**：测试多种不同参数组合，寻找最优模型配置
- 📈 **规则提取**：自动提取各预警等级的气象因子阈值规则
- 📝 **可视化输出**：生成决策树可视化图和详细分析报告
- 🔄 **多版本分析**：支持不同时间点的数据分析（1118、1119、1120版本）

## 🎯 主要成果

### 最优模型配置

经过10种参数组合的批量实验，确定了三种推荐方案：

| 方案 | 参数组合 | max_depth | min_samples_leaf | 整体准确率 | 覆盖样本数 | 综合评分 |
|------|---------|-----------|------------------|-----------|-----------|----------|
| **方案一** | depth7_leaf2 | 7 | 2 | **93.0%** | 38 | **38.00** |
| **方案二** | depth6_leaf2 | 6 | 2 | **93.0%** | 36 | **36.00** |
| **方案三** | depth5_leaf2 | 5 | 2 | **89.5%** | 32 | **32.00** |

**推荐使用方案一（depth7_leaf2）**，该方案在整体准确率和覆盖范围方面表现最优。

### 关键发现

1. **最重要气象因子**：10天累积雨量是影响荔枝霜疫霉发生的最关键因子
2. **预警规则**：成功提取了0-3级预警的阈值规则，规则准确率均达到90%以上
3. **模型性能**：最优模型在57个有效样本上的整体准确率达到93.0%

## 📊 数据说明

### 数据来源

- **气象因子数据**：`样本数据.xlsx`（包含2019-2025年采集的数据）
- **目标变量**：预警等级（0-3级）
  - 0级：不发生
  - 1级：轻度
  - 2级：中度
  - 3级：重度

### 数据统计

- **有效样本数**：57条
- **样本分布**：
  - 0级（不发生）：14条
  - 1级（轻度）：13条
  - 2级（中度）：21条
  - 3级（重度）：9条

### 气象因子特征

模型分析了25个气象因子特征，包括：
- 温度相关：当天平均气温、最高气温、最低气温等
- 湿度相关：当天相对湿度、10天平均相对湿度等
- 降雨相关：当天雨量、3天累积雨量、5天累积雨量、10天累积雨量等
- 日照相关：当天日照时数、3天累积日照时数、5天累积日照时数、10天累积日照时数等

## 🔬 方法概述

### 1. 数据预处理

- 排除预警为"未定义"的样本
- 处理异常值和缺失值
- 特征标准化和归一化

### 2. 特征分析

- **皮尔逊相关分析**：评估线性关系
- **灰色关联分析（GRA）**：评估非线性关系和形状相似度
- **综合评分**：结合两种方法的结果

### 3. 模型构建

- **决策树分类器**：使用熵作为分裂准则
- **参数优化**：批量测试不同 `max_depth` 和 `min_samples_leaf` 组合
- **规则提取**：从决策树中提取可解释的阈值规则

### 4. 模型评估

- **规则准确率**：每个规则在覆盖样本上的预测准确率
- **覆盖样本数**：规则能够覆盖的样本数量
- **整体准确率**：考虑所有样本（包括未覆盖样本）的准确率
- **综合评分**：整体准确率 × 覆盖样本数

## 📁 项目结构

```
LycheeFlowerFruitGrayMold/
├── 样本数据.xlsx                    # 主数据文件（2019-2025年）
├── 样本数据_backup.xlsx             # 数据备份文件
├── 样本数据_验证修复.xlsx            # 验证修复后的数据
├── 影响因子1103_final_更新.xlsx     # 影响因子数据
│
├── scripts/                         # 分析脚本目录
│   ├── analyze_thresholds_single.py      # 单次分析脚本
│   ├── analyze_thresholds_batch.py        # 批量实验脚本（主版本）
│   ├── analyze_thresholds_batch_1118.py   # 1118版本批量分析
│   ├── analyze_thresholds_batch_1119.py   # 1119版本批量分析
│   └── analyze_thresholds_batch_1120.py   # 1120版本批量分析
│
├── analysis_output/                 # 单次分析结果
│   ├── tree_feature_importances.csv # 特征重要性
│   ├── feature_scores_warning_*.csv # 各预警等级特征评分
│   ├── tree_rules_warning_*.txt     # 各预警等级规则
│   ├── 1104.md                      # 1104分析报告
│   └── 1106.md                      # 1106分析报告
│
├── analysis_output_batch/           # 批量实验结果（主版本）
│   ├── README.md                    # 批量实验汇总报告
│   ├── experiments_summary.csv     # 实验汇总表
│   ├── experiments_summary_with_overall_accuracy.csv  # 含整体准确率的汇总表
│   ├── 荔枝霜疫霉分析结果.md        # 综合分析结果
│   ├── 荔枝霜疫霉分析结果.docx      # Word格式分析结果
│   ├── 最优解法对比分析.md          # 最优方案对比
│   ├── 最终前三种方案对比.md        # 前三种方案详细对比
│   ├── 最终解决方案.md              # 最终解决方案文档
│   ├── 决策树_depth7_leaf2.png      # 决策树可视化图（PNG）
│   ├── 决策树_depth7_leaf2.pdf      # 决策树可视化图（PDF）
│   ├── 决策树_depth7_leaf2.txt      # 决策树文本格式
│   ├── 样本预测统计表.xlsx          # 样本预测统计
│   ├── analysis_depth*_leaf*/       # 各参数组合的详细结果
│   └── [其他工具脚本]               # 数据转换、可视化等工具
│
├── analysis_output_batch_1118/      # 1118版本批量分析结果
│   ├── README.md                    # 1118版本汇总报告
│   ├── 1118分析.md                  # 1118分析报告
│   ├── 1118分析.docx                 # Word格式报告
│   ├── experiments_summary.csv      # 实验汇总
│   └── analysis_depth*_leaf*/      # 各参数组合结果
│
├── analysis_output_batch_1119/      # 1119版本批量分析结果
│   ├── [大量分析结果文件]           # 包含多个参数组合的详细结果
│   └── [相关分析报告]
│
├── analysis_output_batch_1120/      # 1120版本批量分析结果
│   ├── README.md                    # 1120版本汇总报告
│   ├── 1120分析.md                  # 1120分析报告
│   ├── experiments_summary.csv      # 实验汇总
│   └── analysis_depth*_leaf*/      # 各参数组合结果
│
├── TestScripts1119(测试样本异常)/   # 测试和验证脚本
│   ├── check_columns.py             # 列检查脚本
│   ├── replace_7day_rainfall_with_daily_avg.py  # 数据修复脚本
│   ├── update_sample_data_7day_avg_rainfall.py   # 数据更新脚本
│   ├── validate_and_fix_sample_data.py          # 数据验证修复
│   ├── 样本数据.xlsx                # 测试数据
│   └── 影响因子1103_final.xlsx     # 影响因子数据
│
├── run_analysis.py                  # 单次分析运行脚本
├── run_full_analysis.py             # 完整分析流程
├── check_data.py                    # 数据检查工具
├── generate_summary.py              # 汇总生成工具
├── update_2025_data.py              # 2025年数据更新工具
├── 运行分析.bat                     # Windows批处理运行脚本
├── 1121推送.bat                     # Git推送脚本
└── README.md                        # 本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- 依赖包：
  ```bash
  pip install pandas numpy scikit-learn openpyxl matplotlib
  ```

### 使用方法

#### 方法1：单次分析

```bash
# 运行单次分析（使用默认参数）
python scripts/analyze_thresholds_single.py --data 样本数据.xlsx --out analysis_output

# 指定参数
python scripts/analyze_thresholds_single.py \
    --data 样本数据.xlsx \
    --out analysis_output \
    --max-depth 7 \
    --min-samples-leaf 2
```

#### 方法2：批量实验（推荐）

```bash
# 运行批量实验，测试10种参数组合（主版本）
python scripts/analyze_thresholds_batch.py

# 运行特定版本的批量分析
python scripts/analyze_thresholds_batch_1120.py  # 1120版本
python scripts/analyze_thresholds_batch_1119.py  # 1119版本
python scripts/analyze_thresholds_batch_1118.py  # 1118版本
```

#### 方法3：使用批处理文件（Windows）

双击 `运行分析.bat` 文件即可自动运行分析。

#### 方法4：使用Python运行脚本

```bash
# 运行单次分析
python run_analysis.py

# 运行完整分析流程
python run_full_analysis.py
```

### 输出说明

#### 单次分析输出

- `tree_feature_importances.csv`：特征重要性排序
- `feature_scores_warning_X.csv`：X级预警的特征评分（Pearson、GRA、综合评分）
- `tree_rules_warning_X.txt`：X级预警的阈值规则
- `1106.md` 或 `1104.md`：详细分析报告

#### 批量实验输出

- `experiments_summary.csv`：所有实验的汇总统计
- `experiments_summary_with_overall_accuracy.csv`：包含整体准确率的详细汇总
- `README.md`：批量实验汇总报告
- `analysis_depth*_leaf*/`：每个参数组合的详细结果
- `荔枝霜疫霉分析结果.md`：综合分析结果文档
- `决策树_depth7_leaf2.png/pdf`：最优模型的决策树可视化图
- `最优解法对比分析.md`：最优方案详细对比分析
- `最终前三种方案对比.md`：前三种推荐方案的详细对比

## 📈 主要结果

### 预警规则示例

#### 0级（不发生）预警规则

**最优规则**：10天累积雨量 ≤ 19.5mm 且 当天相对湿度 > 74.5% 且 10天累积日照时数 > 29.55小时

- 覆盖样本：11条
- 准确率：100%

#### 1级（轻度）预警规则

**最优规则**：10天累积雨量 ≤ 19.5mm 且 当天相对湿度 ≤ 74.5%

- 覆盖样本：6条
- 准确率：100%

#### 2级（中度）预警规则

**最优规则**：10天累积雨量 > 19.5mm 且 10天累积日照时数 ≤ 5.9小时

- 覆盖样本：7条
- 准确率：100%

#### 3级（重度）预警规则

**最优规则**：10天累积雨量 > 19.5mm 且 10天累积日照时数 ≤ 5.9小时

- 覆盖样本：7条
- 准确率：100%

### 决策树可视化

项目提供了最优模型（depth7_leaf2）的决策树可视化图：

- `决策树_depth7_leaf2.png`：高分辨率PNG格式
- `决策树_depth7_leaf2.pdf`：矢量PDF格式
- `决策树_depth7_leaf2.txt`：文本格式

## 🔧 技术栈

- **Python 3.7+**
- **pandas**：数据处理
- **numpy**：数值计算
- **scikit-learn**：机器学习模型
- **matplotlib**：数据可视化
- **openpyxl**：Excel文件处理

## 🛠️ 工具脚本说明

### 数据相关工具

- `check_data.py`：检查数据完整性和格式
- `update_2025_data.py`：更新2025年数据
- `generate_summary.py`：生成分析汇总报告

### 分析相关工具

- `run_analysis.py`：运行单次分析
- `run_full_analysis.py`：运行完整分析流程

### 结果处理工具

- `analysis_output_batch/convert_md_to_docx.py`：将Markdown转换为Word文档
- `analysis_output_batch/plot_decision_tree_depth7_leaf2.py`：绘制决策树可视化图
- `analysis_output_batch/generate_prediction_comparison.py`：生成预测对比表

## 📚 详细文档

- [批量实验汇总报告](analysis_output_batch/README.md)
- [综合分析结果](analysis_output_batch/荔枝霜疫霉分析结果.md)
- [最优解法对比分析](analysis_output_batch/最优解法对比分析.md)
- [最终前三种方案对比](analysis_output_batch/最终前三种方案对比.md)
- [最终解决方案](analysis_output_batch/最终解决方案.md)

## 📝 版本说明

项目包含多个版本的分析结果：

- **主版本**：`analysis_output_batch/` - 最新完整分析结果
- **1120版本**：`analysis_output_batch_1120/` - 2025年11月20日分析
- **1119版本**：`analysis_output_batch_1119/` - 2025年11月19日分析
- **1118版本**：`analysis_output_batch_1118/` - 2025年11月18日分析

每个版本都包含完整的分析结果和报告，便于对比和追溯。

## 📝 引用

如果您使用本项目的研究成果，请引用：

```
荔枝霜疫霉预警等级阈值模型分析
基于决策树模型的荔枝霜疫霉预警等级阈值研究
2025
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 👥 作者

项目维护者：[您的名字/组织]

## 🙏 致谢

感谢所有为本项目提供数据和支持的合作伙伴。

---

**注意**：本项目仅用于科研目的，实际应用时请结合当地实际情况进行调整。
