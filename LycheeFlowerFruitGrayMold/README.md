# 荔枝霜疫霉预警等级阈值模型分析

## 项目说明

本项目用于分析荔枝霜疫霉预警等级与气象因子的关系，通过皮尔逊相关分析、灰色关联分析和决策树模型，确定分预警等级的气象因子阈值。

## 文件结构

```
E:\1106lizhi\
├── 样本数据.xlsx          # 输入数据（包含19-25年采集的数据，预警列和气象因子）
├── scripts/                # 分析脚本目录
│   ├── analyze_thresholds_single.py    # 核心分析脚本（单次分析）
│   └── analyze_thresholds_batch.py   # 批量实验脚本
├── run_full_analysis.py    # 完整分析流程（推荐使用）
├── generate_summary.py    # 生成总结文档
├── 运行分析.bat            # Windows批处理文件（双击运行）
└── analysis_output/        # 分析结果输出目录（运行后生成）
```

## 使用方法

### 方法1: 使用批处理文件（推荐）

双击 `运行分析.bat` 文件，自动完成所有分析步骤。

### 方法2: 使用Python脚本

```bash
cd E:\1106lizhi
python run_full_analysis.py
```

### 方法3: 单独运行各步骤

```bash
# 1. 运行核心分析
python scripts/analyze_thresholds_single.py --data 样本数据.xlsx --out analysis_output

# 2. 生成总结文档
python generate_summary.py analysis_output
```

## 分析流程

1. **数据预处理**
   - 读取样本数据.xlsx
   - 排除预警为"未定义"的样本
   - 处理异常值和缺失值
   - 识别气象因子特征

2. **关联分析**
   - 皮尔逊相关分析：对每个预警等级进行二分类分析
   - 灰色关联分析（GRA）：衡量特征与目标序列的形状接近度
   - 综合评分：结合Pearson和GRA的结果

3. **模型构建**
   - 决策树多分类模型：使用预警等级（0-3级）作为目标变量
   - 特征重要性分析
   - 阈值规则提取

4. **结果输出**
   - 特征重要性文件
   - 各预警等级的特征评分
   - 各预警等级的阈值规则
   - 详细分析报告
   - 总结文档（1106.md）

## 输出文件说明

- `tree_feature_importances.csv`: 决策树特征重要性排序
- `feature_scores_warning_X.csv`: X级预警的特征评分（Pearson、GRA、综合评分）
- `tree_rules_warning_X.txt`: X级预警的阈值规则
- `1104.md`: 详细分析报告
- `1106.md`: 总结文档（包含5遍检查结果）

## 注意事项

1. 确保 `样本数据.xlsx` 文件存在且包含以下列：
   - 日期列（包含"日期"、"时间"等关键词）
   - 预警列（名称为"预警"）
   - 气象因子列（温度、湿度、降雨量、日照等）

2. 需要安装的Python包：
   - pandas
   - numpy
   - scikit-learn
   - openpyxl（用于读取Excel文件）

3. 如果遇到问题，请检查：
   - Python环境是否正确
   - 依赖包是否安装完整
   - 数据文件格式是否正确

## 联系信息

如有问题，请检查分析日志或联系项目维护人员。

