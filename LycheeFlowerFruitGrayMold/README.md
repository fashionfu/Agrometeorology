# 荔枝霜疫霉花果预警阈值模型

## 项目简介

本项目基于机器学习决策树算法，建立荔枝霜疫霉花果感病的预警阈值模型。通过分析气象因子与感病率的关系，提取可解释的预警规则，为荔枝霜疫霉的早期预警和防治提供科学依据。

## 主要功能

- **气象因子重要性分析**：使用皮尔逊相关系数和灰色关联分析（GRA）评估气象因子对感病率的影响
- **决策树规则提取**：从训练好的决策树模型中提取可解释的预警规则
- **多级预警阈值**：建立0级（不发生）、1级（轻度）、2级（中度）、3级（重度）四级预警体系
- **批量参数优化**：支持不同决策树深度和叶子节点数的批量实验
- **预警预测**：基于训练好的模型进行实时预警等级预测

## 项目结构

```
LycheeFlowerFruitGrayMold/
├── README.md                          # 项目说明文档（本文件）
├── scripts/                           # 核心脚本目录
│   ├── analyze_thresholds.py          # 基础阈值分析脚本（皮尔逊+GRA+决策树）
│   ├── analyze_thresholds_1104.py     # 1104版本分析脚本（支持多级预警）
│   ├── analyze_thresholds_1105.py     # 1105版本分析脚本（基于样本数据）
│   ├── analyze_thresholds_batch.py    # 批量参数优化脚本
│   ├── predict_warning.py             # 预警预测脚本
│   ├── generate_warning_excel.py      # 生成预警Excel文件
│   ├── generate_word_report.py        # 生成Word报告文档
│   ├── merge_sample_data.py           # 合并样本数据
│   └── ...                             # 其他辅助脚本
├── metadata/                           # 数据文件目录
│   ├── 影响因子1103_final.csv         # 气象因子数据（CSV格式）
│   ├── 影响因子1103_final.xlsx        # 气象因子数据（Excel格式）
│   ├── 样本数据.xlsx                  # 合并后的样本数据（包含预警等级）
│   ├── 张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20.xlsx
│   └── 张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx
├── analysis_1104/                     # 1104分析结果
│   ├── 1104_霜疫霉处理.md             # 分析报告
│   ├── tree_rules_warning_*.txt        # 各预警等级的决策树规则
│   └── feature_scores_warning_*.csv   # 各预警等级的特征重要性评分
├── analysis_1104_batch/               # 批量实验结果
│   ├── analysis_depth*_leaf*/         # 不同参数组合的结果
│   ├── 改善版.md                      # 综合分析报告
│   └── experiments_summary.csv        # 实验汇总表
└── analysis_1105/                     # 1105分析结果（最新版本）
    ├── 预警等级阈值规则报告.md         # 完整阈值规则报告
    ├── 预警等级阈值规则报告.docx       # Word格式报告
    ├── tree_rules_warning_*.txt        # 各预警等级的规则文件
    └── 特征重要性.csv                 # 特征重要性排序
```

## 环境要求

### Python版本
- Python 3.7+

### 依赖库
```bash
pip install pandas numpy scikit-learn openpyxl python-docx
```

主要依赖：
- `pandas`: 数据处理
- `numpy`: 数值计算
- `scikit-learn`: 机器学习（决策树）
- `openpyxl`: Excel文件读写
- `python-docx`: Word文档生成

## 快速开始

### 1. 数据准备

确保以下数据文件位于 `metadata/` 目录下：
- `影响因子1103_final.csv` - 气象因子数据
- `张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx` - 预警等级数据

### 2. 基础分析（推荐使用1105版本）

```bash
# 使用样本数据分析，提取0-3级预警阈值
python scripts/analyze_thresholds_1105.py --sample metadata/样本数据.xlsx --out analysis_1105
```

### 3. 预警预测

```bash
# 训练模型并保存
python scripts/predict_warning.py \
    --factors metadata/影响因子1103_final.csv \
    --warning metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx \
    --model models/warning_model.pkl

# 使用模型进行预测
python scripts/predict_warning.py \
    --predict metadata/当天气象数据.xlsx \
    --output 预测结果.xlsx \
    --model models/warning_model.pkl
```

## 核心脚本说明

### 1. `analyze_thresholds_1105.py` - 阈值规则提取（推荐）

**功能**：基于样本数据提取0-3级预警阈值规则，生成完整报告

**使用方法**：
```bash
python scripts/analyze_thresholds_1105.py \
    --sample metadata/样本数据.xlsx \
    --out analysis_1105 \
    --max_depth 4 \
    --min_samples_leaf 2
```

**输出**：
- `预警等级阈值规则报告.md` - Markdown格式报告
- `预警等级阈值规则报告.docx` - Word格式报告
- `tree_rules_warning_*.txt` - 各预警等级的规则文件
- `特征重要性.csv` - 特征重要性排序

### 2. `predict_warning.py` - 预警预测

**功能**：使用训练好的决策树模型进行预警等级预测

**使用方法**：
```bash
# 训练并保存模型
python scripts/predict_warning.py \
    --factors metadata/影响因子1103_final.csv \
    --warning metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx \
    --model models/warning_model.pkl

# 批量预测
python scripts/predict_warning.py \
    --predict metadata/新气象数据.xlsx \
    --output 预测结果.xlsx \
    --model models/warning_model.pkl
```

### 3. `analyze_thresholds_batch.py` - 批量参数优化

**功能**：批量测试不同决策树参数组合，找出最优配置

**使用方法**：
```bash
python scripts/analyze_thresholds_batch.py
```

**输出**：
- `analysis_1104_batch/analysis_depth*_leaf*/` - 各参数组合的结果
- `experiments_summary.csv` - 实验汇总表

### 4. `generate_word_report.py` - 生成Word报告

**功能**：将Markdown报告转换为Word文档

**使用方法**：
```bash
python scripts/generate_word_report.py \
    --input analysis_1105/预警等级阈值规则报告.md \
    --output analysis_1105/预警等级阈值规则报告.docx
```

## 预警等级判断流程

根据提取的阈值规则，预警判断顺序为：**0级 → 3级 → 2级 → 1级**

### 第一步：判断0级（不发生）
- 10天累积雨量 ≤ 19.5 mm
- 当天相对湿度 > 74.5%
- 10天累积日照时数 > 29.55 小时

满足 → **0级（不发生）**，无需发布预警

### 第二步：判断3级（重度）
- 10天累积雨量 > 19.5 mm
- 10天累积日照时数 ≤ 24.85 小时
- 当天平均气温 ≤ 23.05 °C

满足 → **3级（重度）预警**

### 第三步：判断2级（中度）
- 10天累积雨量 > 19.5 mm
- 10天累积日照时数 > 24.85 小时 且 ≤ 50.75 小时
- 当天相对湿度 ≤ 97.0%

满足 → **2级（中度）预警**

### 第四步：判断1级（轻度）
- 10天累积雨量 ≤ 19.5 mm
- 当天相对湿度 ≤ 74.5%

满足 → **1级（轻度）预警**

详细的阈值规则和判断流程请参考 `analysis_1105/预警等级阈值规则报告.md`

## 气象因子说明

模型使用的主要气象因子包括：

1. **10天累积雨量** (mm) - 过去10天的累积降雨量
2. **当天相对湿度** (%) - 当天的平均相对湿度
3. **10天累积日照时数** (小时) - 过去10天的累积日照时间
4. **当天平均气温** (°C) - 当天的平均气温
5. **5天平均相对湿度** (%) - 过去5天的平均相对湿度
6. **10天平均相对湿度** (%) - 过去10天的平均相对湿度

## 输出文件说明

### 分析报告
- **预警等级阈值规则报告.md/docx**: 完整的阈值规则报告，包含：
  - 数据描述和样本统计
  - 模型参数说明
  - 各预警等级的阈值规则
  - 气象因子阈值汇总表
  - 判断流程图

### 规则文件
- **tree_rules_warning_0.txt**: 0级（不发生）预警规则
- **tree_rules_warning_1.txt**: 1级（轻度）预警规则
- **tree_rules_warning_2.txt**: 2级（中度）预警规则
- **tree_rules_warning_3.txt**: 3级（重度）预警规则

### 特征重要性
- **特征重要性.csv**: 各气象因子的重要性排序（基于决策树特征重要性）

## 注意事项

1. **数据格式要求**：
   - 气象因子数据需包含日期列（含"日期"、"时间"、"日"等关键词）
   - 预警数据需包含"预警等级"列或感病率列（用于计算预警等级）

2. **文件路径**：
   - 所有数据文件应放在 `metadata/` 目录下
   - 脚本默认从 `metadata/` 目录读取数据文件

3. **模型使用**：
   - 建议使用完整的决策树模型进行预测，而不是单独使用规则
   - 规则主要用于理解和解释模型决策过程

4. **参数调整**：
   - `max_depth`: 决策树最大深度（建议3-7）
   - `min_samples_leaf`: 叶子节点最小样本数（建议2-5）

## 版本历史

- **v1105** (最新): 基于样本数据的完整分析，支持0-3级预警阈值提取
- **v1104**: 支持多级预警分析的早期版本
- **v1103**: 基础版本，支持皮尔逊相关和灰色关联分析

## GitHub仓库使用说明

### 推送到GitHub

本项目将作为子模块推送到以下仓库：
- **主仓库**: https://github.com/fashionfu/Agrometeorology.git
- **子目录**: `LycheeFlowerFruitGrayMold/`

### 推送步骤

```bash
# 1. 初始化git仓库（如果还没有）
git init

# 2. 添加远程仓库
git remote add origin https://github.com/fashionfu/Agrometeorology.git

# 3. 创建并切换到main分支（或master）
git checkout -b main

# 4. 添加所有文件
git add .

# 5. 提交更改
git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型"

# 6. 推送到远程仓库的LycheeFlowerFruitGrayMold目录
# 注意：如果主仓库已存在，需要先创建子目录
git push origin main:refs/heads/main/LycheeFlowerFruitGrayMold
```

或者，如果主仓库是空的或需要单独管理：

```bash
# 方法2：在主仓库中创建子目录
# 1. 克隆主仓库
git clone https://github.com/fashionfu/Agrometeorology.git
cd Agrometeorology

# 2. 创建子目录并复制文件
mkdir -p LycheeFlowerFruitGrayMold
# 将当前项目的所有文件复制到 LycheeFlowerFruitGrayMold 目录

# 3. 添加并提交
git add LycheeFlowerFruitGrayMold/
git commit -m "Add LycheeFlowerFruitGrayMold submodule"
git push origin main
```

## 贡献

如有问题或建议，欢迎提交Issue或Pull Request。

## 许可证

本项目为科研项目，仅供学术研究使用。

## 联系方式

项目仓库: https://github.com/fashionfu/Agrometeorology/tree/main/LycheeFlowerFruitGrayMold

---

**重要提示**：本项目为荔枝霜疫霉预警系统的研究性工作，实际应用时请结合当地气象条件和实际观测数据进行验证和调整。

