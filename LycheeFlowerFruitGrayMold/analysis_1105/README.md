# 样本数据阈值分析说明

## 使用方法

请运行以下命令进行分析：

```bash
python scripts/analyze_thresholds_1105.py --sample 样本数据.xlsx --out analysis_1105
```

或者使用完整路径：

```bash
python scripts/analyze_thresholds_1105.py --sample "metadata/样本数据\.xlsx" --out "analysis_1105"
```

## 输出文件

分析完成后，`analysis_1105` 文件夹将包含：

1. **预警等级阈值规则报告.md** - 完整的Markdown格式报告，包含0-3级所有预警规则和预报判断流程
2. **预警等级阈值规则报告.docx** - Word格式报告（需运行生成脚本）
3. **tree_rules_warning_0.txt** - 0级（不发生）预警规则
4. **tree_rules_warning_1.txt** - 1级（轻度）预警规则
5. **tree_rules_warning_2.txt** - 2级（中度）预警规则
6. **tree_rules_warning_3.txt** - 3级（重度）预警规则
7. **特征重要性.csv** - 各气象因子的重要性排序

## 生成Word文档

如果需要生成Word格式的报告文档，请运行：

```bash
python scripts/generate_word_report.py
```

或者指定输入输出路径：

```bash
python scripts/generate_word_report.py --input "analysis_1105/预警等级阈值规则报告.md" --output "analysis_1105/预警等级阈值规则报告.docx"
```

**注意**：生成Word文档需要安装 `python-docx` 库：

```bash
pip install python-docx
```

## 参数说明

- `--sample`: 样本数据Excel文件路径（默认：样本数据.xlsx）
- `--out`: 输出目录（默认：analysis_1105）
- `--max_depth`: 决策树最大深度（默认：4）
- `--min_samples_leaf`: 叶子节点最小样本数（默认：2）

## 注意事项

1. 确保`样本数据.xlsx`文件中包含：
   - 预警等级列（包含"预警"关键词）
   - 气象因子列（数值类型）
   - 日期列（可选）

2. 脚本会自动识别预警等级列和气象因子列

3. 预警等级会自动映射为：
   - 0级：包含"0"或"不发生"
   - 1级：包含"1"或"轻度"
   - 2级：包含"2"或"中度"
   - 3级：包含"3"或"重度"
