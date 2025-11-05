# 数据文件迁移到metadata文件夹说明

## 需要移动的文件

请将以下根目录下的数据文件移动到 `metadata` 文件夹：

1. `影响因子1103_final.csv`
2. `影响因子1103_final.xlsx`
3. `样本数据.xlsx`
4. `张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20.xlsx`
5. `张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx`

## 已更新的脚本文件

以下脚本文件中的默认路径已更新为 `metadata/文件名`：

### ✅ 已更新
- `scripts/analyze_thresholds.py`
  - `--factors` 默认值：`metadata/影响因子1103_final.csv`
  - `--warning` 默认值：`metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx`

- `scripts/analyze_thresholds_1105.py`
  - `--sample` 默认值：`metadata/样本数据.xlsx`

- `scripts/predict_warning.py`
  - `--factors` 默认值：`metadata/影响因子1103_final.csv`
  - `--warning` 默认值：`metadata/张桂香19-25校内大果园花果带菌率数据分析--给张工分析数据-10.20_预警.xlsx`

### ⚠️ 需要检查的文件

以下文件可能需要手动检查并更新：

- `scripts/analyze_thresholds_1104.py` - 检查 `factors_xlsx` 和 `warning_xlsx` 参数
- `scripts/analyze_thresholds_batch.py` - 检查传递给 `run_analysis` 的文件路径参数
- `scripts/merge_data_with_weather.py` - 如果存在，检查 `--infection` 和 `--weather` 参数

## 操作步骤

1. 在项目根目录创建 `metadata` 文件夹
2. 将上述5个数据文件移动到 `metadata` 文件夹
3. 验证所有脚本是否能正常找到文件（通过运行测试脚本或实际使用）

## 注意事项

- 所有脚本仍然支持通过命令行参数指定自定义路径
- 如果文件已存在于 `metadata` 文件夹，脚本会优先使用该路径
- 确保在运行脚本时，工作目录是项目根目录
