# 2026 专硕机器学习课程考核：家庭电力消耗预测

姓名：朱旭东  
学号：20255227021  
完成方式：单人完成

## 项目内容

本项目完成课程考核 PDF 中要求的家庭电力消耗多变量时间序列预测任务。任务以过去 90 天的电力消耗曲线预测未来 90 天和 365 天的 `global_active_power` 变化曲线，并分别比较三类模型：

1. LSTM
2. Transformer
3. 自提出改进模型 ConvTransformer，即用一维卷积提取局部时间模式后接 Transformer 编码器

每个模型、每个预测长度均运行 5 个随机种子，报告 MSE、MAE 的均值和标准差，并输出预测曲线与 Ground Truth 对比图。

## 数据说明

脚本优先读取 `data/train.csv` 与 `data/test.csv` 或 `data/tes.csv`。如果未提供这些文件，将自动下载 UCI Machine Learning Repository 的 Individual household electric power consumption 数据集，并按天汇总：

- `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 按天求和
- `voltage`、`global_intensity` 按天求平均
- 自动计算 `sub_metering_remainder`
- 添加月份、星期、年内日序等日历变量

天气字段在报告中作为可扩展输入说明。若老师提供带天气字段的 `train.csv/test.csv`，脚本会自动把数值列纳入建模特征。

## 运行方式

```powershell
cd D:\WorkShop\code\ml_power_forecast_final
python run_experiment.py
python make_report.py
```

主要输出位于 `output/`：

- `results.csv`：每轮实验指标
- `summary.csv`：均值和标准差
- `figures/`：预测曲线、数据概览图
- `20255227021_朱旭东_机器学习课程考核报告.docx`：最终报告
- `20255227021_朱旭东_机器学习课程考核提交包.zip`：提交包

## 代码链接

https://github.com/XudongZhu-666/ml_power_forecast_final
