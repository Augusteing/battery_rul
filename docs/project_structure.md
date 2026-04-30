# 项目文件夹结构说明

本项目按“论文复现型科研仓库”组织，目标是保证 NASA 数据集上的每一步处理、训练、评估和论文插图都可以被追踪和复现。

## 核心原则

1. 原始数据只读，不直接修改。
2. 中间数据、特征数据、划分文件由脚本生成。
3. 模型源码与训练权重分离。
4. 每个实验由配置文件驱动。
5. 每次关键实验都记录到实验日志。

## 顶层结构

```text
configs/       实验配置，包括数据、模型和完整实验方案
data/          NASA 原始数据、中间数据、处理后数据和数据划分
docs/          复现计划、实验日志、毕业论文材料
experiments/   命名实验入口和阶段性实验说明
models/        训练得到的模型权重和导出文件
notebooks/     探索性分析，不作为最终复现入口
references/    论文阅读笔记、公式核对和 BibTeX
results/       图表、指标表格和日志
scripts/       可命令行运行的数据处理、训练、评估脚本
src/           可复用 Python 源码包
utils/         历史工具代码，后续逐步迁移到 src/battery_rul/utils
```

## 源码分层

```text
src/battery_rul/data/        NASA .mat 读取、循环提取、特征构造
src/battery_rul/models/      LSTM、Transformer、PI-TNet 等模型定义
src/battery_rul/physics/     Verhulst 退化模型、物理残差、单调性约束
src/battery_rul/training/    损失函数、训练循环、早停和检查点
src/battery_rul/evaluation/  SOH/RUL 指标、预测结果汇总
src/battery_rul/utils/       随机种子、路径、日志、配置读取等通用函数
```

## 推荐复现路线

1. `scripts/audit_nasa_data.py`：核验 NASA 原始 `.mat` 文件和循环数量。
2. `scripts/build_nasa_dataset.py`：生成容量、SOH 和固定长度曲线特征。
3. `scripts/train.py`：按 YAML 配置训练基线模型和 PI-TNet。
4. `scripts/evaluate.py`：导出指标表和论文图。

当前只完成结构初始化。下一步应进入数据审计阶段。
