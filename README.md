# Battery RUL V1 System

本仓库当前已落地 **V1 多工况锂电池 RUL 深度学习系统**，采用“主模型 + 基线模型 + 统一评测协议”的工程化结构。

## 1. 系统总体（Overall）

- 数据层：从 `data/processed` 读取放电数据，提取可学习特征。
- 协议层：统一支持 `same_condition`、`LOBO`、`LOCO` 三类实验协议。
- 模型层：
  - 主模型：`ConditionAwareTransformer`
  - 基线：`BaselineLSTM`
- 训练评估层：统一训练/推理/指标（`RMSE`、`MAE`、`MAPE`）。
- 结果层：Notebook 直接展示预测曲线与协议对照结果。

## 2. 系统部分（Modules）

核心代码位于 `src/rul_system/`：

- `data.py`
  - `load_battery_raw_parameters`
  - `default_condition_map`
  - `build_protocol_splits`
  - `prepare_condition_aware_dataloaders`
- `models.py`
  - `ConditionAwareTransformer`
  - `BaselineLSTM`
- `train.py`
  - `train_model`
  - `predict`
  - `compute_regression_metrics`
  - `run_protocol_experiment`

## 3. 快速开始

1. 打开并运行 notebook：`notebooks/04_system_blueprint.ipynb`
2. 按顺序执行单元：
   - 单元 2：导入依赖与模块
   - 单元 4：加载数据并定义 train/test
   - 单元 6：构建 DataLoader
   - 单元 8：训练 Transformer
   - 单元 10：测试评估与可视化
   - 单元 14：V1 协议实验（Transformer vs LSTM）

## 4. V1 协议说明

- `same_condition`：同一工况组中留一块电池做测试。
- `LOBO`（Leave-One-Battery-Out）：每次留一块电池测试，其他全部训练。
- `LOCO`（Leave-One-Condition-Out）：留一整个工况组做测试，其他工况训练。

## 5. 当前默认约定

- 训练集拟合归一化器，测试集仅 `transform`，避免数据泄露。
- 默认窗口长度 `seq_length=10`。
- 当前 notebook 中 V1 协议实验先做“冒烟验证”（每协议先跑 1 个 split）。

## 6. 下一步建议（V2）

- 将 V1 协议实验从“1 个 split”扩展到“全量 split”。
- 增加异常样本策略（低温异常循环过滤或鲁棒损失）。
- 增加结果落盘（CSV + 图像）用于论文附录复现。