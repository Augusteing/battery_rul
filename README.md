# 锂离子电池剩余使用寿命 (RUL) 预测项目 (NASA 数据集)

本项目致力于通过深度学习技术对NASA锂离子电池老化数据集进行剩余使用寿命 (Remaining Useful Life, RUL) 的预测。本项目作为毕业设计的核心代码库，包含了完整的数据探索、特征工程、基准模型构建以及跨工况泛化能力评估。

## 最新进展 (Phase 1: LSTM 基准验证)

目前项目已完成对 LSTM 基础模型的充分验证，主要包括以下工作：
1. **数据清洗与特征构建 (`utils/dataset_builder.py`)**：
   - 提取并平滑了健康因子 (Health Indicators, HI)：恒流充电时间 (`HI_Duration`)、平滑后的最高温度 (`HI_T_peak_Smoothed`)、放电电压压降 (`HI_V_drop`) 以及放电能量 (`HI_Energy`)。
   - 引入自回归机制处理时序特征，支持灵活划定滑动窗口 (如 20 步) 以及目标 SOH 归一化。
2. **多工况独立组内评估 (`notebooks/14_lstm.ipynb`)**：
   - **G1 组 (室温 24°C 标准工况)**：完成留一交叉验证 (LOOCV)，揭示了在单一分布下 LSTM 仍存在的 Domain Shift (数值平移偏置) 现象。
   - **G3 组 (高温高倍率工况)**：完成交叉验证，暴露了对局部放电突变追踪不够灵敏、存在反馈延迟的问题。
   - **G9 组 (低温 4°C 剧烈波动工况)**：验证发现基础 LSTM 模型极易对异构特征产生“过度平滑”处理，丧失对高频动态特征的敏感度。
3. **全局混合工况泛化盲测**：
   - 将所有已知工况电池建立超级混合训练集，分别对未见过的室温 (B0018)、高温 (B0029) 与低温 (B0053) 电池进行极端跨组盲测。
   - **结论**：验证和证明了在多异构分布以及高时序噪声的干扰下，基础 LSTM 网络因缺乏对特定时段/特定特征的注意力动态分配机制，导致严重滞后。本结论为后续引入具备自注意力机制 (Self-Attention) 的 Transformer 模型提供了强有力的实证对比支撑。

## 项目结构
```text
battery-rul/
├── configs/          # 模型和实验配置文件
├── data/             # 数据集目录 (raw, processed, features)
├── docs/             # 论文撰写与流程图等相关文档
├── notebooks/        # Jupyter实验记录 (EDA, Feature Selection, LSTM等)
├── record/           # 历史实验留档存档
├── results/          # 预测输出和图表保存路径
├── src/              # 模型定义 (LSTM, 待实现的Transformer)
└── utils/            # 数据装载与预处理模块
```

## 技术栈与保障
- **环境**: PyTorch, Pandas, Scikit-learn, Matplotlib
- **实验严谨性**: 已实现基于全量堆栈的强制全局种子绑定 (`torch.manual_seed(42)` 等)，彻底杜绝随机性干扰，保证论文复现率。

## 下一步计划
通过本次验证得出的结论，项目即将进入 Transformer (Self-Attention) 阶段，以期彻底解决 LSTM 在多工况复杂序列中的滞后与过度平滑问题。
