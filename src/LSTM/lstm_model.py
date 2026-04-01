import torch
import torch.nn as nn

class BatteryRULLSTM(nn.Module):
    """
    基于长短期记忆网络 (LSTM) 的锂离子电池剩余使用寿命 (RUL) 预测模型。
    
    架构设计参考了该领域主流文献（如基于时序容量和健康因子退化的数据驱动方法）：
    - 面对容量恢复效应 (Capacity Regeneration) 和长期退化趋势，使用堆叠的 LSTM 层（Stacked LSTM）
      来捕捉输入特征（如前20次循环的HIs或电压/温度特征）中的长程依赖关系。
    - 使用了 Dropout 缓解因为 NASA 电池数据集样本较少而导致的过拟合问题。
    - 使用全连接层 (Fully Connected Layers) 将 LSTM 的末态隐藏特征映射为对当前步（下一阶段）SOH 或容量的回归预测。
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        初始化 LSTM 预测模型。
        
        参数:
        -----------
        input_size : int
            输入特征的维度数量。如果是纯4个健康因子 (HIs)，则为 4；
            如果在后续加入环境温度、放电电流等工况环境特征，则会相应增加（如 6）。
        hidden_size : int
            LSTM 内部隐藏层的维度（神经元个数）。默认 64 可以在捕捉非线性特征和轻量化计算之间取得平衡。
        num_layers : int
            堆叠的 LSTM 层数。默认为 2，能够增加网络深度，提取更高维的退化模式。
        dropout : float
            正则化随机失活率（Dropout Ratio），用于在全连接和多层 LSTM 之间丢弃部分神经元，防止过拟合。
        """
        super(BatteryRULLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. 核心时序特征提取层: LSTM
        # batch_first=True 保证输入张量形状为 (Batch_Size, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0 # 单层 LSTM 不应用隐层间 dropout
        )
        
        # 2. 回归预测端 (Regressive Head): 用多层感知机(MLP)解析最后一个时刻的特征
        # 中间的降维过渡层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # 输出层：降维至 1 个标量，即回归任务的目标 (SOH / Capacity)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播过程。
        
        参数:
        -----------
        x : torch.Tensor
            形状为 (batch_size, seq_len, input_size) 的特征序列张量。
            例如：一批次包含 64 个数据，每个数据包含最近 20 个循环的 4 个退化特征组合。
            
        输出:
        -----------
        predictions : torch.Tensor
            形状为 (batch_size, 1) 的回归预测值张量。
        """
        # 初始化隐藏状态 h_0 和细胞状态 c_0（可选，默认为全零张量）
        # 这里交由 PyTorch 自动默认初始化为 0 以利用硬件加速
        
        # LSTM 前向传播
        # lstm_out shape: (batch_size, seq_len, hidden_size) 记录了所有时间步的隐藏状态
        # h_n shape: (num_layers, batch_size, hidden_size)   记录了最后的隐藏状态
        # c_n shape: (num_layers, batch_size, hidden_size)   记录了最后的细胞状态
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 在 RUL 退化预测中，我们通常只需要最后一个时间步 (也就是时刻 t) 累积融合了此前所有记忆的输出状态
        # 来预测下一时刻 (t+1) 或当前时刻的电池健康度。
        last_step_output = lstm_out[:, -1, :] # shape: (batch_size, hidden_size)
        
        # 通过全连接网络解码预测值
        out = self.fc1(last_step_output)
        out = self.relu(out)
        out = self.dropout_layer(out)
        predictions = self.fc2(out)
        
        return predictions
