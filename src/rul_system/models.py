import torch
import torch.nn as nn


class ConditionAwareTransformer(nn.Module):
    """
    工况感知 Transformer 回归模型（序列到单点）。

    该模型用于多工况电池 RUL/容量预测场景：
    - 输入为按时间窗口组织的多维特征序列
    - 输出为下一时刻容量（标量）
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
                """
                初始化模型结构。

                输入参数:
                - input_dim: int
                    每个时间步的输入特征维度。
                - d_model: int
                    Transformer 的隐藏维度。
                - nhead: int
                    多头注意力头数。
                - num_layers: int
                    TransformerEncoder 堆叠层数。
                - dim_feedforward: int
                    前馈网络隐层维度。
                - dropout: float
                    Dropout 比例。
                - max_seq_len: int
                    位置编码参数允许的最大序列长度。
                """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                前向传播。

                输入:
                - x: torch.Tensor, shape = (batch_size, seq_len, input_dim)

                返回:
                - y_hat: torch.Tensor, shape = (batch_size, 1)
                    预测的下一时刻容量。
                """
        seq_len = x.size(1)
        x = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)
