import torch
import torch.nn as nn


class ClassicLSTMRegressor(nn.Module):
    """
    标准 LSTM 回归模型（序列到单点）。

    设计目标：
    - 作为可复现、可解释的经典基线模型；
    - 与当前协议评估框架直接对接（输入 shape: B x T x F，输出 shape: B x 1）。
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        direction_factor = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_dim * direction_factor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step)


def default_lstm_kwargs() -> dict:
    """返回标准 LSTM 基线推荐超参数。"""
    return {
        "input_dim": 5,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "bidirectional": False,
    }
