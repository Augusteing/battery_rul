"""PI-TNet model components for NASA SOH reproduction.

The paper specifies a Convolutional Data Processor (CDC + HDC + VDC followed
by vanilla convolution), a ViT/Transformer processor, and a SOH regression
output. Exact source code and tensor dimensions are not disclosed, so this file
implements the closest reproducible architecture consistent with the text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PiTNetConfig:
    """Architecture parameters for PI-TNet."""

    in_channels: int = 1
    input_features: int = 4
    sequence_length: int = 128
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    nominal_capacity_ah: float = 2.0


class CentralDifferenceConv2d(nn.Module):
    """Central Difference Convolution approximation.

    CDC enhances local transient variations by subtracting a center-difference
    response from a standard convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, theta: float = 0.7) -> None:
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        standard = self.conv(x)
        if self.theta == 0:
            return standard
        kernel_sum = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        center_response = nn.functional.conv2d(x, kernel_sum, bias=None)
        return standard - self.theta * center_response


class DirectionalDifferenceConv2d(nn.Module):
    """Directional difference convolution for horizontal or vertical gradients."""

    def __init__(self, in_channels: int, out_channels: int, direction: str) -> None:
        super().__init__()
        if direction not in {"horizontal", "vertical"}:
            raise ValueError("direction must be 'horizontal' or 'vertical'")
        self.direction = direction
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _difference_weight(self) -> torch.Tensor:
        if self.direction == "horizontal":
            return self.weight - torch.flip(self.weight, dims=(3,))
        return self.weight - torch.flip(self.weight, dims=(2,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x,
            self._difference_weight(),
            bias=self.bias,
            padding=1,
        )


class ConvolutionalDataProcessor(nn.Module):
    """Hybrid CDP: CDC + HDC + VDC, then vanilla convolution and fusion."""

    def __init__(self, in_channels: int = 1, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.cdc = CentralDifferenceConv2d(in_channels, d_model)
        self.hdc = DirectionalDifferenceConv2d(in_channels, d_model, "horizontal")
        self.vdc = DirectionalDifferenceConv2d(in_channels, d_model, "vertical")
        self.vanilla = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.low_dim = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_dim = self.cdc(x) + self.hdc(x) + self.vdc(x)
        high_dim = self.activation(self.vanilla(high_dim))
        low_dim = self.low_dim(x)
        omega = torch.sigmoid(self.fusion_logit)
        fused = omega * high_dim + (1.0 - omega) * low_dim
        return self.dropout(self.norm(self.activation(fused)))


class PositionalEncoding(nn.Module):
    """Learnable temporal positional embedding for Transformer tokens."""

    def __init__(self, sequence_length: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.size(1)]


class PiTNet(nn.Module):
    """Physics-informed Transformer backbone without the physics loss term."""

    def __init__(self, config: PiTNetConfig) -> None:
        super().__init__()
        self.config = config
        self.cdp = ConvolutionalDataProcessor(
            in_channels=config.in_channels,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.position = PositionalEncoding(config.sequence_length, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, 4, time), got {tuple(x.shape)}")
        if x.size(1) != self.config.input_features:
            raise ValueError(
                f"Expected {self.config.input_features} input features, got {x.size(1)}"
            )
        if x.size(2) != self.config.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.config.sequence_length}, got {x.size(2)}"
            )

        # Treat feature x time as a small physical signal image for CDP.
        image = x.unsqueeze(1)
        features = self.cdp(image)
        tokens = features.mean(dim=2).transpose(1, 2)
        tokens = self.position(tokens)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        soh = self.head(pooled)
        capacity_ah = soh * self.config.nominal_capacity_ah
        return {
            "soh": soh,
            "capacity_ah": capacity_ah,
            "embedding": pooled,
        }


def config_from_mapping(config: dict[str, Any]) -> PiTNetConfig:
    """Create `PiTNetConfig` from the project YAML structure."""

    input_config = config.get("input", {})
    transformer_config = config.get("transformer", {})
    return PiTNetConfig(
        input_features=len(input_config.get("features", [None] * 4)),
        sequence_length=int(input_config.get("sequence_length", 128)),
        d_model=int(transformer_config.get("d_model", 64)),
        nhead=int(transformer_config.get("nhead", 4)),
        num_layers=int(transformer_config.get("num_layers", 2)),
        dim_feedforward=int(transformer_config.get("dim_feedforward", 128)),
        dropout=float(transformer_config.get("dropout", 0.1)),
    )


def build_pi_tnet_from_yaml(path: Path) -> PiTNet:
    """Build PI-TNet from a YAML config file."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return PiTNet(config_from_mapping(config))
