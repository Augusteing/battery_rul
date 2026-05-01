"""Verify PI-TNet model forward and backward passes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery_rul.data.dataloaders import create_nasa_dataloaders
from battery_rul.models import build_pi_tnet_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model/pi_tnet.yaml"))
    parser.add_argument("--battery-id", type=str, default="B0005")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = create_nasa_dataloaders(
        battery_id=args.battery_id,
        batch_size=args.batch_size,
    )
    batch = next(iter(bundle.train))
    x = batch["x"].to(device)
    y = batch["soh"].to(device)
    model = build_pi_tnet_from_yaml(args.config).to(device)
    model.train()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output["soh"], y)
    loss.backward()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_params = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"device: {device}")
    print(f"input: {tuple(x.shape)}")
    print(f"soh output: {tuple(output['soh'].shape)}")
    print(f"capacity output: {tuple(output['capacity_ah'].shape)}")
    print(f"embedding: {tuple(output['embedding'].shape)}")
    print(f"loss: {loss.item():.6f}")
    print(f"trainable parameters: {trainable_params}")
    print(f"parameters with gradients: {grad_params}")


if __name__ == "__main__":
    main()
