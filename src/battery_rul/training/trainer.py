"""Training loop for data-only PI-TNet reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from battery_rul.evaluation import regression_metrics
from battery_rul.physics import PhysicsInformedObjective


@dataclass(frozen=True)
class ChannelStandardizer:
    """Per-channel standardization fitted on the training split."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, tensors: torch.Tensor) -> "ChannelStandardizer":
        mean = tensors.mean(dim=(0, 2), keepdim=True)
        std = tensors.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def to(self, device: torch.device) -> "ChannelStandardizer":
        return ChannelStandardizer(mean=self.mean.to(device), std=self.std.to(device))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


@dataclass
class TrainResult:
    history: pd.DataFrame
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    best_epoch: int
    best_train_predictions: pd.DataFrame
    best_test_predictions: pd.DataFrame
    checkpoint: dict
    best_checkpoint: dict


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["x"].to(device), batch["capacity_ah"].to(device)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    standardizer: ChannelStandardizer | None = None,
    physics_objective: PhysicsInformedObjective | None = None,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = defaultdict(float)
    total_samples = 0
    for batch in loader:
        x, target_capacity = _batch_to_device(batch, device)
        if standardizer is not None:
            x = standardizer.transform(x)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        if physics_objective is None:
            data_loss = nn.functional.mse_loss(output["capacity_ah"], target_capacity)
            loss_terms = {
                "total_loss": data_loss,
                "data_loss": data_loss,
            }
        else:
            loss_terms = physics_objective(output=output, batch=batch, device=device)
        loss = loss_terms["total_loss"]
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        for name, value in loss_terms.items():
            metric_sums[name] += float(value.detach().item()) * batch_size
        total_samples += batch_size
    metrics = {name: value / total_samples for name, value in metric_sums.items()}
    if physics_objective is not None:
        metrics.update(physics_objective.diagnostics())
    return metrics


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    standardizer: ChannelStandardizer | None = None,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        x, _ = _batch_to_device(batch, device)
        if standardizer is not None:
            x = standardizer.transform(x)
        output = model(x)
        pred_capacity = output["capacity_ah"].detach().cpu().numpy().reshape(-1)
        pred_soh = output["soh"].detach().cpu().numpy().reshape(-1)
        true_capacity = batch["capacity_ah"].cpu().numpy().reshape(-1)
        true_soh = batch["soh"].cpu().numpy().reshape(-1)
        for i in range(len(pred_capacity)):
            rows.append(
                {
                    "battery_id": batch["battery_id"][i],
                    "cycle_index": int(batch["cycle_index"][i]),
                    "discharge_index": int(batch["discharge_index"][i]),
                    "capacity_true": float(true_capacity[i]),
                    "capacity_pred": float(pred_capacity[i]),
                    "soh_true": float(true_soh[i]),
                    "soh_pred": float(pred_soh[i]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame, prefix: str) -> dict[str, float]:
    capacity = regression_metrics(
        predictions["capacity_true"].to_numpy(),
        predictions["capacity_pred"].to_numpy(),
    )
    soh = regression_metrics(
        predictions["soh_true"].to_numpy(),
        predictions["soh_pred"].to_numpy(),
    )
    metrics = {}
    for name, value in capacity.items():
        metrics[f"{prefix}_capacity_{name}"] = value
    for name, value in soh.items():
        metrics[f"{prefix}_soh_{name}"] = value
    return metrics


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    standardize: bool = True,
    physics_objective: PhysicsInformedObjective | None = None,
) -> TrainResult:
    standardizer = None
    if standardize:
        standardizer = ChannelStandardizer.fit(train_loader.dataset.x).to(device)

    model.to(device)
    if physics_objective is not None:
        physics_objective.to(device)
    history_rows = []
    best_epoch = 0
    best_metric = float("inf")
    best_train_predictions = None
    best_test_predictions = None
    best_model_state_dict = None
    best_physics_state_dict = None
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            standardizer=standardizer,
            physics_objective=physics_objective,
        )
        train_predictions = predict(model, train_loader, device, standardizer)
        test_predictions = predict(model, test_loader, device, standardizer)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.pop("total_loss"),
        }
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update(evaluate_predictions(train_predictions, "train"))
        row.update(evaluate_predictions(test_predictions, "test"))
        history_rows.append(row)
        current_metric = float(row["test_soh_rmse"])
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_train_predictions = train_predictions.copy()
            best_test_predictions = test_predictions.copy()
            best_model_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if physics_objective is not None:
                best_physics_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in physics_objective.state_dict().items()
                }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "max_epochs": max_epochs,
        "standardize": standardize,
        "physics_objective_state_dict": (
            None if physics_objective is None else physics_objective.state_dict()
        ),
        "standardizer": {
            "mean": None if standardizer is None else standardizer.mean.detach().cpu(),
            "std": None if standardizer is None else standardizer.std.detach().cpu(),
        },
    }
    best_checkpoint = {
        "model_state_dict": best_model_state_dict,
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "standardize": standardize,
        "physics_objective_state_dict": best_physics_state_dict,
        "standardizer": {
            "mean": None if standardizer is None else standardizer.mean.detach().cpu(),
            "std": None if standardizer is None else standardizer.std.detach().cpu(),
        },
    }
    return TrainResult(
        history=pd.DataFrame(history_rows),
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        best_epoch=best_epoch,
        best_train_predictions=best_train_predictions,
        best_test_predictions=best_test_predictions,
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
    )


def save_train_result(result: TrainResult, output_dir: Path, run_name: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    checkpoint_dir = Path("models/checkpoints")
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    history_path = logs_dir / f"{run_name}_history.csv"
    train_pred_path = tables_dir / f"{run_name}_train_predictions.csv"
    test_pred_path = tables_dir / f"{run_name}_test_predictions.csv"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    best_train_pred_path = tables_dir / f"{run_name}_best_train_predictions.csv"
    best_test_pred_path = tables_dir / f"{run_name}_best_test_predictions.csv"
    best_checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"

    result.history.to_csv(history_path, index=False)
    result.train_predictions.to_csv(train_pred_path, index=False)
    result.test_predictions.to_csv(test_pred_path, index=False)
    result.best_train_predictions.to_csv(best_train_pred_path, index=False)
    result.best_test_predictions.to_csv(best_test_pred_path, index=False)
    torch.save(result.checkpoint, checkpoint_path)
    torch.save(result.best_checkpoint, best_checkpoint_path)
    return {
        "history": history_path,
        "train_predictions": train_pred_path,
        "test_predictions": test_pred_path,
        "checkpoint": checkpoint_path,
        "best_train_predictions": best_train_pred_path,
        "best_test_predictions": best_test_pred_path,
        "best_checkpoint": best_checkpoint_path,
    }
