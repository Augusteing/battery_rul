from typing import Dict, List, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import prepare_condition_aware_dataloaders


def train_model(model, train_loader, epochs: int = 50, lr: float = 1e-3, device: str = "cpu"):
    """
    执行模型训练循环并返回损失历史。

    输入参数:
    - model:
        任意可调用的 PyTorch 模型，输入序列输出标量预测。
    - train_loader:
        训练集 DataLoader，迭代返回 (x, y)。
    - epochs: int
        训练轮数。
    - lr: float
        Adam 学习率。
    - device: str
        训练设备，通常为 "cpu" 或 "cuda"。

    返回:
    - history: List[float]
        每一轮的平均 MSE 损失。
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / max(1, len(train_loader))
        history.append(avg_loss)
    return history


def predict(model, test_loader, scaler_y, device: str = "cpu"):
    """
    在测试集上推理并将预测值反归一化到真实容量单位。

    输入参数:
    - model:
        已训练完成的 PyTorch 模型。
    - test_loader:
        测试集 DataLoader，迭代返回 (x, y)。
    - scaler_y:
        标签缩放器（MinMaxScaler），用于 inverse_transform。
    - device: str
        推理设备，通常为 "cpu" 或 "cuda"。

    返回:
    - preds_real: np.ndarray
        反归一化后的预测容量序列（一维）。
    """
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().reshape(-1, 1)
            preds.append(pred)
    preds_scaled = np.vstack(preds)
    preds_real = scaler_y.inverse_transform(preds_scaled).flatten()
    return preds_real


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归任务常用评价指标。

    输入参数:
    - y_true: np.ndarray
        真实容量序列。
    - y_pred: np.ndarray
        预测容量序列。

    返回:
    - metrics: Dict[str, float]
        包含以下指标：
        - RMSE: 均方根误差
        - MAE: 平均绝对误差
        - MAPE: 平均绝对百分比误差（%）
    """
    safe_true = np.clip(y_true, 1e-8, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def run_protocol_experiment(
    battery_dict,
    condition_map,
    protocol_splits: List[Dict[str, List[str]]],
    model_cls: Type,
    model_kwargs: Dict,
    train_kwargs: Dict,
    seq_length: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    在给定协议划分上运行实验并汇总结果。

    设计说明:
    - 每个 split 使用同一模型类型与训练超参数。
    - 每个测试电池单独构建 DataLoader 与缩放器，避免跨测试集污染。

    返回:
    - records: List[Dict]
        每条记录包含 split 信息、测试电池及评估指标。
    """
    records: List[Dict] = []

    for split in protocol_splits:
        train_ids = split["train_ids"]
        for test_id in split["test_ids"]:
            try:
                train_loader, test_loader, scaler_y, test_actual_capacity = prepare_condition_aware_dataloaders(
                    battery_dict=battery_dict,
                    train_ids=train_ids,
                    test_id=test_id,
                    condition_map=condition_map,
                    seq_length=seq_length,
                    batch_size=batch_size,
                )
            except ValueError as error:
                print(f"[Skip] protocol_group={split.get('group', 'mixed')} test_id={test_id} reason={error}")
                continue

            model = model_cls(**model_kwargs)
            _ = train_model(
                model=model,
                train_loader=train_loader,
                device=device,
                **train_kwargs,
            )
            preds = predict(model=model, test_loader=test_loader, scaler_y=scaler_y, device=device)
            y_true = test_actual_capacity[seq_length : seq_length + len(preds)]
            metrics = compute_regression_metrics(y_true, preds)

            records.append(
                {
                    "protocol_group": split.get("group", "mixed"),
                    "train_ids": train_ids,
                    "test_id": test_id,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "num_points": int(len(preds)),
                }
            )

    return records
