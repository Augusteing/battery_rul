import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional

class BatterySequenceDataset(Dataset):
    def __init__(self, data_list: List[pd.DataFrame], seq_len: int, feature_cols: List[str], target_col: str):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.x_data = []
        self.y_data = []
        for df in data_list:
            if len(df) <= self.seq_len:
                continue
            features = df[self.feature_cols].values
            targets = df[self.target_col].values
            for i in range(len(df) - self.seq_len):
                self.x_data.append(features[i : i + self.seq_len])
                self.y_data.append(targets[i + self.seq_len])
        self.x_tensor = torch.tensor(np.array(self.x_data), dtype=torch.float32)
        self.y_tensor = torch.tensor(np.array(self.y_data), dtype=torch.float32).unsqueeze(-1)
    def __len__(self) -> int: return len(self.x_tensor)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: return self.x_tensor[idx], self.y_tensor[idx]

class BatteryDataBuilder:
    def __init__(self, data_dir: str, seq_len: int = 20, feature_cols: Optional[List[str]] = None, target_col: str = 'Capacity'):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.feature_cols = feature_cols if feature_cols else ['HI_Duration', 'HI_T_peak_Smoothed', 'HI_V_drop', 'HI_Energy']
        self.target_col = target_col

    def _load_battery_dfs(self, battery_ids: List[str]) -> List[pd.DataFrame]:
        data_list = []
        for b_id in battery_ids:
            file_path = self.data_dir / f"{b_id}_features.csv"
            if not file_path.exists(): raise FileNotFoundError(f"未找到特征: {file_path}")
            data_list.append(pd.read_csv(file_path))
        return data_list

    def get_dataloaders(self, train_battery_ids: List[str], test_battery_ids: List[str], batch_size: int = 64) -> Tuple[DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:
        train_dfs = self._load_battery_dfs(train_battery_ids)
        test_dfs = self._load_battery_dfs(test_battery_ids)
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        all_train_data = pd.concat(train_dfs)
        scaler.fit(all_train_data[self.feature_cols])
        target_scaler.fit(all_train_data[[self.target_col]])
        processed_train_dfs = []
        for df in train_dfs:
            df_copy = df.copy()
            raw_target = df_copy[[self.target_col]].copy()
            df_copy[self.feature_cols] = scaler.transform(df_copy[self.feature_cols])
            df_copy[self.target_col] = target_scaler.transform(raw_target)
            processed_train_dfs.append(df_copy)
        processed_test_dfs = []
        for df in test_dfs:
            df_copy = df.copy()
            raw_target = df_copy[[self.target_col]].copy()
            df_copy[self.feature_cols] = scaler.transform(df_copy[self.feature_cols])
            df_copy[self.target_col] = target_scaler.transform(raw_target)
            processed_test_dfs.append(df_copy)
        train_dataset = BatterySequenceDataset(processed_train_dfs, self.seq_len, self.feature_cols, self.target_col)
        test_dataset = BatterySequenceDataset(processed_test_dfs, self.seq_len, self.feature_cols, self.target_col)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, test_loader, scaler, target_scaler