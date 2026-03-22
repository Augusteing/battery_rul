import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class BatteryDataModule:
    """封装的数据提取：管理特征提取、归一化状态与张量转换"""
    def __init__(self, seq_length=10, batch_size=16):
        self.seq_length = seq_length
        self.batch_size = batch_size
        try:
            scaler_cls = MinMaxScaler
        except NameError:
            from sklearn.preprocessing import MinMaxScaler as scaler_cls
        # Scaler 被封装在类内部，作为对象的状态 (State) 永久保存，避免丢失
        self.scaler_x = scaler_cls()
        self.scaler_y = scaler_cls()
        self.is_fitted = False # 防止未经训练集拟合就去 transform 测试集

    def _extract_features(self, df_raw):
        """内部方法：从原始 DataFrame 提取标量特征"""
        features = []
        for _, row in df_raw.iterrows():
            try:
                time_arr, temp_arr, cap = row['Time'], row['Temperature_measured'], row['Capacity']
                if cap > 0 and len(time_arr) > 0:
                    features.append([time_arr[-1], np.max(temp_arr), cap])
            except Exception:
                continue
        return np.array(features)

    def _create_sliding_window(self, x_scaled, y_scaled):
        """内部方法：构建 3D 张量"""
        x_seq, y_seq = [], []
        for i in range(len(x_scaled) - self.seq_length):
            x_seq.append(x_scaled[i : i + self.seq_length])
            y_seq.append(y_scaled[i + self.seq_length])
        return torch.tensor(np.array(x_seq), dtype=torch.float32), \
               torch.tensor(np.array(y_seq), dtype=torch.float32)

    def prepare_train_data(self, train_dict):
        """处理训练集：提取 -> 拟合(Fit)并缩放 -> 构建滑动窗口 -> 返回 DataLoader"""
        matrices = [self._extract_features(df) for df in train_dict.values()]
        concat_matrix = np.vstack(matrices)
        
        # 防泄露：只有在这里调用 fit
        x_scaled = self.scaler_x.fit_transform(concat_matrix[:, :-1])
        y_scaled = self.scaler_y.fit_transform(concat_matrix[:, -1].reshape(-1, 1))
        self.is_fitted = True
        
        # 逐电池构建窗口，避免跨电池缝合
        x_tensors, y_tensors = [], []
        for matrix in matrices:
            xs = self.scaler_x.transform(matrix[:, :-1])
            ys = self.scaler_y.transform(matrix[:, -1].reshape(-1, 1))
            x_tensor, y_tensor = self._create_sliding_window(xs, ys)
            x_tensors.append(x_tensor)
            y_tensors.append(y_tensor)
            
        dataset = TensorDataset(torch.cat(x_tensors), torch.cat(y_tensors))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def prepare_test_data(self, test_df):
        """处理测试集：提取 -> 仅应用(Transform)缩放 -> 返回 DataLoader 与真实容量"""
        if not self.is_fitted:
            raise ValueError("DataModule 尚未在训练集上进行 fit，不能处理测试集！")
            
        matrix = self._extract_features(test_df)
        x_scaled = self.scaler_x.transform(matrix[:, :-1])
        y_scaled = self.scaler_y.transform(matrix[:, -1].reshape(-1, 1))
        
        x_tensor, y_tensor = self._create_sliding_window(x_scaled, y_scaled)
        dataset = TensorDataset(x_tensor, y_tensor)
        
        return DataLoader(dataset, batch_size=1, shuffle=False), matrix[:, -1]