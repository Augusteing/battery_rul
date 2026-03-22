import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class RULExperiment:
    """高层调度类：接管训练、测试、反归一化与可视化生命周期"""
    def __init__(self, model, data_module, lr=0.001):
        self.model = model
        self.data_module = data_module # 注入数据引擎
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_loader, epochs=100):
        """一键训练接口"""
        print(f"开始训练模型 (Device: {self.device})...")
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}")

    def test_and_plot(self, test_loader, actual_capacity, title_suffix=""):
        """一键测试与绘图接口"""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().item())
                
        # 自动调用 data_module 里的 scaler 进行反归一化
        scaler_y = self.data_module.scaler_y
        preds_real = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # 自动出图
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(actual_capacity)), actual_capacity, label='Ground Truth', color='blue')
        predict_x_axis = range(self.data_module.seq_length, len(actual_capacity))
        plt.plot(predict_x_axis, preds_real, label='LSTM Prediction', color='red', linestyle='--')
        plt.axhline(y=1.4, color='black', linestyle=':', label='EOL (1.4 Ah)')
        plt.title(f'RUL Prediction {title_suffix}')
        plt.xlabel('Cycle Index')
        plt.ylabel('Capacity (Ah)')
        plt.legend()
        plt.grid(True)
        plt.show()