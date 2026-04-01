import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict

class ModelTrainer:
    """
    通用模型训练器。
    用于将模型的训练循环（包含前向传播、损失计算、反向传播、验证集评估）从 Notebook 中剥离，
    使得代码更加规范，并且支持未来无缝切换 LSTM、GRU 和 Transformer 等不同模型。
    """
    def __init__(self, model: nn.Module, model_type: str = 'LSTM', device: str = None, 
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        """
        初始化训练器。
        
        参数:
        -----------
        model : nn.Module
            需要训练的 PyTorch 模型。
        model_type : str
            模型标识（如 'LSTM'），用于日志打印。
        device : str
            运行设备 ('cuda' 或 'cpu')。如果未指定，将自动检测。
        lr : float
            Adam 优化器的学习率。在电池寿命预测中，1e-3 到 5e-4 是最佳区间。
        weight_decay : float
            L2 正则化惩罚项。因为 NASA 数据量小，加入 weight_decay 能极大程度抑制过拟合。
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model_type = model_type
        
        # 【核心推荐 1：损失函数】 
        # RUL 预测属于标准的时序回归任务。MSE (均方误差) 依然是最成熟且梯度稳定的选择。
        # 它可以对预测偏差较大的“异常高点/低点”给予严厉的二次方惩罚，这对 RUL 任务防止提前崩盘非常重要。
        self.criterion = nn.MSELoss()
        
        # 【核心推荐 2：优化器】 
        # Adam (或 AdamW) 绝对足够，甚至可以说是所有时序退化网络的最优解。
        # 这里使用了目前更先进的 AdamW (能更好地处理 L2 正则化以防过拟合)。
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        执行单个 Epoch 的训练循环。
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_dict() if hasattr(self.optimizer, 'zero_dict') else self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(batch_x)
            
            # 计算损失
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播 & 权重更新
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping)，防止 RNN/LSTM 特有的梯度爆炸问题
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        return epoch_loss / len(train_loader.dataset)

    def evaluate(self, test_loader: DataLoader) -> float:
        """
        在验证集/测试集上评估模型，并不计算和更新梯度。
        返回的是验证集的平均 MSE。可以用于外层的 Early Stopping 判断。
        """
        self.model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                
        return test_loss / len(test_loader.dataset)
        
    def fit(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 100, patience: int = 15, verbose: bool = True) -> Dict:
        """
        执行完整的训练流程，并带有一种称为 Early Stopping 的防过拟合机制。
        记录并返回每一个 Epoch 的损失追踪，用于在 Notebook 里画出平滑的收敛折线图。
        """
        print(f"🚀 开始在 {self.device} 上训练 {self.model_type} 模型...")
        
        history = {'train_loss': [], 'test_loss': []}
        best_test_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate(test_loader)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            
            # 触发 Early Stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"📊 Epoch [{epoch}/{epochs}] | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} | 停止容忍度: {patience_counter}/{patience}")
                
            # 当测试集表现连续 N 个 Epoch 没有创新低，说明开始过拟合了，此时及时掐断
            if patience_counter >= patience:
                print(f"⚠️ 在 Epoch {epoch} 触发 Early Stopping (早停)，停止训练防过拟合。")
                break
                
        # 恢复历史上最好时刻（而非最终过拟合时刻）的权重
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✨ 训练结束。已回滚至最佳验证权重，最优测试集 MSE 为: {best_test_loss:.6f}")
            
        return history
