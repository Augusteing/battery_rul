import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def load_battery_raw_parameters(data_dir: str, target_batteries: List[str]) -> Dict[str, pd.DataFrame]:
        """
        从预处理后的 pkl 文件中加载指定电池的放电阶段原始参数。

        该函数是数据入口：遍历电池 ID，读取对应文件，筛选放电工况，
        并仅保留下游特征提取所需字段。

        输入参数:
        - data_dir: str
            存放电池 pkl 文件的目录路径。
        - target_batteries: List[str]
            目标电池编号列表，例如 ["B0005", "B0006"]。

        返回:
        - raw_parameters_dict: Dict[str, pd.DataFrame]
            键为电池编号，值为筛选后的放电数据表。
            若文件不存在或数据为空则自动跳过。
        """
        raw_parameters_dict: Dict[str, pd.DataFrame] = {}
        for battery_id in target_batteries:
                file_path = os.path.join(data_dir, f"{battery_id}.pkl")
                if not os.path.exists(file_path):
                        continue

                df_raw = pd.read_pickle(file_path)
                if "Cycle_Type" in df_raw.columns:
                        df_discharge = df_raw[df_raw["Cycle_Type"] == "discharge"].copy()
                else:
                        df_discharge = df_raw.copy()

                columns_to_keep = [
                        "Specific_Index",
                        "Capacity",
                        "Temperature_measured",
                        "Time",
                ]
                available_columns = [col for col in columns_to_keep if col in df_discharge.columns]
                df_selected = df_discharge[available_columns]
                if not df_selected.empty:
                        raw_parameters_dict[battery_id] = df_selected

        return raw_parameters_dict


def default_condition_map() -> Dict[str, Dict[str, float]]:
        """
        构建默认的“电池-工况”映射表。

        该映射用于显式注入工况先验信息，帮助模型区分不同实验条件下的
        时序退化行为，而不是只依赖放电时间/温度等序列统计量。

        返回:
        - condition_map: Dict[str, Dict[str, float]]
            每个电池包含以下工况字段:
            - ambient_temp_c: 环境温度（摄氏度）
            - discharge_current_a: 放电电流（安培）
            - cutoff_voltage_v: 截止电压（伏）
        """
        return {
                "B0005": {"ambient_temp_c": 24.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.7},
                "B0006": {"ambient_temp_c": 24.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.5},
                "B0007": {"ambient_temp_c": 24.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.2},
                "B0018": {"ambient_temp_c": 24.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.5},
                "B0053": {"ambient_temp_c": 4.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.0},
                "B0054": {"ambient_temp_c": 4.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.2},
                "B0055": {"ambient_temp_c": 4.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.5},
                "B0056": {"ambient_temp_c": 4.0, "discharge_current_a": 2.0, "cutoff_voltage_v": 2.7},
        }


def _extract_feature_matrix(
        df_raw: pd.DataFrame,
        battery_id: str,
        condition_map: Dict[str, Dict[str, float]],
) -> np.ndarray:
        """
        将单块电池的原始放电数据转换为标量特征矩阵。

        对每个有效放电循环提取:
        1) discharge_time: 放电结束时刻（Time 最后一个点）
        2) max_temp: 放电过程最高温度
        3) ambient_temp_c / discharge_current_a / cutoff_voltage_v: 工况先验
        4) capacity: 当前循环真实容量（回归目标）

        输入参数:
        - df_raw: pd.DataFrame
            某一块电池的放电数据。
        - battery_id: str
            电池编号，用于检索 condition_map。
        - condition_map: Dict[str, Dict[str, float]]
            工况映射字典。

        返回:
        - matrix: np.ndarray, shape = (N, 6)
            列顺序为:
            [discharge_time, max_temp, ambient_temp_c,
             discharge_current_a, cutoff_voltage_v, capacity]
        """
        cond = condition_map[battery_id]
        rows = []
        for _, row in df_raw.iterrows():
                try:
                        cap = float(row["Capacity"])
                        time_arr = row["Time"]
                        temp_arr = row["Temperature_measured"]
                        if cap <= 0 or len(time_arr) == 0 or len(temp_arr) == 0:
                                continue

                        discharge_time = float(time_arr[-1])
                        max_temp = float(np.max(temp_arr))
                        rows.append(
                                [
                                        discharge_time,
                                        max_temp,
                                        cond["ambient_temp_c"],
                                        cond["discharge_current_a"],
                                        cond["cutoff_voltage_v"],
                                        cap,
                                ]
                        )
                except Exception:
                        continue

        return np.array(rows, dtype=np.float32)


def _to_windows(matrix: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        将按循环排列的矩阵转换为滑动窗口监督学习样本。

        序列到单点预测定义:
        - 输入 X: 连续 seq_length 个时间步的特征
        - 标签 y: 窗口后一个时间步的容量

        输入参数:
        - matrix: np.ndarray
            最后一列为目标容量，其余列为输入特征。
        - seq_length: int
            滑动窗口长度。

        返回:
        - x_seq: np.ndarray, shape = (N-seq_length, seq_length, num_features)
        - y_seq: np.ndarray, shape = (N-seq_length, 1)
        """
        x_seq, y_seq = [], []
        for i in range(len(matrix) - seq_length):
                x_seq.append(matrix[i : i + seq_length, :-1])
                y_seq.append(matrix[i + seq_length, -1])

        return np.array(x_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32).reshape(-1, 1)


def prepare_condition_aware_dataloaders(
        battery_dict: Dict[str, pd.DataFrame],
        train_ids: List[str],
        test_id: str,
        condition_map: Dict[str, Dict[str, float]],
        seq_length: int = 10,
        batch_size: int = 32,
):
        """
        构建“工况感知”训练/测试 DataLoader，并执行防泄露归一化。

        处理流程:
        1) 对 train_ids 与 test_id 提取标量特征矩阵；
        2) 仅用训练电池拟合 MinMaxScaler（避免测试信息泄露）；
        3) 对每块电池执行 transform；
        4) 进行滑动窗口切分；
        5) 封装为 PyTorch DataLoader。

        输入参数:
        - battery_dict: Dict[str, pd.DataFrame]
            电池数据字典。
        - train_ids: List[str]
            用于训练与拟合 scaler 的电池编号列表。
        - test_id: str
            用于测试的目标电池编号。
        - condition_map: Dict[str, Dict[str, float]]
            工况映射字典。
        - seq_length: int
            窗口长度。
        - batch_size: int
            训练集批大小。

        返回:
        - train_loader: DataLoader
            训练数据加载器。
        - test_loader: DataLoader
            测试数据加载器（batch_size=1，保持时序顺序）。
        - scaler_y: MinMaxScaler
            标签缩放器，用于预测后 inverse_transform。
        - test_actual_capacity: np.ndarray
            测试电池原始容量轨迹（未缩放），便于绘图与对比。
        """
        feature_mats = {
                bid: _extract_feature_matrix(battery_dict[bid], bid, condition_map)
                for bid in train_ids + [test_id]
        }

        train_concat = np.vstack([feature_mats[bid] for bid in train_ids])

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x.fit(train_concat[:, :-1])
        scaler_y.fit(train_concat[:, -1].reshape(-1, 1))

        x_train_list, y_train_list = [], []
        for bid in train_ids:
                mat = feature_mats[bid].copy()
                mat[:, :-1] = scaler_x.transform(mat[:, :-1])
                mat[:, -1:] = scaler_y.transform(mat[:, -1:].reshape(-1, 1))
                x_seq, y_seq = _to_windows(mat, seq_length)
                x_train_list.append(x_seq)
                y_train_list.append(y_seq)

        train_x = np.vstack(x_train_list)
        train_y = np.vstack(y_train_list)

        test_mat = feature_mats[test_id].copy()
        test_actual_capacity = test_mat[:, -1].copy()
        test_mat[:, :-1] = scaler_x.transform(test_mat[:, :-1])
        test_mat[:, -1:] = scaler_y.transform(test_mat[:, -1:].reshape(-1, 1))
        test_x, test_y = _to_windows(test_mat, seq_length)

        train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
        test_ds = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        return train_loader, test_loader, scaler_y, test_actual_capacity
