import os
import scipy.io
import pandas as pd

class NasaRawDataExtractor:
    """
    NASA 电池数据集原始数据提取器。
    遍历原始目录，解析 .mat 文件，提取充/放电阶段的所有原始数据数组，
    并按原有目录层级保存为 .pkl 格式，供下游任务加载。
    """
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

    def _parse_mat_file(self, file_path: str) -> pd.DataFrame:
        """
        [私有方法] 解析单个 .mat 文件。我很遗憾python没有 private 这是面向对象中设计上的失败，按理来说你不应当在任何的外部代码中显式的带哦用此函数
        动态读取数据结构，避免字段硬编码，确保数据无损传递。
        """
        mat_data = scipy.io.loadmat(file_path)
        battery_id = [k for k in mat_data.keys() if not k.startswith('__')][0]
        cycles = mat_data[battery_id][0, 0]['cycle'][0]
        
        parsed_data = []
        
        # 引入独立的分类计数器
        charge_count = 0
        discharge_count = 0
        
        for i in range(len(cycles)):
            cycle_type = cycles[i]['type'][0]
            
            if cycle_type == 'charge':
                charge_count += 1
                specific_index = charge_count
            elif cycle_type == 'discharge':
                discharge_count += 1
                specific_index = discharge_count
            else:
                continue # 忽略 impedance 等其他操作
                
            try:
                cycle_data = cycles[i]['data'][0, 0]
                
                cycle_dict = {
                    'Battery_ID': battery_id,
                    'Global_Step': i + 1,            # 绝对时间轴（保留原始文件中的物理操作顺序）
                    'Cycle_Type': cycle_type,        # 区分是充电还是放电
                    'Specific_Index': specific_index # 相对时间轴（如：第 5 次放电，极其重要！）
                }
                
                for field_name in cycle_data.dtype.names:
                    field_data = cycle_data[field_name][0]
                    if field_data.size == 1:
                        cycle_dict[field_name] = field_data[0]
                    else:
                        cycle_dict[field_name] = field_data
                        
                parsed_data.append(cycle_dict)
                
            except (IndexError, ValueError):
                continue
                
        return pd.DataFrame(parsed_data)

    def convert_dataset(self) -> None:
        """
        遍历原始数据目录，执行转换并按原结构写入 processed 目录。
        """
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"未找到原始数据目录: {self.raw_data_dir}")

        print(f"开始转换数据...\n源目录: {self.raw_data_dir}\n目标目录: {self.processed_data_dir}")

        for folder_name in os.listdir(self.raw_data_dir):
            raw_folder_path = os.path.join(self.raw_data_dir, folder_name)
            
            if os.path.isdir(raw_folder_path):
                # 在 processed 目录下镜像创建同名分组文件夹
                processed_folder_path = os.path.join(self.processed_data_dir, folder_name)
                os.makedirs(processed_folder_path, exist_ok=True)
                
                for file_name in os.listdir(raw_folder_path):
                    if file_name.endswith('.mat'):
                        raw_file_path = os.path.join(raw_folder_path, file_name)
                        battery_name = file_name.replace('.mat', '')
                        save_file_path = os.path.join(processed_folder_path, f"{battery_name}.pkl")
                        
                        try:
                            # 解析并保存
                            df_raw = self._parse_mat_file(raw_file_path)
                            df_raw.to_pickle(save_file_path)
                            print(f"转换成功: {folder_name}/{battery_name}.pkl (共 {len(df_raw)} 个充放电循环)")
                        except Exception as e:
                            print(f"解析失败: {raw_file_path}, 错误信息: {e}")

# ==========================================
# 执行入口
# ==========================================
if __name__ == "__main__":
    RAW_DIR = r"C:\Users\PLUTO\Desktop\battery-rul\data\raw"
    PROCESSED_DIR = r"C:\Users\PLUTO\Desktop\battery-rul\data\processed"
    
    extractor = NasaRawDataExtractor(RAW_DIR, PROCESSED_DIR)
    extractor.convert_dataset()