import pandas as pd
import numpy as np


def get_data(file_name, time_step=5):
    df = pd.read_csv(file_name)
    # label = np.where(label >= 0, 1, 0)
    # label = np.where(label == 0, 0, label)
    # label = np.where(label < 0, 2, label)

    data = df.iloc[:, 1:].values
    std_data = np.std(data, axis=0)
    mean_data = np.mean(data, axis=0)
    # normalized_data = data
    normalized_data = (data - mean_data) / std_data  # 归一化
    data_x, data_y = [], []
    for i in range(len(normalized_data) - time_step - 1):
        x = normalized_data[i:i + time_step, :5]
        y = normalized_data[i + time_step, -2]
        data_x.append(x)
        data_y.append(y)
    return data_x, data_y, mean_data, std_data



