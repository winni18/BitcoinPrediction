import pandas as pd
import numpy as np
import random


def balance_data(y):
    zero_index = np.squeeze(np.argwhere(y == 0).astype(int)).tolist()
    one_index = np.squeeze(np.argwhere(y == 1).astype(int)).tolist()
    two_index = np.squeeze(np.argwhere(y == 2).astype(int)).tolist()
    min_index_len = np.min([len(zero_index), len(one_index), len(two_index)])
    random.shuffle(zero_index)
    random.shuffle(one_index)
    random.shuffle(two_index)
    zero_index = zero_index[:min_index_len]
    one_index = one_index[:min_index_len]
    two_index = two_index[:min_index_len]
    index = []
    index.extend(zero_index)
    index.extend(one_index)
    index.extend(two_index)
    return index


def show_class_num(y):
    key = np.unique(y)
    result = {}
    for k in key:
        mask = (y == k)
        y_new = y[mask]
        v = y_new.size
        result[k] = v
    print(result)


def get_data(file_name, time_step=5):
    df = pd.read_csv(file_name)
    label = (df['Close'] - df['Open'])
    label = np.where(label > 0, 1, label)
    label = np.where(label == 0, 0, label)
    label = np.where(label < 0, 2, label)
    keep_keys = ['Open', 'High', 'Low', 'Close', 'Volume']
    for key in df.columns:
        if key not in keep_keys:
            df = df.drop(key, axis=1)
    data = df.values
    # data = data
    # normalized_data = data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # 归一化
    label = np.reshape(label, [-1, 1])
    normalized_data = np.hstack((data, label))
    data_x, data_y = [], []
    for i in range(len(normalized_data) - time_step - 1):
        x = normalized_data[i:i + time_step, :5]
        y = normalized_data[i + time_step + 1, -1]
        data_x.append(x)
        data_y.append(y)
    return data_x, data_y


def norm(data):

    data = np.squeeze(data)
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # 归一化
    normalized_data = np.expand_dims(normalized_data, axis=2)
    return normalized_data

