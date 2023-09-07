import json
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from Yaml_save import to_yaml
import os

def encode_board(board):
    """
    将19x19的围棋棋盘编码为19x19x3的形式。
    """
    black_channel = (board == 1).astype(np.float32)
    white_channel = (board == 2).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([black_channel, white_channel, empty_channel], axis=-1)

def dataset(path, save_name):    # 打开并读取JSON文件
    if not os.path.isfile(save_name + '.yaml'):
        to_yaml(path, save_name)
    with open(save_name+ '.yaml', 'r') as f:
        data = json.load(f)
    # 获取和打印数据的数量
    num_entries = len(data)
    print(f"JSON数据中共有{num_entries}条数据。")

    train_data = []
    dataloder = []
    target = []
    for key, value in data.items():
        train_data.append(key)
        dataloder.append(value)
    pattern = re.compile(r'\((.*?)\)')

    for index, str in enumerate(train_data):
        str = str.replace("（", "__(").replace("）", ")__")
        match = pattern.findall(str)
        target.append(float(match[0]))
    # 使用 np.newaxis 来增加一个新的维度

    data_reshaped = encode_board(np.array(dataloder))
    data_tensor = torch.from_numpy(data_reshaped)

    data_tensor = data_tensor.permute(0, 3, 1, 2)
    target_tensor = torch.tensor(target)

    # 使用unsqueeze来改变形状
    target = target_tensor.unsqueeze(1)
    print(data_tensor.shape)
    return target, data_tensor
# (batch_size, 通道数， 高，宽)
# (batch_size, 标签)
class CustomDataset(Dataset):
    def __init__(self, image_path,save_name):
        self.targets, self.data = dataset(image_path, save_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board = self.data[idx]

        sample = {'data': board, 'target': self.targets[idx]}
        return sample

