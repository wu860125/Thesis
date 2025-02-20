import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        
        data = raw_data[:-1]
        labels = raw_data[-1]

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.z = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr, mrr_arr = [], [], []

        past_wafer, future_step, wafer_len = [self.config[k] for k in ['past_wafer', 'future_step', 'wafer_len']]

        #future_step = wafer_len * (future_step - 1)
        
        node_num, total_time_len = data.shape

        # sensor data
        for i in range(wafer_len, total_time_len - wafer_len * (past_wafer - 1), wafer_len):
            ft = data[:, i - wafer_len:i]
            x_arr.append(ft)

        # MRR
        single_labels = torch.unique_consecutive(labels)
        for i in range(past_wafer, len(single_labels), 1):
            tar = single_labels[i].expand(1)#.repeat(node_num)
            y_arr.append(tar)

            mrr = single_labels[i - past_wafer:i]
            mrr_arr.append(mrr)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        z = torch.stack(mrr_arr).contiguous()
        
        return x, y, z

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()
        pastmrr = self.z[idx].double()

        edge_index = self.edge_index.long()

        return feature, y, edge_index, pastmrr