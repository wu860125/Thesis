import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, vm_pred, config=None, mode='train'):
        self.raw_data = raw_data
        self.mode = mode
        self.vm_pred = vm_pred

        self.config = config
        self.edge_index = edge_index

        x_data = raw_data[:-1]
        labels = raw_data[-1]
        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.z = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr, mrr_arr = [], [], []

        past_wafer, future_step = [self.config[k] for k in ['past_wafer', 'future_step']]

        past_win = 316 * past_wafer
        future_step = 316 * (future_step - 1)

        node_num, total_time_len = data.shape

        # sample setting
        mrr_all = torch.unique_consecutive(labels)

        # different setting for training and testing set
        if self.mode == 'train':
            rang1 = range(316 + 316 * 2, total_time_len - past_win + 316, 316 * 2)
            rang2 = range(past_win + 316 * 2, total_time_len - future_step, 316 * 2)
            mrr_all = mrr_all[2:]  # for add former MRR
            mrr_sampling= torch.tensor([mrr_all[i] if i % 2 == 0 else self.vm_pred[int(i / 2)] for i in range(len(mrr_all))])

        else:
            rang1 = range(316, total_time_len - past_win + 316, 316)
            rang2 = range(past_win, total_time_len - future_step, 316)
            mrr_sampling = torch.tensor([mrr_all[i] if i % 2 == 0 else self.vm_pred[i] for i in range(len(mrr_all))])

        for i in rang1:
            ft = data[:, i - 316:i]
            x_arr.append(ft)

        for i in rang2:
            tar = labels[i + future_step].expand(1)
            y_arr.append(tar)

            mrr = mrr_sampling[int(i / 316) - past_wafer:int(i / 316)]  # retain only one MRR for a wafer
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
