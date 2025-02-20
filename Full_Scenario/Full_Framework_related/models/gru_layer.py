import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRULayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定義GRU層
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定義全連接層
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(),
            #nn.Linear(64, output_size),
            #nn.LeakyReLU()
            )
        
        #self.relu = nn.LeakyReLU()


    def forward(self, x):
        # 初始化隱藏狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(x.device)

        # 前向傳播，x的形狀應該是 (batch_size, sequence_length, input_size)
        out, _ = self.gru(x, h0)

        # 只取序列的最後一個時間步的輸出
        #out = self.mlp(out[:, -1, :])
        #out = self.relu(out[:, -1, :])

        return out[:, -1, :]