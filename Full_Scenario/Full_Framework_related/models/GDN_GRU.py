import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .GDN import GDN

import math

class GDN_GRU(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, input_dim=316, topk=12, past_wafer=4, gru_inter_dim=64, gru_layer_num=1):
        super(GDN_GRU, self).__init__()

        # 定義GDN層
        self.GDN_layer = GDN(edge_index_sets, node_num, dim, input_dim, topk)

        # 定義GRU層
        self.gru_layer = nn.GRU(int(node_num / past_wafer) * dim, gru_inter_dim, gru_layer_num, batch_first=True)
        
        # 定義MLP層
        self.mlp = nn.Sequential(nn.Linear(gru_inter_dim + past_wafer, 1),
                                 nn.ReLU()) 
        
        # 取出特徵用
        self.identity_layer = nn.Identity()

        self.dp = nn.Dropout(0.2)

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for name, param in self.gru_layer.named_parameters(): 
    #         if 'weight' in name:
    #             nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    #         elif 'bias' in name:
    #             nn.init.constant_(param, 0)
        
    #     # 初始化 MLP 層的權重
    #     for m in self.mlp:
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    #             if m.bias is not None:
    #                 fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    #                 bound = 1 / math.sqrt(fan_in)
    #                 nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, data, org_edge_index, pastmrr):
        
        # sensor and MRR data
        xs = data.clone().detach()
        mrr = pastmrr.clone().detach()

        # go through GDN layer
        gdn_out = self.GDN_layer(xs, org_edge_index)
        # reshape by (batch, wafer, features)
        gdn_out = gdn_out.reshape(gdn_out.shape[0], 4 ,-1)
        
        # go through GRU layer
        gru_out, _ = self.gru_layer(gdn_out)
        gru_out = gru_out[:, -1, :]
        
        # avoid extreme values
        gru_out = torch.clamp(gru_out, min=0, max=1)
        
        # concate with MRR data
        out = torch.cat((gru_out, mrr), dim=1)

        # go through MLP layer
        out = self.mlp(out)

        return out
