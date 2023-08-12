# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/12 10:35
@Author  : Rao Zhi
@File    : Transformer_reg.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super().__init__()
        # out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sigmoid = nn.Sigmoid()
        self.pred = pred
        if pred:
            self.fc2 = nn.Linear(hidden_features, 2)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        # print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        # print(x.size())
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if not self.pred:
            x += x1

        x = x.squeeze(0)
        x = self.sigmoid(x)

        return x


class TF(nn.Module):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.Block1 = Mlp(in_features=in_features, hidden_features=256, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=512, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_2 = Mlp(in_features=in_features, hidden_features=1024, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_3 = Mlp(in_features=in_features, hidden_features=2024, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_4 = Mlp(in_features=in_features, hidden_features=512, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_5 = Mlp(in_features=in_features, hidden_features=256, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_6 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.Block2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=True)

    def forward(self, x):
        x = self.Block1(x)
        # x = self.Block1_1(x)
        # x = self.Block1_2(x)
        # x = self.Block1_3(x)
        # x = self.Block1_4(x)
        # x = self.Block1_5(x)
        # x = self.Block1_6(x)
        x = self.Block2(x)
        return x
