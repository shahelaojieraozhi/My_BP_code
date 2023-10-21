# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/22 11:43
@Author  : Rao Zhi
@File    : Transformer_reg_v2.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(input_dim, embedding_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        # a = self.embedding(torch.LongTensor(x))
        # b = self.position_embedding(positions)
        x = self.embedding(torch.LongTensor(x)) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        return x


# 定义回归模型
class RegressionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=1000, num_layers=4, num_heads=8, ff_dim=128, dropout=0.1):
        super(RegressionTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, embedding_dim, num_layers, num_heads, ff_dim, dropout)
        self.output1 = nn.Linear(embedding_dim, output_dim)
        self.output2 = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        output1 = self.output1(x)
        output2 = self.output2(x)
        return output1, output2


# # 定义输入信号长度和输出维度
# input_length = 875
# output_dim = 2
#
# # 构建模型
# model = RegressionTransformer(input_dim=input_length, output_dim=output_dim)
#
# # 打印模型结构
# print(model)

if __name__ == '__main__':
    inputs = torch.rand([2048, 1, 875])
    model = RegressionTransformer(inputs, )
