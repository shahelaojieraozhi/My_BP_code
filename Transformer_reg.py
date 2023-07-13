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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# 定义Transformer模型
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers):
        super(TransformerRegressor, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # 平均池化操作，可以根据任务需求修改
        x = self.linear(x)
        return x


# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target = self.targets[idx]
        return input_data, target


# 设置超参数和训练过程
input_dim = 875
output_dim = 2
d_model = 256
nhead = 4
num_layers = 4
batch_size = 32
epochs = 10
learning_rate = 0.001

# 创建模型实例
model = TransformerRegressor(input_dim, output_dim, d_model, nhead, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备数据
# 假设你有一个名为"inputs"的输入数据张量，维度为[数据样本数, 875]，一个名为"targets"的目标张量，维度为[数据样本数, 2]
# 这里的数据可以是你自己的实际数据集
inputs = torch.randn(100, 875)  # 示例随机输入数据
targets = torch.randn(100, 2)  # 示例随机目标数据

dataset = CustomDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

# 使用训练好的模型进行预测
input_example = torch.randn(1, 875)  # 示例输入数据
with torch.no_grad():
    predicted_output = model(input_example)
    print("Predicted Output:", predicted_output)
