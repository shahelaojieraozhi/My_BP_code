# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/17 11:24
@Author  : Rao Zhi
@File    : pulse_filter.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import warnings
import pandas as pd
import torch
from tqdm import tqdm
from model.Resnet import resnet18

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# out_domain infer
# ppgs = torch.load("../data_normal/train/ppg.h5")
# ppgs = torch.load("../data_normal/val/ppg.h5")
ppgs = torch.load("../data_normal/test/ppg.h5")

model = resnet18(input_c=1, num_classes=1)
model = model.to(device)

model_name = 'abnormal_pulse_detection_total=1798_2023121711'
model.load_state_dict(torch.load('save/' + model_name + '/best_w.pth')['state_dict'])
best_epoch = torch.load('save/' + model_name + '/best_w.pth')["epoch"]
print(f"best epoch:{best_epoch}")
model.eval()

pre = []
with torch.no_grad():
    # for idx in range(len(ppgs)):
    for idx in tqdm(range(len(ppgs)), desc="Processing process", unit="sample"):
        ppg = ppgs[idx, :].unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        output = torch.sigmoid(model(ppg).cpu().squeeze())
        predict = (output >= 0.5).int()
        pre.append(predict.numpy())
        # plt.plot(ppg.squeeze().cpu())
        # plt.title(f"predict:{predict}")
        # plt.title(f"predict:{predict}, gt:{labels[i]}")
        # plt.show()

# print(pre.count(1))   # show count  of one in list
# pd.DataFrame(pre).to_csv("train_filter_idx.csv", index=False)
pd.DataFrame(pre).to_csv("test_filter_idx.csv", index=False)
# pd.DataFrame(pre).to_csv("val_filter_idx.csv", index=False)


