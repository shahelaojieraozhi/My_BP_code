# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/18 15:17
@Author  : Rao Zhi
@File    : reshape_test_data.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

N_samples = 2.5e5
win_len = 875
ppg_sum = np.empty(shape=(int(N_samples), int(win_len)))
BP_sum = np.empty(shape=(int(N_samples), 2))
subject_idx_sum = np.empty(shape=(int(N_samples)))

idx = 1
root_path = r"G:\Blood_Pressure_dataset\cvprw\h5_record\test"
for sig_h5 in os.listdir(root_path):
    with h5py.File(os.path.join(root_path, sig_h5), 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')[:]
        BP = f.get('/label')[:]
        subject_idx = f.get('/subject_idx')[:]

        ppg_sum[(idx - 1) * 1000:idx * 1000, :] = ppg_h5
        BP_sum[(idx - 1) * 1000:idx * 1000, :] = BP
        subject_idx_sum[(idx - 1) * 1000:idx * 1000] = subject_idx
    idx += 1

np.savez("reshaped_test_dataset.npz", ppg_sum, BP_sum, subject_idx_sum, ['ppg', 'BP', 'subject_idx'])
