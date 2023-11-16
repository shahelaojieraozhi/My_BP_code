# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/29 14:10
@Author  : Rao Zhi
@File    : rPPG_pulse_preprocess.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import numpy as np
import h5py

pulse_path = r'G:\Blood_Pressure_dataset\ours\subject-1\rPPG_pulse'
bp_path = r'G:\Blood_Pressure_dataset\ours\subject-1\subject_bp.csv'
save_path = 'rPPG_bp_subject.h5'

# bp value modify
bp = np.loadtxt(bp_path, delimiter=',')
bp_numbers = len(bp)  # number of bp sampling points
new_bp = np.zeros([bp_numbers * 2, 3])  # one minute a sample
bp_ = bp.copy()
for each_point in range(bp_numbers):
    new_bp[2 * each_point, :] = np.array(bp[each_point], dtype=np.float32)
    new_bp[2 * each_point + 1, :] = np.array(bp_[each_point], dtype=np.float32)

# define all the containers
container = []
rPPG = np.empty([bp_numbers * 2, 875])
subject_id = np.ones(bp_numbers * 2)
writer = h5py.File(save_path, 'w')

# get all signals
order_pulse_list = os.listdir(pulse_path)
order_pulse_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
for each_section in order_pulse_list:
    whole_signal = np.loadtxt(os.path.join(pulse_path, each_section), delimiter=',')
    container.append(whole_signal)

all_rPPG = np.concatenate(container, axis=0)

# cut signal(reshape)
for i in range(bp_numbers * 2):
    rPPG[i, :] = np.array(all_rPPG[i * 875: (i + 1) * 875], dtype=np.float32)

# save
writer.create_dataset('rPPG', data=rPPG)
writer.create_dataset('label', data=new_bp)
writer.create_dataset('subject_idx', data=subject_id)
writer.close()
