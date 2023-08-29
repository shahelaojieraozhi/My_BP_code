# -*- coding: utf-8 -*-
"""
@Project ：Blood_P 
@Time    : 2023/7/5 9:20
@Author  : Rao Zhi
@File    : save_data.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import h5py
import tensorflow as tf
# ks.enable_eager_execution()
import numpy as np
from sklearn.model_selection import train_test_split

from datetime import datetime
from os.path import expanduser, isdir, join
from os import mkdir
from sys import argv


def ppg_hdf2tfrecord(h5_file, tfrecord_path, samp_idx, weights_SBP=None, weights_DBP=None):
    # Function that converts PPG/BP sample pairs into the binary .tfrecord file format.
    # This function creates a .tfrecord file containing a defined number os samples

    # Parameters:
    # h5_file: file containing ppg and BP data
    # tfrecordpath: full path for storing the .tfrecord files
    # samp_idx: sample indizes of the data in the .h5 file to be stored in the .tfrecord file
    # weights_SBP: sample weights for the systolic BP (optional)
    # weights_DBP: sample weights for the diastolic BP (optional)

    N_samples = len(samp_idx)
    # open the .h5 file and get the samples with the indizes specified by samp_idx
    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')

        # writer = tf.io.TFRecordWriter(tfrecord_path)
        writer = h5py.File(tfrecord_path, 'w')

        ppg_ny = np.zeros([len(samp_idx), 875])
        label_ny = np.zeros([len(samp_idx), 2])
        subject_idx_ny = np.zeros(len(samp_idx))
        weight_SBP_ny = np.ones(len(samp_idx))
        weight_DBP_ny = np.ones(len(samp_idx))
        Nsamples_ny = np.zeros(len(samp_idx))

        # iterate over each sample index and convert the corresponding data to a binary format
        count = 0
        for i in np.nditer(samp_idx):
            ppg = np.array(ppg_h5[i, :])

            target = np.array(BP[i, :], dtype=np.float32)
            sub_idx = np.array(subject_idx[i], dtype=np.int)  # 7/10 修改

            ppg_ny[count, :] = ppg
            label_ny[count, :] = target
            subject_idx_ny[count] = sub_idx
            Nsamples_ny[count] = N_samples

            count += 1

        writer.create_dataset('ppg', data=ppg_ny)
        writer.create_dataset('label', data=label_ny)
        writer.create_dataset('subject_idx', data=subject_idx_ny)
        writer.create_dataset('weight_SBP', data=weight_SBP_ny)
        writer.create_dataset('weight_DBP', data=weight_DBP_ny)
        writer.create_dataset('Nsamples', data=Nsamples_ny)

        writer.close()


def ppg_hdf2tfrecord_sharded(h5_file, samp_idx, tfrecordpath, Nsamp_per_shard, modus='train', weights_SBP=None,
                             weights_DBP=None):
    # Save PPG/BP pairs as .tfrecord files. Save defined number os samples per file (Sharding)
    # Weights can be defined for each sample
    #
    # Parameters:
    # h5_file: File that contains the whole dataset (in .h5 format), created by
    # samp_idx: sample indizes from the dataset in the h5. file that are used to create this tfrecords dataset
    # tfrecordpath: full path for storing the .tfrecord files   ex: train  val test
    # N_samp_per_shard: number of samples per shard/.tfrecord file
    # modus: define if the data is stored in the "train", "val" or "test" subfolder of "tfrecordpath"
    # weights_SBP: sample weights for the systolic BP (optional)
    # weights_DBP: sample weights for the diastolic BP (optional)

    base_filename = join(tfrecordpath, 'MIMIC_III_ppg')

    N_samples = len(samp_idx)

    # calculate the number of Files/shards that are needed to store the whole dataset
    N_shards = np.ceil(N_samples / Nsamp_per_shard).astype(int)

    # iterate over every shard
    for i in range(N_shards):
        idx_start = i * Nsamp_per_shard
        idx_stop = (i + 1) * Nsamp_per_shard
        if idx_stop > N_samples:
            idx_stop = N_samples

        idx_curr = samp_idx[idx_start:idx_stop]
        output_filename = '{0}_{1}_{2:05d}_of_{3:05d}.h5'.format(base_filename,
                                                                 modus,
                                                                 i + 1,
                                                                 N_shards)
        # OUTPUT file name is MIMIC_III_ppg_test_00001_of_00250.tfrecord
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, ': processing ',
              modus,
              ' shard ', str(i + 1), ' of ', str(N_shards))
        ppg_hdf2tfrecord(h5_file, output_filename, idx_curr, weights_SBP=weights_SBP, weights_DBP=weights_DBP)


def h5_to_tfrecords(SourceFile, tfrecordsPath, N_train=1e6, N_val=2.5e5, N_test=2.5e5, save_tfrecords=True):
    N_train = int(N_train)
    N_val = int(N_val)
    N_test = int(N_test)

    tfrecord_path_train = join(tfrecordsPath, 'train')
    tfrecord_path_val = join(tfrecordsPath, 'val')
    tfrecord_path_test = join(tfrecordsPath, 'test')
    os.makedirs(tfrecord_path_train, exist_ok=True)
    os.makedirs(tfrecord_path_val, exist_ok=True)
    os.makedirs(tfrecord_path_test, exist_ok=True)

    csv_path = tfrecordsPath

    Nsamp_per_shard = 1000

    with h5py.File(SourceFile, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP)
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1]
    subject_idx = subject_idx[:N_samp_total]

    # Divide the dataset into training, validation and test set
    # -------------------------------------------------------------------------------

    valid_idx = np.arange(subject_idx.shape[-1])  # subject_idx.shape[-1] = 9054000

    # divide the subjects into training, validation and test subjects
    subject_labels = np.unique(subject_idx)
    subjects_train_labels, subjects_val_labels = train_test_split(subject_labels, test_size=0.5)
    subjects_val_labels, subjects_test_labels = train_test_split(subjects_val_labels, test_size=0.5)

    # Calculate samples belong to training, validation and test subjects
    train_part = valid_idx[np.isin(subject_idx, subjects_train_labels)]
    val_part = valid_idx[np.isin(subject_idx, subjects_val_labels)]
    test_part = valid_idx[np.isin(subject_idx, subjects_test_labels)]

    # draw a number samples defined by N_train, N_val and N_test from the training, validation and test subjects
    idx_train = np.random.choice(train_part, N_train, replace=False)
    idx_val = np.random.choice(val_part, N_val, replace=False)
    idx_test = np.random.choice(test_part, N_test, replace=False)

    # save ground truth BP values of training, validation and test set in csv-files for future reference
    BP_train = BP[:, idx_train]
    d = {"SBP": np.transpose(BP_train[0, :]), "DBP": np.transpose(BP_train[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + 'MIMIC-III_BP_trainset.csv')
    BP_val = BP[:, idx_val]
    d = {"SBP": np.transpose(BP_val[0, :]), "DBP": np.transpose(BP_val[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + 'MIMIC-III_BP_valset.csv')
    BP_test = BP[:, idx_test]
    d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + 'MIMIC-III_BP_testset.csv')

    # create tfrecord dataset
    # ----------------------------
    if save_tfrecords:
        np.random.shuffle(idx_train)  # 再打乱一次有啥意义 ?
        ppg_hdf2tfrecord_sharded(SourceFile, idx_test, tfrecord_path_test, Nsamp_per_shard, modus='test')
        ppg_hdf2tfrecord_sharded(SourceFile, idx_train, tfrecord_path_train, Nsamp_per_shard, modus='train')
        ppg_hdf2tfrecord_sharded(SourceFile, idx_val, tfrecord_path_val, Nsamp_per_shard, modus='val')
    print("Script finished")


if __name__ == "__main__":
    np.random.seed(seed=42)

    SourceFile = 'G:\\Blood_Pressure_dataset\\cvprw\\MIMIC-III_ppg_dataset.h5'
    tfrecordsPath = './h5_record'
    N_train = 1e6
    N_val = 2.5e5
    N_test = 2.5e5
    divbysubj = True

    h5_to_tfrecords(SourceFile=SourceFile, tfrecordsPath=tfrecordsPath,
                    N_train=N_train, N_val=N_val, N_test=N_test)

"""
为了训练神经网络，prepare_MIMIC_dataset.py脚本创建的数据集必须分为训练集、验证集和测试集。
h5_to_tfrecord.py脚本通过以下两种方式来划分数据集:
(a)基于主题的划分;
(b)根据用户的选择随机分配样本。
数据将分别存储在.tfrecord文件中用于训练、验证和测试集，这些文件将在训练期间使用。
"""
