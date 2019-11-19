# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-03-29 17:06:48
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-03-30 19:02:24


import numpy as np

import sys

sys.path.append('../')

#from karman_data_iter import DataIter
from data_iter import DataIter
from srgan_solver import SRGANTrainer

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class Config(object):
    def __init__(self, lr, num_epoch, seed, upscale_factor):
        self.lr = lr
        self.num_epoch = num_epoch
        self.seed = seed
        self.upscale_factor = upscale_factor


if __name__ == '__main__':
    # Please change the following accordingly
    seed = 2
    #main_path = '/home/ziqizeng/Documents/Study/CSCI599_Deep_Learning/Project/final/data/'
    step = 4
    batch_size = 5
    train_portion = 0.2
    num_workers = 2
    num_channels = 1
    class_name = ['pressure', 'velocity', 'density']

    config = Config(1e-5, 10, seed, 2)
    for i in range(1):
        data_iter = DataIter(class_name[i], shuffle_list_seed=seed)
        file_cnt = len(data_iter)
        print(file_cnt)
        indices = list(range(file_cnt))
        split = int(np.floor(file_cnt * train_portion))

        print(split)

        train_sampler = SubsetRandomSampler(indices[:split])
        test_sampler = SubsetRandomSampler(indices[split:])

        train_data = DataLoader(data_iter, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
                                sampler=train_sampler)
        test_data = DataLoader(data_iter, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
                               sampler=test_sampler)

        #print (train_data.)

        # net = Net(num_channels=1, base_filter=64, upscale_factor=step)
        # net = SRCNN(num_channels=num_channels)
        trainer = SRGANTrainer(config, train_data, test_data, class_name[i])
        trainer.run()

    '''
    for xb, yb in train_data:
        print(xb.shape)
        print(yb.shape)
        zb = net(xb)
        print(zb.shape)
        break
    '''