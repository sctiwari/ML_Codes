# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-03-29 17:15:22
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-03-30 18:59:07

import os

import numpy as np
import logging
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler


class DataIter(data.Dataset):
    def __init__(self,
                 class_name='pressure', # the name of the physical quantity
                 step=2,
                 shuffle_list_seed=None
                 ):

        super(DataIter, self).__init__()
        self.main_path = '/home/ziqizeng/Documents/Study/CSCI599_Deep_Learning/Project/final/data/data-128/data'
        self.class_name = class_name
        self.step = step
        self.file_paths = self._get_files()
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.file_paths)
        self.back_up = []
        
    def _get_files(self):
        class_dict = {} # {'pressure': [file1, file2, ...], 'density':[file1, file2, ...], 'velocity':[file1, file2, ...]}

        for dir_name in os.listdir(self.main_path):
            dir_path = os.path.join(self.main_path, dir_name)

            for file_name in os.listdir(dir_path):
                class_name = file_name.split('-')[0]
                if class_name not in class_dict:
                    class_dict[class_name] = []
                file_path = os.path.join(dir_path, file_name)
                class_dict[class_name].append(file_path)

        return class_dict[self.class_name]

    def get_arr(self, file_path):
        try:
            yb = np.load(file_path)
            if self.rng.rand() < 0.5:
                self.back_up.append(file_path)
        except:
            assert len(self.back_up) != 0, '******* No back up file to use, please change seed and run again *******'
            back_up_file = self.back_up[self.rng.randint(low=0, high=len(self.back_up))]
            logging.error('=========> Cannot load file {}, using back up file {} instead.'.format(
                                                file_path, back_up_file))
            yb = np.load(back_up_file)
        return yb

    def downsample(self, arr, step):
        assert arr.ndim == 3, '******* Please unify the input dimension *******'
        return arr[:, ::step, ::step]

        
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        yb = self.get_arr(file_path)[0].transpose(2, 0, 1)
        xb = self.downsample(yb, self.step).astype(np.float32)
        #print(xb.shape)
        return xb, yb



    def __len__(self):
        return len(self.file_paths)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # Please change the following accordingly
    seed = 2
    main_path = '/home/ziqizeng/Documents/Study/CSCI599_Deep_Learning/Project/srcnn/data/data'
    step = 4
    batch_size = 8
    train_portion = 0.2

    dataiter = DataIter(shuffle_list_seed=seed)
    print('dataiter has', len(dataiter),'files.')
    length = len(dataiter)
    for i in range(length):
        _, content = dataiter.__getitem__(i)
        print (content.shape)




