# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-03-29 17:15:22
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-19 14:49:03

import os
import logging
import sys

import numpy as np

from .transform import Resize

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

logging.basicConfig(level=logging.INFO)

class DataIter(data.Dataset):
    def __init__(self, 
                 data_path,
                 class_name='pressure', # the name of the physical quantity
                 scale_factor=4,
                 shuffle_list_seed=None,
                 use_resizer=False
                 ):

        super(DataIter, self).__init__()
        self.data_path = data_path
        self.class_name = class_name
        self.scale_factor = scale_factor
        self.file_paths = self._get_files()
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)

        self.use_resizer = use_resizer
        self.resizer_init = False
        # load video list
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.file_paths)
        self.back_up = []
        
    def _get_files(self):
        class_dict = {} # {'pressure': [file1, file2, ...], 'density':[file1, file2, ...], 'velocity':[file1, file2, ...]}

        for dir_name in os.listdir(self.data_path):
            dir_path = os.path.join(self.data_path, dir_name)

            for file_name in os.listdir(dir_path):
                try:
                    class_name = file_name.split('-')[0]
                except:
                    logging.error("********** file name has to be 'XXXX-XXX.npy' with XXXX being the physical quantity name **********")
                    logging.warning("********** Force exit **********")
                    sys.exit()
                if class_name not in class_dict:
                    class_dict[class_name] = []
                file_path = os.path.join(dir_path, file_name)
                class_dict[class_name].append(file_path)

        return class_dict[self.class_name]

    def downsample(self, arr):
        assert arr.ndim == 3, '******* Please unify the input dimension *******'
        if self.use_resizer:
            return arr[::self.scale_factor, ::self.scale_factor, :]
        return arr[:, ::self.scale_factor, ::self.scale_factor]


    def get_arr(self, file_path):
        try:
            yb = np.load(file_path)
            if self.rng.rand() < 0.5 and len(self.back_up) <= 200:
                self.back_up.append(file_path)
        except:
            assert len(self.back_up) != 0, '******* No back up file to use, please change seed and run again *******'
            back_up_file = self.back_up[self.rng.randint(low=0, high=len(self.back_up))]
            logging.error('=========> Cannot load file {}, using back up file {} instead.'.format(
                                                file_path, back_up_file))
            yb = np.load(back_up_file)
        return yb
        
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        if self.use_resizer:
            yb = self.get_arr(file_path)[0]
            # yb = (yb - np.min(yb)) / np.max(yb)
            xb = self.downsample(yb)
            if self.use_resizer:
                if not self.resizer_init:
                    self.resizer = Resize(yb.shape[:2][::-1]) # [::-1] is because cv2.resize uses the format of w first & h second.
                self.resizer_init = True
                xb = self.resizer(xb)
            yb = yb.transpose(2, 0, 1)
            xb = xb.transpose(2, 0, 1).astype(np.float32)
        else:
            yb = self.get_arr(file_path)[0].transpose(2, 0, 1)
            xb = self.downsample(yb).astype(np.float32)
        return xb, yb



    def __len__(self):
        return len(self.file_paths)



def get_train_test(**kwargs):
    seed = kwargs['seed']
    data_path = kwargs['data_path']
    scale_factor = kwargs['scale_factor']
    batch_size = kwargs['batch_size']
    train_portion = kwargs['train_portion']
    num_workers = kwargs['num_workers']
    num_channels = kwargs['num_channels']
    class_name = kwargs['class_name']
    if kwargs['model_name'].upper() == 'SRCNN2D' or kwargs['model_name'].upper() == 'MODIFIED_SRCNN2D':
        use_resizer = True
    else:
        use_resizer = False
    

    data_iter = DataIter(data_path, class_name=class_name, scale_factor=scale_factor, shuffle_list_seed=seed, use_resizer=use_resizer)
    file_cnt = len(data_iter)
    indices = list(range(file_cnt))
    split = int(np.floor(file_cnt*train_portion))

    train_sampler = SubsetRandomSampler(indices[:split])
    test_sampler  = SubsetRandomSampler(indices[split:])
    
    train_data = DataLoader(data_iter, batch_size=batch_size, num_workers=num_workers, pin_memory=False, sampler=train_sampler)
    test_data = DataLoader(data_iter, batch_size=batch_size, num_workers=num_workers, pin_memory=False, sampler=test_sampler)

    return train_data, test_data

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # Please change the following accordingly
    seed = 2
    data_path = 'E:\\mantaflow\\data'
    scale_factor = 4
    batch_size = 8
    train_portion = 0.2

    dataiter = DataIter(data_path, shuffle_list_seed=seed)
    print('dataiter has', len(dataiter),'files.')




