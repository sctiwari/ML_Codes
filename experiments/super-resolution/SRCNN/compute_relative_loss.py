# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-04-21 14:57:36
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-21 15:48:25

from torch.utils.data import DataLoader
from data.data_iter import DataIter
import torch
from torch.autograd import Variable
import numpy as np

if __name__ == '__main__':
    data_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\data\\data-128\\data'
    class_name = 'density'
    scale_factor = 2
    seed = 123
    use_resizer = True
    num_workers = 2
    batch_size = 16
    threshold = 1e-3

    data_iter = DataIter(data_path, class_name=class_name, scale_factor=scale_factor, shuffle_list_seed=seed, use_resizer=use_resizer)
    data = DataLoader(data_iter, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    density_model_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\models\\trained_models\\density\\20190419150454\\srcnn2d_epoch_70.pth'
    density_modified_model_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\models\\trained_models\\density\\20190419191825\\modified_srcnn2d_epoch_70.pth'


    loss = 0
    cnt = len(data_iter)
    net = torch.load(density_modified_model_path).cuda()

    for i, (xb, yb) in enumerate(data):
        print('===> Batch %4d/%4d' % (i+1, len(data)))
        xb = Variable(xb, requires_grad=False).float().cuda()
        yb = Variable(yb, requires_grad=False).float().cuda()
        output_map = net(xb)

        small_val_arr = torch.ones(yb.shape).float().cuda()*threshold
        denom = torch.max(yb, small_val_arr)
        relative = ((output_map - yb) / denom).cpu().data.numpy()
        loss += np.sum(np.mean(relative))

    loss = loss / cnt
    print('Loss =', loss)
