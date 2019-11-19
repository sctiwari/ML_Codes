# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-04-05 23:45:36
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-21 12:16:18

import os
import argparse
import json
import logging

import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch

from data.transform import Resize

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--cls', '-c', type=str, default='density', help='class name')
parser.add_argument('--seed', '-s', type=int, default=123, help='seed')
parser.add_argument('--remove_bg', type=bool, default=False, help='whether remove background')
parser.add_argument('--add_obstacles', type=str, default='True', help='whether add obstacles')
parser.add_argument('--test', type=int, default=123, help='test')


class Standardizer():
    def __init__(self, original):
        self.ori_min = np.min(original)
        self.ori_max = np.max(original)

    def standardize(self, arr):
        return (arr - np.min(arr)) / np.max(arr) * (self.ori_min+self.ori_max) - self.ori_min


def remove_background(arr):
    threshold = 1e-2
    arr[arr<=threshold] = np.nan
    return arr

# def save_without_border(data, filename, cmap='jet', scale=4):
#     sizes = np.shape(data)
#     height = float(sizes[0])
#     width = float(sizes[1])
     
#     fig = plt.figure()
#     fig.set_size_inches(width/height, 1, forward=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
 
#     ax.imshow(data, cmap=cmap)
#     plt.savefig(filename, dpi = scale*height) 
#     plt.close()

def save_image(arr, name, save_path, vmin, vmax, step, f=None, scale=4, cmap='jet'):
    use_f = (f is not None) and (not name.startswith('difference'))
    if use_f:
        if f.shape[0] != image.shape[0]:
            f = f[::step, ::step]
            scale = scale * step


    # output
    sizes = np.shape(arr)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    if name.startswith('difference'):
        cmap = 'GnBu'
        # arr[arr<1e-3] = 0.0

    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    if use_f:
        ax.imshow(f, cmap='binary', alpha=0.4, animated=True)

    plt.savefig(os.path.join(save_path, '%s_%s.png' % (args.cls, name)), dpi=scale*height) 
    plt.close()
    # fig, ax = plt.subplots()
    
    # if use_f:
    #     ax.imshow(f, cmap='binary', alpha=0.4, animated=True)
    # ax.axis('off')
    # fig.savefig(os.path.join(save_path, '%s_%s.png' % (args.cls, name)), bbox_inches='tight')
    # plt.close()
    return

if __name__ == '__main__':
    args = parser.parse_args()
    step = 2
    rng = np.random.RandomState(args.seed)

    with open('file_dict.json', 'r') as f:
        file_dict = json.load(f)

    save_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\pics\\new pics'
    if args.add_obstacles == 'True':
        save_path = os.path.join(save_path, 'with obstacles')
    else:
        save_path = os.path.join(save_path, 'without obstacles')
    model_data_dict = {'density':{}, 'pressure':{}, 'velocity':{}}

    density_model_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\models\\trained_models\\density\\20190419150454\\srcnn2d_epoch_70.pth'
    pressure_model_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\models\\trained_models\\pressure\\20190403101521\\srcnn2d_epoch_150.pth'
    velocity_model_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\models\\trained_models\\velocity\\20190403182351\\srcnn2d_epoch_150.pth'

    density_modified_model_path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\spatial_expansion\\models\\trained_models\\density\\20190419191825\\modified_srcnn2d_epoch_70.pth'
    # density_data_path = file_dict['density'][rng.randint(0, high=len(file_dict['density']))]
    # pressure_data_path = file_dict['pressure'][rng.randint(0, high=len(file_dict['pressure']))]
    # velocity_data_path = file_dict['velocity'][rng.randint(0, high=len(file_dict['velocity']))]

    # model_data_dict['density'].update({'model': density_model_path})
    # model_data_dict['density'].update({'data': density_data_path})

    # model_data_dict['pressure'].update({'model': pressure_model_path})
    # model_data_dict['pressure'].update({'data': pressure_data_path})
    
    # model_data_dict['velocity'].update({'model': velocity_model_path})
    # model_data_dict['velocity'].update({'data': velocity_data_path})

    model_data_dict['density'].update({'model': density_model_path, 'modified_model': density_modified_model_path})
    model_data_dict['density'].update({'data': 'new_data\\with_flags\\density-400.npy'})

    if args.add_obstacles == 'True':
        try:
            model_data_dict['density'].update({'flag': 'new_data\\with_flags\\flags-400.npy'})
            f = np.load(model_data_dict['density']['flag'])
            f = np.squeeze(f, axis=2).T.astype(float)
            f[(f % 4 != 2)] = np.nan
            f[f % 4 == 2] = 1.0
            f[0, 0] = 0.0
        except:
            f = None
    else:
        f = None



    model_data_dict['pressure'].update({'model': pressure_model_path})
    model_data_dict['pressure'].update({'data': 'new_data\\pressure-100.npy'})
    
    model_data_dict['velocity'].update({'model': velocity_model_path})
    model_data_dict['velocity'].update({'data': 'new_data\\velocity-100.npy'})

    file_name = model_data_dict[args.cls]['data']
    yb = np.load(file_name)[0]
    xb = yb[::step, ::step, :]
    

    original = yb.copy()[:, :, 0].astype(np.float32)
    small = xb.copy()[:, :, 0].astype(np.float32)

    # srcnn special
    resizer = Resize(yb.shape[:2][::-1])
    xb = resizer(xb)
    bicubic = xb.copy()[:, :, 0].astype(np.float32)

    yb = torch.from_numpy(yb.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)).cuda(1)
    xb = torch.from_numpy(xb.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)).cuda(1)



    net = torch.load(model_data_dict[args.cls]['model']).cuda(1)
    y_hat = net(xb)
    generated_srcnn = y_hat.data[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
    difference_srcnn = np.abs(generated_srcnn-original)


    net = torch.load(model_data_dict[args.cls]['modified_model']).cuda(1)
    y_hat = net(xb)
    generated_modified = y_hat.data[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
    difference_modified = np.abs(generated_modified-original)


    images = [original, small, bicubic, generated_srcnn, difference_srcnn, generated_modified, difference_modified]
    image_names = ['original', 'small_64', 'bicubic', 'generated_srcnn_64', 'difference_srcnn', 'generated_modified_srcnn', 'difference_modified_srcnn']

    # remove background
    if args.remove_bg:
        images = [remove_background(arr) for arr in images]
    
    # stdzr = Standardizer(original)


    # save pics
    vmin = np.min(original)
    vmax = np.max(original)
    saved_images = []
    not_saved_images = []
    for image, name in zip(images, image_names):
        try:
            save_image(image, name, save_path, vmin, vmax, step, f)
            saved_images.append(name)
        except Exception as e:
            print(e)
            not_saved_images.append(name)

    if len(saved_images) == len(image_names):
        print('******* All %d pics  are saved (%s) *******' % (len(image_names), str(image_names)))
    else:
        print('******* %s saved *******' % str(saved_images))
        print('******* %s not saved *******' % str(not_saved_images))

    handlers = [logging.FileHandler('{}_test.log'.format(args.cls)), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, 
                        handlers = handlers)
    file_name = os.path.join(os.path.basename(os.path.dirname(file_name)), os.path.basename(file_name))
    logging.info(file_name)




