# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-03-29 17:06:48
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-19 15:03:12

import os
import argparse
import sys
import datetime
import json
from math import log10
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from data.data_iter import get_train_test
from network.srcnn_model import SRCNN2D, MODIF_SRCNN2D

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')

parser.add_argument('--class_name', '-p', type=str, default='density', help='class name')
parser.add_argument('--model_name', type=str, default='srcnn2d', help='model name')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--data_path', type=str, default='C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\data\\raw\\mantaflow\\mantaflow\\data',
                        help='data path')
# parser.add_argument('--data_path', type=str, default='D:\\Documents\\CSCI 599\\Final Project\\src\\data',
#                     help='data path')

parser.add_argument('--scale_factor', type=int, default=2, help='up and down scaling factor')

parser.add_argument('--num_workers', type=int, default=2, help='number of subprocesses')
parser.add_argument('--num_channels', type=int, default=1, help='number of channels, e.g. BW or RGB')




parser.add_argument('--save_epoch', type=int, default=1, help='save the model every x epochs')
parser.add_argument('--save_path', type=str, default='models\\trained_models', help='data path')

parser.add_argument('--save_stat_pic', type=bool, default=True, help='whether save the showed picture of statistics')
parser.add_argument('--stat_iter', type=int, default=100, help='show (or possibly save) statistics every x iterations')
parser.add_argument('--stat_save_path', type=str, default='stats\\', help='statistics saving path')

parser.add_argument('--logging_file_save_path', type=str, default='logging\\', help='statistics saving path')


parser.add_argument('--use_gpu', type=bool, default=True, help='whether use GPU')
parser.add_argument('--gpus', type=int, default=0, help="define gpu id")





parser.add_argument('--num_epochs', type=int, default=70, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--train_portion', type=float, default=0.8, help='ratio: number of training data / whole number of data')
parser.add_argument('--test_iter', type=int, default=50, help='test the average psnr on test set every x iterations')


parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning Rate decay')
parser.add_argument('--lr_decay_step', type=int, default=1, help='learning decay step')


def build(args):
    assert os.path.exists(args.data_path), '******* Data path not exist *******'
    kwargs = {}
    kwargs.update(vars(args))


    logging_file_save_dir_path = os.path.join(args.logging_file_save_path, args.class_name)
    logging_file_save_dir_path = os.path.join(os.getcwd(), logging_file_save_dir_path)
    logging_file_name = ''.join(s for s in str(datetime.datetime.now()).split('.')[0] if s.isdigit())
    logging_file_save_path = os.path.join(logging_file_save_dir_path, logging_file_name+'.log')

    if not os.path.exists(logging_file_save_dir_path):
        os.makedirs(logging_file_save_dir_path)

    handlers = [logging.FileHandler(logging_file_save_path), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        # format='%(asctime)s: %(message)s',
                        # datefmt='%Y-%m-%d %H:%M:%S',
                        handlers = handlers)
    logging.info(json.dumps(kwargs, indent=2))

    


    args.save_path = os.path.join(args.save_path, args.class_name)
    args.save_path = os.path.join(os.getcwd(), args.save_path)
    model_save_folder_name = logging_file_name
    args.save_path = os.path.join(args.save_path, model_save_folder_name)


    args.stat_save_path = os.path.join(args.stat_save_path, args.class_name)
    args.stat_save_path = os.path.join(os.getcwd(), args.stat_save_path)
    stat_folder_name = model_save_folder_name
    args.stat_save_path = os.path.join(args.stat_save_path, stat_folder_name)

    



    train_data, test_data = get_train_test(**kwargs)
    if args.model_name == 'srcnn2d':
        net = SRCNN2D(num_channels=args.num_channels, name=args.model_name)
    elif args.model_name == 'modified_srcnn2d':
        net = MODIF_SRCNN2D(num_channels=args.num_channels, name=args.model_name)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    if args.use_gpu:
        assert torch.cuda.is_available(), "******* CUDA is not available *******"
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        net = net.cuda(args.gpus)
        criterion = criterion.cuda(args.gpus)

    return kwargs.copy(), train_data, test_data, net, criterion, optimizer, scheduler

# def get_test_avg_psnr(test_data, net, criterion):
#     avg_psnr = 0
#     on_cuda = next(net.parameters()).is_cuda

#     for xb, yb in test_data:
#         if on_cuda:
#             xb = Variable(xb).float().cuda(args.gpus)
#             yb = Variable(yb).float().cuda(args.gpus)
#         else:
#             xb = Variable(xb)
#             yb = Variable(yb)

#         output_map = net(xb)

#         mse = criterion(output_map, yb)
#         psnr = 10 * log10(1 / mse.item())
#         avg_psnr += psnr
#     return avg_psnr / len(test_data)

def get_test_loss(test_data, net, criterion):
    on_cuda = next(net.parameters()).is_cuda
    loss = 0

    for xb, yb in test_data:
        if on_cuda:
            xb = Variable(xb).float().cuda(args.gpus)
            yb = Variable(yb).float().cuda(args.gpus)
        else:
            xb = Variable(xb).float()
            yb = Variable(yb).float()
        # logging.info('Test {}/{}'.format(i+1, len(test_data)))
        output_map = net(xb)
        loss += criterion(output_map, yb).item()
    return loss / len(test_data)

# def save_statistics(args, kwargs, loss_history, avg_psnr_history, output):
#     if not os.path.exists(args.stat_save_path):
#         logging.warning('===> Creating path {} to save statistics for this run.'.format(args.stat_save_path))
#         os.makedirs(args.stat_save_path)

#         with open(os.path.join(args.stat_save_path, 'settings.txt'), 'w') as f:
#             json.dump(kwargs, f, indent=2, ensure_ascii=False)
#         logging.warning('===> Finished.')

#     fig, ax1 = plt.subplots()
#     color = colors[0]
#     ax1.set_xlabel('Iteration')
#     ax1.set_ylabel('Loss', color=color)
#     line1 = ax1.plot(range(1, iteration+1), loss_history, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#     color = colors[1]
#     ax2.set_ylabel('Test Avg PSNR', color=color)  # we already handled the x-label with ax1
#     line2 = ax2.plot(avg_psnr_history.keys(), avg_psnr_history.values(), color=color)
#     ax2.tick_params(axis='y', labelcolor=color)

#     pic_name = "stat_iter_{}.png".format(iteration)
#     pic_path = os.path.join(args.stat_save_path, pic_name)
#     fig.savefig(pic_path)
#     output = output +'\tstatistics saved at {}'.format(pic_path)
#     return output

def save_statistics(args, kwargs, loss_history, test_loss_history, output):
    if not os.path.exists(args.stat_save_path):
        logging.warning('===> Creating path {} to save statistics for this run.'.format(args.stat_save_path))
        os.makedirs(args.stat_save_path)

        with open(os.path.join(args.stat_save_path, 'settings.txt'), 'w') as f:
            json.dump(kwargs, f, indent=2, ensure_ascii=False)
        logging.warning('===> Finished.')

    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.plot(range(1, iteration+1), loss_history, color=colors[0], label='Training Loss')
    ax1.plot(test_loss_history.keys(), test_loss_history.values(), color=colors[1], label='Test Set Loss', linewidth=2)
    ax1.legend(loc='upper right')
    
    pic_name = "stat_iter_{}.png".format(iteration)
    pic_path = os.path.join(args.stat_save_path, pic_name)
    fig.savefig(pic_path)
    plt.close()
    output = output +'\tstatistics saved at {}'.format(pic_path)
    return output

if __name__ == '__main__':
    args = parser.parse_args()
    kwargs, train_data, test_data, net, criterion, optimizer, scheduler = build(args)

    iteration = 0
    loss_history = []
    # avg_psnr_history = {}
    test_loss_history = {}

    # for save the plot
    if args.save_stat_pic:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
    # enable cudnn tune
    cudnn.benchmark = True

    # Start training
    logging.info('===> Start training ...')
    for epoch in range(args.num_epochs):
        for i, (xb, yb) in enumerate(train_data):
            start = time.time()
            iteration += 1
            if args.use_gpu:
                xb = Variable(xb).float().cuda(args.gpus)
                yb = Variable(yb).float().cuda(args.gpus)
            else:
                xb = Variable(xb).float()
                yb = Variable(yb).float()

            output_map = net(xb)
            loss = criterion(output_map, yb)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            output = "===> Epoch[{:2d}/{:2d}]({:4d}/{:4d}): Loss: {:5.4f}".format(
                        epoch+1, args.num_epochs, i+1, len(train_data), float(loss.item()))


            
            end = time.time()
            output += '\ttime: {:3f}s'.format(end-start)

            if args.save_stat_pic and iteration%args.stat_iter == 0:
                # output = save_statistics(args, kwargs, loss_history, avg_psnr_history, output)
                output = save_statistics(args, kwargs, loss_history, test_loss_history, output)

            
            if iteration%args.test_iter == 0:
                start = time.time()
                # avg_psnr = get_test_avg_psnr(test_data, net, criterion)
                # output += '\tTest set Avg PSNR: {:4.4f} dB'.format(float(avg_psnr))
                # avg_psnr_history[iteration] = avg_psnr
                test_loss = get_test_loss(test_data, net, criterion)
                output += '\tTest set loss: {:4.4f}'.format(float(test_loss))
                test_loss_history[iteration] = test_loss
                end = time.time()
                output += '\ttime: {:3f}s'.format(end-start)


            logging.info(output)




        
        scheduler.step()



        if (epoch+1)%args.save_epoch == 0:
            if not os.path.exists(args.save_path):
                logging.info('===> Creating path {} to save trained models.'.format(args.save_path))
                os.makedirs(args.save_path)
                with open(os.path.join(args.save_path, 'settings.txt'), 'w') as f:
                    json.dump(kwargs, f, indent=2, ensure_ascii=False)

            model_name = "{}_epoch_{}.pth".format(net.name, epoch+1)
            model_path = os.path.join(args.save_path, model_name)
            torch.save(net, model_path)
            logging.info('********** Model {} saved at {} **********\n'.format(os.path.basename(model_path), args.save_path))

