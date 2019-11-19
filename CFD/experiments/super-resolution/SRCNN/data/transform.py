# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-03-30 21:47:58
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-19 13:44:10

import numpy as np
import cv2

class Resize():
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_CUBIC
    """
    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size # [w, h]
        self.interpolation = interpolation

    def __call__(self, data):
        h, w, c = data.shape

        if isinstance(self.size, int):
            slen = self.size
            if min(w, h) == slen:
                return data
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), self.interpolation)
        else:
            scaled_data = data
        if c == 1:
            scaled_data = scaled_data[:, :, np.newaxis]
        return scaled_data

if __name__ == '_main_':
    path = 'D:\\Documents\\CSCI 599\\Final Project\\src\\spatial_expansion\\data\\raw\\2019-03-24_16-11-50_res_128\\density-100.npy'
    step = 4
    yb = np.load(path)[0]
    xb = yb[::step, ::step, :]
    print('Shrinked original map xb.shape = {}, max: {}, min: {}'.format(xb.shape, np.max(xb), np.min(xb)))
    print('Original map yb.shape =', yb.shape)

    resizer = Resize(yb.shape[:2])
    zb = resizer(xb)

    print('resized map zb.shape =', zb.shape)
    print('shrinked resized map shrink_zb.shape =', shrink_zb.shape)

    print('\n\n{}/{} pixels of shrinked resized map have the same value with the orinal shrinked map'.format(
                                np.sum(shrink_zb == xb), np.multiply(*shrink_zb.shape[:2])))
    print('Total difference:', np.sum(np.abs(shrink_zb - xb)))