# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-04-21 10:54:57
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-21 12:07:21

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


def show_grid_2d(imgs):
    import matplotlib.gridspec as gridspec
    width = 10
    height = 2.5

    # fig = plt.figure(0, figsize=(10, 2.5))
    fig = plt.figure()
    fig.set_size_inches(width, height, forward=False)
    rows, columns = len(imgs), len(imgs[0])
    gs1 = gridspec.GridSpec(rows, columns)
    gs1.update(wspace=0025, hspace=0.05)
    
    for i in range(columns * rows):
        fig.add_subplot(gs1[i])
        plt.axis('off')
        plt.imshow(imgs[i // columns][(i) % columns])
    plt.savefig('test.png', dpi=300)


if __name__ == '__main__':

	folder_name = 'with obstacles'
	ori = mpimg.imread(os.path.join(folder_name, 'density_original.png'))
	small = mpimg.imread(os.path.join(folder_name, 'density_small_64.png'))
	srcnn = mpimg.imread(os.path.join(folder_name, 'density_generated_srcnn_64.png'))
	dsrcnn = mpimg.imread(os.path.join(folder_name, 'density_generated_modified_srcnn.png'))
	srcnn_dif = mpimg.imread(os.path.join(folder_name, 'density_difference_srcnn.png'))
	dsrcnn_dif = mpimg.imread(os.path.join(folder_name, 'density_difference_modified_srcnn.png'))

	imgs = [
			[ori,   srcnn,  dsrcnn],
			[small, srcnn_dif, dsrcnn_dif]
		   ]

	# imgs = [[]]
	# for file in os.listdir(os.path.join(os.getcwd(), folder_name)):
	# 	imgs[0].append(mpimg.imread(os.path.join(folder_name, file)))
	
	show_grid_2d(imgs)

