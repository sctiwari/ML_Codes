# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-04-06 20:20:18
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-06 21:46:10

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
	path = 'pics\\effect_pics'
	class_names = ['density', 'pressure', 'velocity']
	suf = ['_original.png', '_small.png', '_generated.png', '_generated_ziqizeng.png']
	img_type = ['Original 128x128', 'Low-Res 32x32', 'SRCNN 128x128', 'SRGAN 128x128']
	fig = plt.figure()
	for i in range(3):
		for j in range(4):
			plt.subplot(3, 4, i*4+j+1)
			if i == 0:
				plt.title(img_type[j])
			if j == 0:
				plt.ylabel(class_names[i])
			plt.axis('off')
			img = mpimg.imread(os.path.join(path, class_names[i]+suf[j]))
			plt.imshow(img)

	plt.show()
	# fig.savefig('integrated.png', bbox_inches='tight')




