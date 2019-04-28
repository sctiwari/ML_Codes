# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2019-04-06 10:39:21
# @Last Modified by:   Jun Luo
# @Last Modified time: 2019-04-06 11:29:31

import os
import json

def get_files(data_path):
    class_dict = {} # {'pressure': [file1, file2, ...], 'density':[file1, file2, ...], 'velocity':[file1, file2, ...]}

    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)

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

    return class_dict

if __name__ == '__main__':
	path = 'C:\\Users\\CACS\\Desktop\\CSCI_599_2019\\data\\raw\\mantaflow\\mantaflow\\data'
	d = get_files(path)
	for name, lst in d.items():
		print('%s has file list length = %d' % (name, len(lst)))
	with open('file_dict.json', 'w') as f:
		json.dump(d, f)
	print('******* Dictionary saved *******')