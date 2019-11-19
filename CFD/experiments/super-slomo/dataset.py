
''' Dataset iterator '''

import os
import re
import random

import numpy as np

def listdir(path, pattern=re.compile(r'.*')):
    return [os.path.join(path, f) for f in os.listdir(path) if pattern.match(f)]

def iterate_dataset(root_dir):
    vel_re = re.compile(r'velocity-(\d+).npy')

    def _get_pressure_at(dir_name, snap_time):
        return os.path.join(dir_name, f'pressure-{snap_time}.npy')

    def _get_density_at(dir_name, snap_time):
        return os.path.join(dir_name, f'density-{snap_time}.npy')

    def _get_snap_time(vel_file):
        dir_name, vel_file_name = os.path.dirname(vel_file), os.path.basename(vel_file)
        return int(vel_re.match(os.path.basename(vel_file_name)).group(1))

    for sess_dir in sorted(listdir(root_dir)):
        for vel_file in sorted(listdir(sess_dir, pattern=vel_re), key=_get_snap_time):
            snap_time = _get_snap_time(vel_file)
            yield {
                'sess_name': os.path.basename(sess_dir),
                'snap_time': snap_time,
                'vel': np.load(vel_file),
                'pressure': np.load(_get_pressure_at(sess_dir, snap_time)),
                'density': np.load(_get_density_at(sess_dir, snap_time)),
            }

def iterate_sequences(dataset_iter, window_size=1):
    ''' Returns lists of 3 of consequtive items '''
    prev_sess, prev_items = None, []
    for item in dataset_iter:
        if prev_sess != item['sess_name']:
            prev_items = []
        if len(prev_items) < 2 * window_size:
            pass
        else:
            yield [prev_items[0], prev_items[window_size], item]
            prev_items = prev_items[1:]
        prev_items.append(item)
        prev_sess = item['sess_name']

def shuffle(dataset_iter, buffer_size=350):
    buffer = []
    for item in dataset_iter:
        if len(buffer) < buffer_size:
            buffer.append(item)
        else:
            random.shuffle(buffer)
            while buffer:
                yield buffer[0]
                buffer = buffer[1:]

if __name__ == '__main__':

    # for data in iterate_dataset('data'):
    #     print(data['sess_name'], data['snap_time'])

    # for data in sequence_iterator(iterate_dataset('data')):
    #     print(data[0]['sess_name'], data[0]['snap_time'])
    #     print(data[1]['sess_name'], data[1]['snap_time'])
    #     print(data[2]['sess_name'], data[2]['snap_time'])
    #     print()

    pass