import os
import numpy as np
from utils.misc import write_list_to_txt, read_txt_as_list
import random

"""
10355
train: 5285  0-5284         # 分出 1000 作为 valid
test:  5050  5285-10334     # 分出 1000 作为 valid

"""

root = '/datasets/rgbd_dataset/SUNRGBD'


def check_list_path(plist):
    ERROR = False
    for p in plist:
        if os.path.exists(p):
            continue
        else:
            ERROR = True
            print(p, 'not exist!')
    if not ERROR:
        print('all pass!')


def save_train_test_txt():  # valid 只要随机筛选路径即可
    for split in ['train', 'test']:
        img_dir = os.path.join(root, split, 'image')
        depth_dir = os.path.join(root, split, 'depth')  # png, 16bits
        target_dir = os.path.join(root, split, 'mask')  # npy

        img_names = [p.split('.')[0] for p in os.listdir(img_dir)]

        img_paths = [os.path.join(img_dir, p + '.jpg') for p in img_names]
        depth_paths = [os.path.join(depth_dir, p + '.png') for p in img_names]
        target_paths = [os.path.join(target_dir, p + '.npy') for p in img_names]  # npy 存储占空间很大? 可能格式错了

        print(f'{split}, img:', len(img_paths), 'depth:', len(depth_paths), 'target:', len(target_paths))

        check_list_path(img_paths)
        check_list_path(depth_paths)
        check_list_path(target_paths)

        write_list_to_txt(img_paths, txt_path=os.path.join(root, f'{split}_img_paths.txt'))
        write_list_to_txt(depth_paths, txt_path=os.path.join(root, f'{split}_depth_paths.txt'))
        write_list_to_txt(target_paths, txt_path=os.path.join(root, f'{split}_target_paths.txt'))


def generate_valid_txt(valid_num=1000):
    random.seed(100)

    valid_img_paths, valid_depth_paths, valid_target_paths = [], [], []

    for split in ['train', 'test']:
        img_paths = read_txt_as_list(os.path.join(root, f'{split}_img_paths.txt'))
        depth_paths = read_txt_as_list(os.path.join(root, f'{split}_depth_paths.txt'))
        target_paths = read_txt_as_list(os.path.join(root, f'{split}_target_paths.txt'))

        chose_idxs = random.sample(range(len(img_paths)), valid_num)

        chose_img_paths = [img_paths[i] for i in chose_idxs]
        chose_depth_paths = [depth_paths[i] for i in chose_idxs]
        chose_target_paths = [target_paths[i] for i in chose_idxs]

        valid_img_paths += chose_img_paths
        valid_depth_paths += chose_depth_paths
        valid_target_paths += chose_target_paths

    print('valid, img:', len(valid_img_paths), 'depth:', len(valid_depth_paths), 'target:', len(valid_target_paths))

    # 恐怖....
    # random.shuffle(valid_img_paths)
    # random.shuffle(valid_depth_paths)
    # random.shuffle(valid_target_paths)

    write_list_to_txt(valid_img_paths, os.path.join(root, 'valid_img_paths.txt'))
    write_list_to_txt(valid_depth_paths, os.path.join(root, 'valid_depth_paths.txt'))
    write_list_to_txt(valid_target_paths, os.path.join(root, 'valid_target_paths.txt'))


if __name__ == '__main__':
    # save_train_test_txt()
    generate_valid_txt()
