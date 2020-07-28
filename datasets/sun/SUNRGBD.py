import os
import numpy as np
from torch.utils.data import Dataset
from utils.misc import read_txt_as_list
from utils.vis import get_label_name_colors
import datasets.transforms as tr
from PIL import Image
import constants

"""
SUN-RGBD
10335 = 5285(train) + 5050(test)
train 8000, valid 3000, test 2335
"""
this_dir = os.path.dirname(__file__)


class SUNRGBD(Dataset):

    def __init__(self, root, split, base_size=None, crop_size=None):
        super().__init__()
        self.img_paths = read_txt_as_list(os.path.join(root, f'{split}_img_paths.txt'))
        # self.depth_paths = read_txt_as_list(os.path.join(root, f'{split}_depth_paths.txt'))
        self.target_paths = read_txt_as_list(os.path.join(root, f'{split}_target_paths.txt'))

        # debug 可以用 iters_per_epoch debug 了
        # self.img_paths, self.target_paths = self.img_paths[:100], self.target_paths[:100]

        self.base_size = base_size  # train 基准 size
        self.crop_size = crop_size  # train, valid, test

        self.transform = self.get_transform(split)

        self.bg_idx = 0
        self.num_classes = 37
        self.mapbg_fn = tr.mapbg(self.bg_idx)
        self.remap_fn = tr.remap(self.bg_idx)

        self.label_names, self.label_colors = get_label_name_colors(os.path.join(this_dir, 'sun37.csv'))

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        # depth = Image.open(self.depth_paths(index))
        target = np.load(self.target_paths[index]).astype(int)
        # target = self.reduce_class14(target)
        target = self.mapbg_fn(target)
        target = Image.fromarray(target)

        sample = {
            'img': img,
            # 'depth': depth,
            'target': target
        }
        if self.transform:  # 只要设置 transform 为 None, 就能方便地得到原始图片
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_paths)

    def get_transform(self, split):
        if split == 'train':
            return tr.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=self.base_size,
                                   crop_size=self.crop_size,
                                   scales=(0.8, 2.0),
                                   fill=constants.BG_INDEX),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        elif split == 'valid':
            return tr.Compose([
                tr.FixScaleCrop(crop_size=self.crop_size),  # valid, 固定长宽比 crop
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        elif split == 'test':
            return tr.Compose([
                tr.FixedResize(size=self.crop_size),  # test, 直接 resize 到 crop size
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else:
            return None

    def reduce_class14(self, label):
        # 19
        abort_classes = [4, 11, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37]

        # reduce classes 37 -> 14
        # 0 abort 19 classes to background
        for idx in abort_classes:
            label[np.where(label == idx)] = 0

        # whether use big class
        # ======================
        # # 7 table(counter, desk) 12,14
        label[np.where(label == 12)] = 7
        label[np.where(label == 14)] = 7
        # 10 bookshelf(shelves) 15
        label[np.where(label == 15)] = 10
        # 13 blinds(curtain) 16
        label[np.where(label == 16)] = 13
        # total: 19 + 4 + 14 = 37
        # ======================
        # use desk, not use table, counter
        # for idx in [7, 12]:
        #     label[np.where(label == idx)] = 0

        # 0,1,2,3 no change
        label[np.where(label == 5)] = 4
        label[np.where(label == 6)] = 5
        label[np.where(label == 7)] = 6
        label[np.where(label == 8)] = 7
        label[np.where(label == 9)] = 8
        label[np.where(label == 10)] = 9
        label[np.where(label == 13)] = 10
        label[np.where(label == 22)] = 11
        label[np.where(label == 25)] = 12
        label[np.where(label == 29)] = 13
        label[np.where(label == 31)] = 14

        return label
