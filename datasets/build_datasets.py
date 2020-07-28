import os
from datasets import *
from utils.vis import get_label_name_colors

this_dir = os.path.dirname(__file__)

data_cfg = {
    'SUNRGBD': {
        'root': '/datasets/rgbd_dataset/SUNRGBD',
        'class': SUNRGBD,
        'label_name_colors': get_label_name_colors(os.path.join(this_dir, 'sun/sun37.csv'))
    }
}


def build_datasets(dataset, base_size, crop_size):
    if dataset not in data_cfg:
        raise NotImplementedError('no such dataset')

    config = data_cfg[dataset]
    cls = config['class']

    trainset = cls(config['root'], 'train', base_size, crop_size)
    validset = cls(config['root'], 'valid', base_size, crop_size)
    testset = cls(config['root'], 'test', base_size, crop_size)

    return trainset, validset, testset


if __name__ == '__main__':
    from utils.vis import plt_img_target
    from utils.misc import recover_color_img
    import numpy as np
    from utils.calculate_weights import cal_class_weights

    dataset = 'SUNRGBD'
    trainset, validset, testset = build_datasets(dataset, base_size=512, crop_size=512)
    # cal_class_weights(trainset, trainset.num_classes, save_dir='/datasets/rgbd_dataset/SUNRGBD/train')

    for idx, sample in enumerate(validset):
        img, target = sample['img'], sample['target']
        img, target = img.squeeze(0), target.squeeze(0)
        print(img.shape)

        img, target = img.numpy(), target.numpy()
        img = recover_color_img(img)
        print(np.unique(target))

        target = target.astype('uint8')
        target = trainset.remap_fn(target)

        plt_img_target(img, target, trainset.label_colors)

        if idx == 4:
            exit(0)
