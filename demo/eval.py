import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

import torch
from utils.metrics import Evaluator
from utils.misc import approx_print
import numpy as np
from tqdm import tqdm
from datasets.build_datasets import data_cfg, build_datasets
from model.bisenet import BiSeNet
from utils.misc import load_state_dict
import matplotlib.pyplot as plt
import torch.nn.functional as F


@torch.no_grad()
def evals(arch='res18'):
    """
    class IoU & mIoU, Acc & mAcc
    """
    trainset, valset, testset = build_datasets(dataset='SUNRGBD', base_size=512, crop_size=512)

    # load model
    if arch == 'res18':
        model = BiSeNet(37, context_path='resnet18', in_planes=32)
        load_state_dict(model, ckpt_path='runs/SUNRGBD/kd_pi_lr1e-3_Jul28_002404/checkpoint.pth.tar')
    elif arch == 'res101':
        model = BiSeNet(37, context_path='resnet101', in_planes=64)
        load_state_dict(model, ckpt_path='runs/SUNRGBD/res101_inp64_deconv_Jul26_205859/checkpoint.pth.tar')
    else:
        raise NotImplementedError

    model.eval()
    model.cuda()

    evaluator = Evaluator(testset.num_classes)
    evaluator.reset()

    print('imgs:', len(testset))
    for sample in tqdm(testset):  # already transfrom
        image, target = sample['img'], sample['target']
        image = image.unsqueeze(0).cuda()
        pred = model(image)
        pred = F.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        target = target.numpy()
        evaluator.add_batch(target, pred)

    print('PixelAcc:', evaluator.Pixel_Accuracy())

    print('mAcc')  # 各类的 acc 均值
    Accs = evaluator.Acc_of_each_class()
    print(np.nanmean(Accs))  # mAcc, mean of non-NaN elements
    approx_print(Accs)

    print('mIoU')
    IOUs = evaluator.IOU_of_each_class()
    print(np.nanmean(IOUs))  # mIoU
    approx_print(IOUs)


results = {
    'res18(pi)': {
        'acc': [80.85, 87.48, 56.45, 63.27, 78.97, 43.23, 55.04, 43.91, 67.55, 39.17, 59.6, 37.32, 34.79, 22.28, 11.5, 55.31, 45.36, 40.39, 47.57, 0.0, 29.44,
                81.86, 45.62, 28.09, 46.84, 30.37, 14.32, 0.0, 30.08, 62.07, 19.18, 22.79, 78.56, 59.2, 37.63, 52.47, 20.61],
        'iou': [69.11, 80.0, 37.21, 53.74, 51.76, 35.57, 40.81, 27.64, 41.49, 23.95, 38.55, 28.27, 25.08, 16.26, 6.74, 37.46, 32.96, 25.95, 32.54, 0.0, 13.41,
                58.53, 24.09, 22.6, 31.84, 17.78, 11.31, 0.0, 12.1, 46.05, 9.91, 13.28, 51.49, 39.55, 20.73, 38.68, 9.52],
        'Acc': 69.58,
        'mIoU': 30.43
    },
    'res101': {
        'acc': [86.04, 93.08, 69.26, 72.1, 85.25, 58.81, 66.68, 58.51, 72.58, 47.83, 74.31, 43.0, 41.78, 32.6, 20.34, 65.79, 59.12, 56.35, 58.64, 0.02, 45.64,
                82.94, 59.95, 52.79, 75.8, 50.62, 41.36, 1.61, 48.72, 68.63, 63.91, 31.98, 87.75, 70.76, 63.7, 69.97, 44.99
                ],
        'iou': [76.74, 87.19, 46.49, 65.72, 66.73, 49.55, 50.68, 41.15, 50.87, 35.88, 50.18, 34.82, 27.59, 23.69, 11.3, 49.93, 41.08, 36.79, 38.42, 0.01, 24.88,
                67.75, 35.37, 45.1, 53.87, 27.59, 30.64, 1.38, 27.39, 58.07, 45.7, 18.68, 68.92, 53.55, 34.65, 50.06, 20.75],
        'Acc': 77.59,
        'mIoU': 41.87
    }
}


def plt_class_evals(arch):
    label_names, label_colors = data_cfg['SUNRGBD']['label_name_colors']
    label_names, label_colors = label_names[1:], label_colors[1:]

    xs = np.arange(len(label_names))

    accs = results[arch]['acc']
    ious = results[arch]['iou']

    plt.figure(figsize=(14, 5), dpi=100)

    width = 0.4
    fontsize = 8
    rotation = 0

    for idx, (x, y) in enumerate(zip(xs, accs)):
        plt.bar(x - 0.2, y, width=width, align='center',  # 底部 tick 对应位置
                linewidth=1, edgecolor=[0.7, 0.7, 0.7],
                color=[a / 255.0 for a in label_colors[idx]])
        plt.text(x - 0.2, y + 0.2,
                 s='%.2f' % y,
                 rotation=rotation,
                 ha='center', va='bottom', fontsize=fontsize)

    for idx, (x, y) in enumerate(zip(xs, ious)):
        plt.bar(x + 0.2, y, width=width,
                linewidth=1, edgecolor=[0.7, 0.7, 0.7],
                color=[a / 255.0 for a in label_colors[idx]])
        plt.text(x + 0.2, y + 0.2,
                 s='%.2f' % y,
                 rotation=rotation,
                 ha='center', va='bottom', fontsize=fontsize)

    plt.xticks(xs, label_names, size='small', rotation=60)
    plt.ylim([0, 100])
    plt.title(f"{arch}(pi). Acc & IoU of SUNRGBD-37class testset (5050) | Acc: {results[arch]['Acc']}, mIoU: {results[arch]['mIoU']}")
    plt.show()


if __name__ == '__main__':
    arch = 'res18'
    # evals(arch)
    plt_class_evals(arch)
