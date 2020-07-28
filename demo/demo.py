import torch
import os
from datasets.build_datasets import data_cfg
from datasets.transforms import remap

from model.bisenet import BiSeNet
from utils.misc import load_state_dict, recover_color_img, mkdir
from utils.vis import color_code_target
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

img_size = (480, 640)

# trans
test_trans = transforms.Compose([
    transforms.Resize(img_size),  # test, 直接 resize 到 crop size
    transforms.ToTensor(),  # Normalize tensor image
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
remap_fn = remap(0)
label_names, label_colors = data_cfg['SUNRGBD']['label_name_colors']


@torch.no_grad()
def infer_img(img, vis=False):  # pillow read img
    img = test_trans(img).unsqueeze(0).cuda()
    pred = model(img)
    pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype('uint8')

    pred = remap_fn(pred)
    predict = color_code_target(pred, label_colors)

    if vis:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(recover_color_img(img))
        ax[0].set_title('img')
        ax[1].imshow(predict)
        ax[1].set_title('predict')
        plt.show()

    return predict


if __name__ == '__main__':
    # arch = 'res18'
    arch = 'res101'

    # load model
    if arch == 'res18':
        model = BiSeNet(37, context_path='resnet18', in_planes=32)
        load_state_dict(model, ckpt_path='runs/SUNRGBD/kd_pi_lr1e-3_Jul28_002404/checkpoint.pth.tar')
    elif arch == 'res101':
        model = BiSeNet(37, context_path='resnet101', in_planes=64)
        load_state_dict(model, ckpt_path='runs/SUNRGBD/res101_inp64_deconv_Jul26_205859/checkpoint.pth.tar')
    else:
        raise NotImplementedError
    model.eval().cuda()

    # infer dir
    exp = 'sun'
    img_dir = f'img/{exp}/rgb'
    save_dir = f'img/{exp}/seg_{arch}'
    mkdir(save_dir)

    for img in os.listdir(img_dir):
        print(img)
        img_name = img.split('.')[0]
        img = Image.open(f'{img_dir}/{img}').convert('RGB')
        predict = infer_img(img, vis=True)
        cv2.imwrite(f'{save_dir}/{img_name}.png', predict[:, :, ::-1])
