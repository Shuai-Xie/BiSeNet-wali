import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))  # 添加项目目录，python2 不支持中文

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.resnet import build_contextpath


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))  # CBR


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))  # CBR


class Spatial_path(nn.Module):
    def __init__(self, out_channels):  # 1/2 chans
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=out_channels // 4)
        self.conv2 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 2)
        self.conv3 = ConvBlock(in_channels=out_channels // 2, out_channels=out_channels)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


# ARM, channel attention
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # in = out
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.bn(self.conv1x1(self.gap(x))))  # channel attention
        out = torch.mul(x, att)  # 没有将 input 加入
        return out


import skimage.transform
import numpy as np


def cpu_resize(x, target_size):
    assert isinstance(x, torch.Tensor), 'x must be tensor'

    B, C, _, _ = x.size()
    y = np.zeros((B, target_size[0], target_size[1], C))
    x = x.detach().cpu().permute((0, 2, 3, 1)).numpy()  # B,H,W,C

    for i in range(B):
        y[i] = skimage.transform.resize(x[i], target_size, order=1,  # bilinear
                                        anti_aliasing=True, preserve_range=True)  # 保持原始值范围

    y = torch.from_numpy(y).permute((0, 3, 1, 2)).float().cuda()

    return y


# FFM
# todo: feature fusion 采用 self-attention, 直达 loss
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, num_classes):  # out_channels = num_classes
        super().__init__()
        self.cbr = ConvBlock(in_channels, num_classes, stride=1)  # todo: 特征压缩太少了?

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.conv1x1_1 = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU()
        self.conv1x1_2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sp, x1, x2):
        # _, _, h, w = sp.shape
        # x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)  # 最快
        # x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
        # x1 = cpu_resize(x1, (h, w))  # 耗时
        # x2 = cpu_resize(x2, (h, w))

        x = torch.cat((sp, x1, x2), dim=1)  # fusion feature
        x = self.cbr(x)  # chans -> num_classes
        att = self.sigmoid(self.conv1x1_2(self.relu(self.conv1x1_1(self.gap(x)))))
        out = x + torch.mul(x, att)
        return out


class BiSeNet(nn.Module):
    def __init__(self, num_classes, context_path, in_planes=64):
        super().__init__()

        # sp_chans = max(in_planes * 2, 64)  # 最低要有 64 chan
        sp_chans = 128  # 维持为 128 尝试
        self.saptial_path = Spatial_path(sp_chans)
        self.context_path = build_contextpath(context_path, in_planes, pretrained=True)

        if context_path == 'resnet18':
            arm_chans = [in_planes * 4, in_planes * 8]
            ffm_chans = sum(arm_chans) + sp_chans
        elif context_path == 'resnet50' or context_path == 'resnet101':
            arm_chans = [in_planes * 4 * 4, in_planes * 8 * 4]  # expansion=4
            ffm_chans = sum(arm_chans) + sp_chans
        else:
            raise NotImplementedError

        # middle features after attention
        self.arm1 = AttentionRefinementModule(arm_chans[0], arm_chans[0])
        self.arm2 = AttentionRefinementModule(arm_chans[1], arm_chans[1])

        # deconv for ffm
        self.deconv1 = DeconvBlock(arm_chans[0], arm_chans[0], kernel_size=4, stride=2, padding=1)  # x2
        self.deconv2 = DeconvBlock(arm_chans[1], arm_chans[1], kernel_size=4, stride=2, padding=1)  # pad 减小 input size

        # middle supervision
        self.mid1 = nn.Conv2d(arm_chans[0], num_classes, kernel_size=1)
        self.mid2 = nn.Conv2d(arm_chans[1], num_classes, kernel_size=1)

        self.ffm = FeatureFusionModule(ffm_chans, num_classes)
        self.last_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        sp = self.saptial_path(x)
        cx1, cx2 = self.context_path(x)
        cx1, cx2 = self.arm1(cx1), self.arm2(cx2)  # gap 已经在 arm 中做了，没必要再乘 tail

        # deconv 上采样
        cx1 = self.deconv1(cx1)  # tx2, torch 不支持 upsample
        cx2 = self.deconv2(cx2)

        res = self.last_conv(self.ffm(sp, cx1, cx2))  # 1/8

        # for onnx output or infer mode
        # return res

        # for teacher/student training
        res1, res2 = self.mid1(cx1), self.mid2(cx2)  # 1/16, 1/16
        return [res, res1, res2, cx1, cx2]

        # 单模型可用 self.training 判断状态
        # if self.training:  # 使用 nn.Module 自带属性判断 training/eval 状态
        #     res1, res2 = self.mid1(cx1), self.mid2(cx2)  # 1/16, 1/16
        #     return [res, res1, res2]  # 1/8, 1/16, 1/16
        # else:
        #     return res


@torch.no_grad()
def cmp_infer_time(test_num=20):
    import time
    import itertools

    # 首个 resnet50 预热 GPU
    archs = ['resnet50', 'resnet18', 'resnet50', 'resnet101']
    inplanes = [16, 32, 64]

    x = torch.rand(1, 3, 512, 512)
    x = x.cuda()

    for arch, inp in itertools.product(archs, inplanes):  # 笛卡儿积
        model = BiSeNet(37, context_path=arch, in_planes=inp)
        model.cuda()
        model.eval()

        torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成, CPU 等待 cuda 所有运算执行完才退出
        t1 = time.time()
        for _ in range(test_num):
            model(x)
        t2 = time.time()
        torch.cuda.synchronize()

        t = (t2 - t1) / test_num
        fps = 1 / t

        # print(f'{arch} - {inp} \t time: {t} \t fps: {fps}')
        print('{} - {} \t time: {} \t fps: {}'.format(arch, inp, t, fps))


if __name__ == '__main__':
    # model = BiSeNet(num_classes=37, context_path='resnet50', in_planes=16)
    # model.eval()
    # model.cuda()
    # x = torch.rand(2, 3, 512, 512).cuda()
    # res = model(x)
    # print(type(res))
    # for r in res:
    #     print(r.size())

    cmp_infer_time()
