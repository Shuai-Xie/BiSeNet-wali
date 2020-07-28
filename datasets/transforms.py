import torch
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import constants
import PIL.ImageEnhance as ImageEnhance
from PIL import Image, ImageOps, ImageFilter


def mapbg(bg_idx):
    """
    image bg 转成 constants.BG_INDEX, 类别从 [0,..,C-1]
    """

    # bg 在首部，需要调整 实际类别 前移1位
    def map_headbg(target):
        target = target.astype(int)
        target -= 1  # 1->0
        target[target == -1] = constants.BG_INDEX  # 255
        return target.astype('uint8')

    # bg 在尾部，直接替换为 constant 即可
    def map_other(target):
        target = target.astype(int)
        target[target == bg_idx] = constants.BG_INDEX
        return target.astype('uint8')

    if bg_idx == 0:
        return map_headbg
    else:
        return map_other


def remap(bg_idx):
    """
    分割结果 -> 回归原始 bg idx，方面 vis
    """

    def remap_headbg(target):
        target = target.astype(int)
        target += 1
        target[target == constants.BG_INDEX + 1] = bg_idx
        return target.astype('uint8')

    def remap_other(target):
        target = target.astype(int)
        target[target == constants.BG_INDEX] = bg_idx
        return target.astype('uint8')

    if bg_idx == 0:
        return remap_headbg
    else:
        return remap_other


class Compose:  # 可以采用 默认的
    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, sample):
        for t in self.trans_list:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('

        for t in self.trans_list:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class RandomHorizontalFlip:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        return {'img': img,
                'target': target}


class RandomRotate:
    def __init__(self, degree):  # 旋角上限
        self.degree = degree

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        target = target.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'target': target}


class RandomGaussianBlur:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {'img': img,
                'target': target}


class RandomScaleCrop:
    def __init__(self, base_size, crop_size, scales=(0.75, 2.0), fill=0):  # fill bg_idx
        self.base_size = base_size  # 1个值，不用给定 h,w
        self.crop_size = crop_size
        self.scales = scales
        self.fill = fill

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        # 保持原图 aspect ratio，依照 短边 进行缩放
        short_size = random.randint(int(self.base_size * self.scales[0]),
                                    int(self.base_size * self.scales[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # random scale
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        # scale 后短边 < 要 crop 尺寸，补图
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # img fill 0, 后面还有 normalize
            target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=self.fill)  # target fill bg_idx

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'img': img,
                'target': target}


class FixScaleCrop:
    def __init__(self, crop_size):  # valid, 固定原图 aspect，crop 到 model size
        self.crop_size = crop_size

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)  # 保证长宽比，以短边为 513，放缩长边
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        w, h = img.size  # 放缩后的 size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'img': img,
                'target': target}


class FixedResize:
    def __init__(self, size):
        self.size = (size, size)  # test, 直接 resize 到 crop_size

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        assert img.size == target.size

        img = img.resize(self.size, Image.BILINEAR)
        target = target.resize(self.size, Image.NEAREST)

        return {'img': img,
                'target': target}


class ColorJitter:
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])

        img = Image.fromarray(img)  # np->Image [np uint8]
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)

        return {'img': img,
                'target': target}


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'img': img,
                'target': target}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        target = np.array(target).astype(np.float32)

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).float()

        return {'img': img,
                'target': target}


class ToRGBD:
    def __call__(self, sample):
        img, depth, target = sample['img'], sample['depth'], sample['target']
        rgbd = torch.cat((img, depth), 0)
        return {'rgbd': rgbd,
                'target': target}
