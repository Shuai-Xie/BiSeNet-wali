import os
import time
import numpy as np
import torch
import random
import sys
import pickle
import json
import constants


def load_state_dict(model, ckpt_path, device):
    # 默认在 cuda:0 就容易报错
    state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    print('load', ckpt_path)


def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    print(string + ': Number of params: %.2f M' % (sum(b) / 1e6))


def generate_target_error_mask(pred, target, class_aware=False, num_classes=0):
    """
    :param pred: H,W
    :param target: H,W
    :param class_aware:
    :param num_classes: use with class_aware
    :return:
    """

    if isinstance(target, torch.Tensor):
        pred, target = to_numpy(pred), to_numpy(target)
    target_error_mask = (pred != target).astype('uint8')  # 0,1
    target_error_mask[target == constants.BG_INDEX] = 0

    if class_aware:
        # 不受类别数量影响
        error_mask = target_error_mask == 1
        target_error_mask[~error_mask] = constants.BG_INDEX  # bg

        for c in range(num_classes):  # C
            cls_error = error_mask & (target == c)
            target_error_mask[cls_error] = c

    return target_error_mask


def to_numpy(var, toint=False):
    #  Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    if isinstance(var, torch.Tensor):
        var = var.squeeze().detach().cpu().numpy()
    if toint:
        var = var.astype('uint8')
    return var


# pickle io
def dump_pickle(data, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        print('write data to', out_path)


def load_pickle(in_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)  # list
        return data


# json io
def dump_json(adict, out_path):
    with open(out_path, 'w', encoding='UTF-8') as json_file:
        # 设置缩进，格式化多行保存; ascii False 保存中文
        json_str = json.dumps(adict, indent=2, ensure_ascii=False)
        json_file.write(json_str)


def load_json(in_path):
    with open(in_path, 'rb') as f:
        adict = json.load(f)
        return adict


# io: txt <-> list
def write_list_to_txt(a_list, txt_path):
    with open(txt_path, 'w') as f:
        for p in a_list:
            f.write(p + '\n')


def read_txt_as_list(f):
    with open(f, 'r') as f:
        return [p.replace('\n', '') for p in f.readlines()]


def approx_print(arr):
    arr = np.around(arr * 100, decimals=2)
    print(','.join(map(str, arr)))


def recover_color_img(img):
    """
    cvt tensor image to RGB [note: not BGR]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()

    img = np.transpose(img, axes=[1, 2, 0])  # h,w,c
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)  # 直接通道相成?
    img = (img * 255).astype('uint8')
    return img


def colormap(N=256, normalized=False):
    """
    return color
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def mkdir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# dropout
def turn_on_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.train()


def turn_off_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.eval()


# topk
def get_topk_idxs(a, k):
    if isinstance(a, list):
        a = np.array(a)
    return a.argsort()[::-1][:k]


def get_group_topk_idxs(scores, groups=5, select_num=10):
    total_num = len(scores)
    base = total_num // groups
    remain = total_num % groups
    per_select = select_num // groups
    if remain > groups / 2:
        base += 1
        per_select += 1  # 多组多选
    last_select = select_num - per_select * (groups - 1)

    begin_idxs = [0] + [base * (i + 1) for i in range(groups - 1)] + [total_num]
    total_idxs = list(range(total_num))
    random.shuffle(total_idxs)

    select_idxs = []
    for i in range(groups):
        begin, end = begin_idxs[i], begin_idxs[i + 1]
        group_rand_idxs = total_idxs[begin:end]
        group_scores = [scores[s] for s in group_rand_idxs]
        if i == groups - 1:  # 最后一组
            per_select = last_select
        group_select_idxs = get_topk_idxs(group_scores, k=per_select).tolist()
        group_select_idxs = [group_rand_idxs[s] for s in group_select_idxs]  # 转成全局 idx

        select_idxs += group_select_idxs

    return select_idxs


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time


def max_normalize_to1(a):
    return a / (np.max(a) + 1e-12)


def minmax_normalize(a):  # min/max -> [0,1]
    min_a, max_a = np.min(a), np.max(a)
    return (a - min_a) / (max_a - min_a)


def cvt_mask_to_score(mask, pixel_scores):  # len(pixel_scores) = num_classes
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    valid = mask != constants.BG_INDEX
    class_cnts = np.bincount(mask[valid], minlength=len(pixel_scores))  # 0-5
    diver_score = np.sum(pixel_scores * class_cnts) / class_cnts.sum()
    return diver_score


class Logger:
    """logger"""

    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='UTF-8')  # 打开时自动清空文件

    def write(self, msg):
        self.terminal.write(msg)  # 命令行打印
        self.log.write(msg)

    def flush(self):  # 必有，不然 AttributeError: 'Logger' object has no attribute 'flush'
        pass

    def close(self):
        self.log.close()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccCaches:
    """acc cache queue"""

    def __init__(self, patience):
        self.accs = []  # [(epoch, acc), ...]
        self.patience = patience

    def reset(self):
        self.accs = []

    def add(self, epoch, acc):
        if len(self.accs) >= self.patience:  # 先满足 =
            self.accs = self.accs[1:]  # 队头出队列
        self.accs.append((epoch, acc))  # 队尾添加

    def full(self):
        return len(self.accs) == self.patience

    def max_cache_acc(self):
        max_id = int(np.argmax([t[1] for t in self.accs]))  # t[1]=acc
        max_epoch, max_acc = self.accs[max_id]
        return max_epoch, max_acc
