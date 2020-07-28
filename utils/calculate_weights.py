from tqdm import tqdm
import numpy as np
import copy


def cal_class_freqs(dataset, num_classes):
    class_cnts = np.zeros((num_classes,))

    dataset = copy.deepcopy(dataset)  # 替换 transforms，计算 class weights
    dataset.transform = None  # 不改变原图 target

    # 计算每个像素点数目
    for sample in tqdm(dataset):
        y = np.asarray(sample['target'])
        mask = np.logical_and((y >= 0), (y < num_classes))  # 逻辑或，合理区域
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)  # 19D vec
        class_cnts += count_l  # class cnt

    class_freqs = class_cnts / class_cnts.sum()
    return class_cnts, class_freqs


# freq 和 weight 不同
def cal_class_weights(dataset, num_classes, save_dir=None):
    """
    weight = 1/√num, 再归一化
    """
    class_cnts, class_freqs = cal_class_freqs(dataset, num_classes)
    z = np.nan_to_num(np.sqrt(1 + class_cnts))  # smooth num, 防止下文分母 frequency=0
    class_weights = [1 / f for f in z]  # frequency

    ret = np.nan_to_num(np.array(class_weights))
    ret[ret > 2 * np.median(ret)] = 2 * np.median(ret)
    ret = ret / ret.sum()
    print('Class weights: ')
    print(ret)

    if save_dir:
        np.save(f'{save_dir}/class_weights.npy', ret)
        np.save(f'{save_dir}/class_cnts.npy', class_cnts)
        np.save(f'{save_dir}/class_freqs.npy', class_freqs)

    return ret
