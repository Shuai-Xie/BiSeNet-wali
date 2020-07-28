import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
import constants


def get_label_name_colors(csv_path):
    """
    read csv_file and save as label names and colors list
    :param csv_path: csv color file path
    :return: lable name list, label color list
    """
    label_names, label_colors = [], []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i > 0:  # 跳过第一行
                label_names.append(row[0])
                label_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return label_names, label_colors


def color_code_target(target, label_colors):
    return np.array(label_colors)[target.astype('int')]


def get_legends(class_set, label_names, label_colors):
    legend_names, legend_lines = [], []
    for i in class_set:
        legend_names.append(label_names[i])  # 图例
        legend_lines.append(Line2D([0], [0], color=map_color(label_colors[i]), lw=2))  # 颜色线
    return legend_names, legend_lines


def map_color(rgb):
    return [v / 255 for v in rgb]


def plt_img_target_pred(img, target, pred, label_colors, vertial=False):
    # target_class_set = set(target.astype('int').flatten().tolist())
    # pred_class_set = set(pred.astype('int').flatten().tolist())
    # target_leg_names, target_leg_lines = get_legends(target_class_set, label_names, label_colors)
    # pred_leg_names, pred_leg_lines = get_legends(pred_class_set, label_names, label_colors)

    if vertial:
        f, axs = plt.subplots(nrows=3, ncols=1)
        f.set_size_inches((4, 9))
    else:
        f, axs = plt.subplots(nrows=1, ncols=3, dpi=150)
        f.set_size_inches((10, 3))

    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]

    # ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    # ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    # ax3.axis('off')
    ax3.imshow(color_code_target(pred, label_colors))
    ax3.set_title('predict')

    plt.show()


def plt_img_target_pred_error(img, target, pred, error_mask, label_colors, title=None):
    f, axs = plt.subplots(nrows=1, ncols=4, dpi=150)
    f.set_size_inches((18, 5))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    # ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    # ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    # ax3.axis('off')
    ax3.imshow(color_code_target(pred, label_colors))
    ax3.set_title('predict')

    ax4.imshow(color_code_target(error_mask, label_colors))
    ax4.set_title('error')

    if title:
        plt.title(title)  # 做到最后一个图

    plt.show()


def plt_img_target(img, target, label_colors, title=None):
    f, axs = plt.subplots(nrows=1, ncols=2, dpi=100)
    f.set_size_inches((7, 4))
    ax1, ax2 = axs.flat[0], axs.flat[1]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_img_target_ceal(img, target, ceal, label_colors):
    f, axs = plt.subplots(nrows=1, ncols=3)
    f.set_size_inches((10, 3))
    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(ceal, label_colors))
    ax3.set_title('ceal')

    plt.show()


def plt_color_label(target, label_colors, title):
    plt.figure()
    plt.axis('off')
    plt.imshow(color_code_target(target, label_colors))
    plt.title(title)
    plt.show()


def plt_img_target_gt_ceal(img, target, gt, ceal, label_colors):
    f, axs = plt.subplots(nrows=2, ncols=2)
    f.set_size_inches((8, 6))  # 800, 600
    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(gt, label_colors))
    ax3.set_title('gt')

    ax4.axis('off')
    ax4.imshow(color_code_target(ceal, label_colors))
    ax4.set_title('ceal')

    plt.show()


def get_plt_img_target_gt_ceal(img, target, gt, ceal, label_colors, figsize=(8, 6), title=None):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(figsize)
    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(gt, label_colors))
    ax3.set_title('gt')

    ax4.axis('off')
    ax4.imshow(color_code_target(ceal, label_colors))
    ax4.set_title('ceal')

    if title:
        plt.suptitle(title)

    # cvt plt result to np img
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((h, w, 3))  # 转成 img 实际大小
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.cla()
    plt.close("all")

    return img


import torch
from utils.misc import to_numpy


def save_error_mask(error_mask, save_path):
    if isinstance(error_mask, torch.Tensor):
        error_mask = to_numpy(error_mask)
    plt.axis('off')
    plt.imshow(error_mask, cmap='jet')  # crop pad，所以不能按照 error_mask 设置 fig 大小
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def plt_smooth_error_mask(image, target, label_colors,
                          target_error_mask, smooth_error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=2)
    f.set_size_inches((8, 6))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    # semantic
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('image')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    # mask
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(target_error_mask, cmap='gray')
    ax3.set_title('target_error_mask')

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(smooth_error_mask, cmap='gray')
    ax4.set_title('smooth_error_mask')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_smooth_error_mask_v2(image, target, predict, label_colors,
                             target_error_mask, smooth_error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=3)
    f.set_size_inches((10, 6))

    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3], axs.flat[4], axs.flat[5]

    # semantic
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('image')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(predict, label_colors))
    ax3.set_title('predict')

    # mask
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(target_error_mask, cmap='gray')
    ax4.set_title('target_error_mask')

    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.imshow(smooth_error_mask, cmap='gray')
    ax5.set_title('smooth_error_mask')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_target_pred_masks(target, pred, label_colors,
                          target_error_mask, pred_error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=2)
    f.set_size_inches((8, 6))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(pred, label_colors))
    ax3.set_title('predict')

    # mask
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(color_code_target(target_error_mask, label_colors))
    ax2.set_title('target_error_mask')

    ax4.axis('off')
    ax4.imshow(pred_error_mask, cmap='jet')
    ax4.set_title('pred_error_mask')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_class_superpixels(targets, label_names, label_colors, title=None, save_path=None):
    C = targets.shape[0]

    # plt.subplots_adjust(wspace=0, hspace=0)

    rows, cols = 3, 4
    f, axs = plt.subplots(nrows=rows, ncols=cols)
    f.set_size_inches((10, 6))

    for i in range(rows * cols):
        ax = axs.flat[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i < C:
            ax.imshow(color_code_target(targets[i], label_colors))
            ax.set_title(label_names[i], fontsize=10)
        else:
            ax.axis('off')

    if title:
        plt.suptitle(title)

    f.tight_layout()  # 调整整体空白

    if save_path:
        plt.savefig(save_path)

    # plt.show()


def plt_class_errors(errors, label_names, title=None, save_path=None):
    # 做出 backbone C features 或 target C features

    C = errors.shape[0]

    # plt.subplots_adjust(wspace=0, hspace=0)

    rows, cols = 3, 4
    f, axs = plt.subplots(nrows=rows, ncols=cols)
    f.set_size_inches((10, 6))

    for i in range(rows * cols):
        ax = axs.flat[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i < C:
            ax.imshow(errors[i], cmap='jet')
            ax.set_title(label_names[i], fontsize=10)
        else:
            ax.axis('off')

    if title:
        plt.suptitle(title)

    f.tight_layout()  # 调整整体空白

    if save_path:
        plt.savefig(save_path)

    # plt.show()


def plt_superpixel_scores(targets, all_sps, label_names, label_colors, title=None, save_path=None):
    C = targets.shape[0]

    rows, cols = 3, 4
    f, axs = plt.subplots(nrows=rows, ncols=cols)
    f.set_size_inches((20, 13))

    for i in range(rows * cols):
        ax = axs.flat[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i < C:
            ax.imshow(color_code_target(targets[i], label_colors))  # 从1开始，直接取最大
            for sps in all_sps[i]['sps']:
                ax.annotate('{:.3f}'.format(sps['iou']),
                            xy=(sps['centroid'][1], sps['centroid'][0]), fontsize=8,
                            xycoords='data', xytext=(2, -10), textcoords='offset points',
                            fontweight='bold',
                            bbox=dict(boxstyle='round, pad=0.3',  # linewidth=0 可以不显示边框
                                      alpha=0.5,
                                      facecolor=[c / 255 for c in [255, 255, 255]], lw=0),
                            color='b')
            ax.set_title(label_names[i] + f'({np.max(targets[i])})')
        else:
            ax.axis('off')

    if title:
        plt.suptitle(title)

    f.tight_layout()  # 调整整体空白

    if save_path:
        plt.savefig(save_path)

    # plt.show()


def plt_cmp_top_errors(targets, predicts, errors, label_names, label_colors, top_idxs,
                       title=None, save_path=None):
    rows, cols = 3, len(top_idxs)
    f, axs = plt.subplots(nrows=rows, ncols=cols)
    f.set_size_inches((10, 6))

    for i in range(rows):
        for j in range(cols):
            ax = axs.flat[cols * i + j]
            ax.set_xticks([])
            ax.set_yticks([])
            idx = top_idxs[j]
            if i == 0:
                ax.imshow(color_code_target(targets[idx], label_colors))
                ax.set_title(label_names[idx], fontsize=10)
            if i == 1:
                ax.imshow(color_code_target(predicts[idx], label_colors))
                ax.set_title(label_names[idx], fontsize=10)
            if i == 2:  # error score
                ax.imshow(errors[idx], cmap='jet')
                ax.set_title(label_names[idx], fontsize=10)

    if title:
        plt.suptitle(title)

    f.tight_layout()  # 调整整体空白

    if save_path:
        plt.savefig(save_path)

    plt.show()


from utils.misc import minmax_normalize


def plt_backbone_features(features, label_names):
    # 做出 backbone C features 或 target C features

    C = features.shape[0]

    rows, cols = 3, 4
    f, axs = plt.subplots(nrows=rows, ncols=cols)
    f.set_size_inches((10, 6))

    for i in range(rows * cols):
        ax = axs.flat[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i < C:
            ax.imshow(minmax_normalize(features[i]), cmap='jet')
            ax.set_title(label_names[i])
        else:
            ax.axis('off')

    plt.show()


def plt_all(target, pred, label_colors,
            target_error_mask, pred_error_mask, right_mask, error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=3)
    f.set_size_inches((10, 5))

    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3], axs.flat[4], axs.flat[5]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    ax4.axis('off')
    ax4.imshow(color_code_target(pred, label_colors))
    ax4.set_title('predict')

    # mask
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(color_code_target(target_error_mask, label_colors))
    ax2.set_title('target_error_mask')

    ax3.axis('off')
    ax3.imshow(pred_error_mask, cmap='jet')
    ax3.set_title('pred_error_mask')

    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.imshow(right_mask, cmap='gray')
    ax5.set_title('right_mask')

    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.imshow(error_mask, cmap='gray')
    ax6.set_title('error_mask')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_att(target, pred, label_colors, atten,
            target_error_mask, pred_error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=3)
    f.set_size_inches((10, 6))

    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]
    ax4, ax5, ax6 = axs.flat[3], axs.flat[4], axs.flat[5]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    # mask
    ax2.axis('off')
    ax2.imshow(color_code_target(target_error_mask, label_colors))
    ax2.set_title('target_error_mask')

    # att
    if atten is not None:
        ax3.axis('off')
        ax3.imshow(atten, cmap='jet')
        ax3.set_title('attention')

    # predict
    ax4.axis('off')
    ax4.imshow(color_code_target(pred, label_colors))
    ax4.set_title('predict')

    # error mask
    ax5.axis('off')
    ax5.imshow(pred_error_mask, cmap='jet')
    ax5.set_title('pred_error_mask')

    ax6.axis('off')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_all_atten(target, pred, label_colors, atten,
                  target_error_mask, pred_error_mask, right_mask, error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=4)
    f.set_size_inches((12, 6))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]
    ax5, ax6, ax7, ax8 = axs.flat[4], axs.flat[5], axs.flat[6], axs.flat[7]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    # mask
    ax2.axis('off')
    ax2.imshow(color_code_target(target_error_mask, label_colors))
    ax2.set_title('target_error_mask')

    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # ax2.imshow(target_error_mask, cmap='gray')
    # ax2.set_title('target_error_mask')

    ax3.axis('off')
    ax3.imshow(pred_error_mask, cmap='jet')
    ax3.set_title('pred_error_mask')

    ax4.axis('off')
    ax4.imshow(atten, cmap='jet')
    ax4.set_title('attention')

    ax5.axis('off')
    ax5.imshow(color_code_target(pred, label_colors))
    ax5.set_title('predict')

    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.imshow(right_mask, cmap='gray')
    ax6.set_title('right_mask')

    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.imshow(error_mask, cmap='gray')
    ax7.set_title('error_mask')

    ax8.axis('off')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_all_v2(target, pred, label_colors,
               target_error_mask, pred_error_mask,
               thre_right_mask, thre_error_mask,
               right_mask, error_mask,
               title=None, save_path=None):
    f, axs = plt.subplots(nrows=2, ncols=4)
    f.set_size_inches((12, 6))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]
    ax5, ax6, ax7, ax8 = axs.flat[4], axs.flat[5], axs.flat[6], axs.flat[7]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    ax5.axis('off')
    ax5.imshow(color_code_target(pred, label_colors))
    ax5.set_title('predict')

    # mask
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(target_error_mask, cmap='gray')
    ax2.set_title('target_error_mask')

    ax6.axis('off')
    ax6.imshow(pred_error_mask, cmap='jet')
    ax6.set_title('pred_error_mask')

    # thre
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(thre_right_mask, cmap='gray')
    ax3.set_title('thre_right_mask')

    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.imshow(thre_error_mask, cmap='gray')
    ax7.set_title('thre_error_mask')

    # 0.5
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(right_mask, cmap='gray')
    ax4.set_title('right_mask')

    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.imshow(error_mask, cmap='gray')
    ax8.set_title('error_mask')

    if title:
        plt.suptitle(title)

    f.tight_layout()  # 调整整体空白

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plt_error_mask(error_mask, save_path=None):
    plt.figure(figsize=(4, 3), dpi=200)

    plt.axis('off')
    plt.imshow(error_mask, cmap='jet')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_img_target_error(img, target, error_mask, label_colors, save_path=None, title=None):
    f, axs = plt.subplots(nrows=1, ncols=3, dpi=100)
    f.set_size_inches((8, 3))

    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]
    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    # ax2.axis('off')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    # ax3.axis('off')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(error_mask, cmap='gray')
    ax3.set_title('crop score')

    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.02)  # 调整子图间距(inch)，存储时能看到调节了间距

    if title:
        plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_cmp(img, lc, ms, entro, dropout, error_mask, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=6, dpi=200)
    f.set_size_inches((20, 3))

    maps = [img, lc, ms, entro, dropout, error_mask]
    titles = ['Image', 'LC', 'MS', 'Entropy', 'Dropout', 'Ours']

    ax = axs.flat[0]
    ax.axis('off')
    ax.imshow(maps[0])
    # ax.set_title(titles[0])

    for i in range(1, 6):
        ax = axs.flat[i]
        ax.axis('off')
        if maps[i] is not None:
            ax.imshow(maps[i], cmap='jet')
        # ax.set_title(titles[i])

    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.04)  # 调整子图间距(inch)，存储时能看到调节了间距

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_cmp_v2(img, lc, ms, en, dr, pred_error_mask, save_path=None):
    f, axs = plt.subplots(nrows=2, ncols=5, dpi=200)
    f.set_size_inches((16, 5))

    # img
    ax = axs.flat[0]
    ax.axis('off')
    ax.imshow(img)

    # norm uncer map
    lc, ms, en = minmax_normalize(lc), minmax_normalize(ms), minmax_normalize(en)
    dr = minmax_normalize(dr)

    maps = [lc, ms, en, dr]

    for i in range(len(maps)):
        ax = axs.flat[i + 1]
        ax.axis('off')
        ax.imshow(maps[i], cmap='jet')

    # pred error mask
    ax = axs.flat[5]
    ax.axis('off')
    ax.imshow(pred_error_mask, cmap='jet')

    # semantic attention uncer_map, and normlize
    for i in range(len(maps)):
        ax = axs.flat[i + 6]
        ax.axis('off')
        att_uncer_map = minmax_normalize(maps[i] * pred_error_mask)
        # att_uncer_map = maps[i] * pred_error_mask
        ax.imshow(att_uncer_map, cmap='jet')

    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.04)  # 调整子图间距(inch)，存储时能看到调节了间距

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_compare(target, pred, label_colors,
                target_error_mask, pred_error_mask, lc, ms, en, mc_droput=None):
    # 不同方法得到的 uncertain map 对比
    f, axs = plt.subplots(nrows=2, ncols=4)
    f.set_size_inches((12, 6))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]
    ax5, ax6, ax7, ax8 = axs.flat[4], axs.flat[5], axs.flat[6], axs.flat[7]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    ax5.axis('off')
    ax5.imshow(color_code_target(pred, label_colors))
    ax5.set_title('predict')

    # mask
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(target_error_mask, cmap='gray')
    ax2.set_title('target_error_mask')

    ax6.axis('off')
    ax6.imshow(pred_error_mask, cmap='jet')
    ax6.set_title('pred_error_mask')

    # compare
    ax3.axis('off')
    ax3.imshow(lc, cmap='jet')
    ax3.set_title('least confidence')

    ax4.axis('off')
    ax4.imshow(ms, cmap='jet')
    ax4.set_title('margin sampling')

    ax7.axis('off')
    ax7.imshow(en, cmap='jet')
    ax7.set_title('entropy')

    ax8.axis('off')
    ax8.set_title('mc droput')

    plt.show()
