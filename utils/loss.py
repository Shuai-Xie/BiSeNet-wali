import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


def focal_loss_sigmoid(y_pred, labels, alpha=0.25, gamma=2):
    """
    :param y_pred: binary classification, output after sigmoid
    :param labels: gt label
    :param alpha: 负例样本权值，值越小，分给正例的权值越大
    :param gamma:
    :return:
    """
    labels = labels.float()

    # loss = label1 + label0
    # 难易样本占比，1- y_pred 约束, y_pred 越高，越容易
    loss = -labels * (1 - alpha) * ((1 - y_pred) ** gamma) * torch.log(y_pred) - \
           (1 - labels) * alpha * (y_pred ** gamma) * torch.log(1 - y_pred)

    return loss


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W) 将 C 通道带出
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))  # 可以 >=4d
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def onehot(tensor, num_class):
    B, H, W = tensor.shape
    y = torch.zeros(num_class, B, H, W)
    if tensor.is_cuda:
        y = y.cuda()

    for i in range(num_class):
        y[i][tensor == i] = 1  # 自动过滤掉 bg_idx
    return y.permute(1, 0, 2, 3)  # B,C,H,W


class SegmentationLosses:

    def __init__(self, mode='ce', weight=None, batch_average=False, ignore_index=constants.BG_INDEX, cuda=True):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

        self.losses = {
            'ce': self.CELoss,
            'bce': self.BCELoss,
            'focal': self.FocalLoss,
            'dice': self.DiceLoss,
            'mce': self.MultiOutput_CELoss,
        }

        if mode not in self.losses:
            raise NotImplementedError

        self.loss_fn = self.losses[mode]

    def __call__(self, output, target):
        return self.loss_fn(output, target)

    def MultiOutput_CELoss(self, outputs, target):
        _, h, w = target.shape
        if not isinstance(outputs, list):
            outputs = [outputs]

        loss = 0.
        for out in outputs:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            loss += self.CELoss(out, target)

        return loss

    def CELoss(self, output, target):
        """
        @param output: [B,C,H,W]    模型的输出；不需要 softmax, CELoss 内部会完成
        @param target: [B,H,W]
        """
        return F.cross_entropy(output, target.long(),
                               weight=self.weight, ignore_index=self.ignore_index, reduction='mean')

    def BCELoss(self, output, target):
        """
         @param output: [B,C,H,W]    模型的输出；不需要 softmax, binary_cross_entropy_with_logits 会完成
         @param target: [B,H,W]
         """
        if len(target.shape) == 3:
            target = onehot(target, num_class=output.size(1))

        loss = F.binary_cross_entropy_with_logits(output, target,
                                                  weight=self.weight, reduction='mean')
        return loss

    def FocalLoss(self, output, target, gamma=2, alpha=None):
        """
        @param output: [B,C,H,W]    模型的输出；不需要 softmax, CELoss 内部会完成
        @param target: [B,H,W]
        @param gamma: hard-easy regulatory factor 调节难易样本的抑制
        @param alpha: class imbalance regulatory factor 定义正样本的权值, CE 只用了正样本
        """
        logpt = -F.cross_entropy(output, target.long(),  # log(pt^) = -CE(pt)
                                 weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(logpt)  # loss -> pt, loss=0, pt=1

        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt  # element-wise

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        # OHEM, _ = loss.topk(k=int(OHEM_percent * [*loss.shape][0]))
        loss = loss.mean()

        return loss

    def DiceLoss(self, output, target):
        """
        @param output: [B,C,H,W]    模型输出
        @param target: [B,C,H,W]    one-hot label
        """
        if len(target.shape) == 3:
            target = onehot(target, num_class=output.size(1))
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)  # 转成 probs
        output, target = flatten(output), flatten(target)  # C,N

        # element-wise 视角: dice = p/(1+p), 连续化处理，使得 loss 可导
        # sum(-1) 把同类 pixel 分子/分母 分别加和了
        inter = (output * target).sum(-1)  # C, 乘积 target =1 取出了 gt class 的 p
        union = (output + target).sum(-1)
        dice = inter / union  # C, 每一类的 iou, 应该由很多 0 项
        dice = torch.mean(dice)  # mIoU

        return 1. - dice


class OHEM_CrossEntroy_Loss(nn.Module):
    def __init__(self, threshold, keep_num):
        super(OHEM_CrossEntroy_Loss, self).__init__()
        self.threshold = threshold
        self.keep_num = keep_num
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_function(output, target).view(-1)
        loss, loss_index = torch.sort(loss, descending=True)
        threshold_in_keep_num = loss[self.keep_num]
        if threshold_in_keep_num > self.threshold:
            loss = loss[loss > self.threshold]
        else:
            loss = loss[:self.keep_num]  # 保存部分 hard example 训练
        return torch.mean(loss)


class PixelWise_Loss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super().__init__()  # teacher 相当于生成 soft label, 依然可以用 weight
        self.logsoftmax = nn.LogSoftmax(dim=0)
        # todo: 可为 distill loss 加上 weight

    def forward(self, preds_S, preds_T):
        """
        predict + middle supervison predicts
        :param preds_S: [res, res1, res2]
        :param preds_T: [res, res1, res2]
        :return:
        """
        losses = 0.
        for s, t in zip(preds_S, preds_T):  # B,C,[1/8, 1/16, 1/16]
            t = t.detach()  # teacher infer 结果 detach()
            B, C, H, W = t.shape

            # -p(x) * log(q(x))
            softmax_t = F.softmax(flatten(t), dim=0)  # p(x), flatten return C,B*H*W
            logsoftmax_s = F.log_softmax(flatten(s), dim=0)  # log(q(x))

            loss = torch.sum(-softmax_t * logsoftmax_s) / H / W / B  # KL diver of each pixel
            losses += loss

        return losses


class PairWise_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = sim_dis_compute  # cos similarity distance

    def forward(self, feats_S, feats_T):
        """
        context path middle features
        :param feats_S: [cx1, cx2]
        :param feats_T: [cx1, cx2]
        :return:
        """
        losses = 0.
        for s, t in zip(feats_S, feats_T):  # B,C,1/16
            t = t.detach()  # context path feature
            B, C, H, W = t.shape
            patch_h, patch_w = H // 2, W // 2  # max_pool 到 2x2 计算
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h),
                                   padding=0, ceil_mode=True)
            loss = self.criterion(maxpool(s), maxpool(t))
            losses += loss
        return losses


def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


if __name__ == "__main__":
    a = torch.rand(1, 10, 32, 32)
    b = torch.rand(1, 10, 32, 32)
    c = torch.rand(1, 10, 64, 64)

    # b = torch.ones(1, 2, 2)
    #
    # print(SegmentationLosses(mode='ce', cuda=False)(a, b))
    # print(SegmentationLosses(mode='bce', cuda=False)(a, b))
    # print(SegmentationLosses(mode='focal', cuda=False)(a, b))
    # print(SegmentationLosses(mode='dice', cuda=False)(a, b))
