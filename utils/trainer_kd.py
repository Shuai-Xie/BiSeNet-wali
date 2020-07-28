import torch
import constants
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.misc import get_learning_rate
from utils.loss import SegmentationLosses, PairWise_Loss, PixelWise_Loss
from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import Evaluator
from utils.misc import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from tqdm import tqdm
import torch.nn.functional as F


class Trainer:

    def __init__(self, args, student, teacher, train_set, val_set, test_set, class_weights, saver, writer):
        self.args = args
        self.saver = saver
        self.saver.save_experiment_config()  # save cfgs
        self.writer = writer

        self.num_classes = train_set.num_classes

        # dataloaders
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        self.val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        self.test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        self.dataset_size = {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)}
        print('dataset size:', self.dataset_size)

        # 加快训练，减少每轮迭代次数；不需要从引入样本时就截断数据，这样更好
        self.iters_per_epoch = args.iters_per_epoch if args.iters_per_epoch else len(self.train_dataloader)

        self.device = torch.device(f'cuda:{args.gpu_ids}')

        # todo: 二者可以考虑在 2个 device 上?
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device).eval()  # 用来生成训练 target

        # student is generator
        self.G_optimizer = torch.optim.SGD([{
            'params': filter(lambda p: p.requires_grad, self.student.parameters()),
            'initial_lr': args.lr_g
        }], args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
        self.G_lr_scheduler = LR_Scheduler(mode=args.lr_scheduler, base_lr=args.lr_g,
                                           num_epochs=args.epochs, iters_per_epoch=self.iters_per_epoch)

        # todo: discriminator
        # self.D_solver = optim.SGD([{
        #     'params': filter(lambda p: p.requires_grad, D_model.parameters()),
        #     'initial_lr': args.lr_d
        # }], args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        # loss
        if args.use_balanced_weights:
            weight = torch.from_numpy(class_weights.astype(np.float32)).to(self.device)
        else:
            weight = None

        # 原有 loss
        self.criterion = SegmentationLosses(mode=args.loss_type, weight=weight, ignore_index=constants.BG_INDEX)
        self.criterion_pi = PixelWise_Loss(weight=weight, ignore_index=constants.BG_INDEX)
        self.criterion_pa = PairWise_Loss()

        # evaluator
        self.evaluator = Evaluator(self.num_classes)

        self.best_epoch = 0
        self.best_mIoU = 0.0
        self.best_pixelAcc = 0.0

    def training(self, epoch, prefix='Train', evaluation=False):
        self.student.train()
        if evaluation:
            self.evaluator.reset()

        train_losses = AverageMeter()
        segment_losses = AverageMeter()
        pi_losses, pa_losses = AverageMeter(), AverageMeter()

        tbar = tqdm(self.train_dataloader, desc='\r', total=self.iters_per_epoch)  # 设置最多迭代次数, 从0开始..

        if self.writer:
            self.writer.add_scalar(f'{prefix}/learning_rate', get_learning_rate(self.G_optimizer), epoch)

        for i, sample in enumerate(tbar):
            image, target = sample['img'], sample['target']
            image, target = image.to(self.device), target.to(self.device)

            # adjust lr
            self.G_lr_scheduler(self.G_optimizer, i, epoch)

            # forward
            with torch.no_grad():
                preds_T = self.teacher(image)  # [res, res1, res2, cx1, cx2]
            preds_S = self.student(image)

            # 分割 loss
            G_loss = self.criterion(preds_S[:3], target)  # multiple output loss
            segment_losses.update(G_loss.item())

            # 蒸馏 loss
            if self.args.pi:  # pixel wise loss
                loss = self.args.lambda_pi * self.criterion_pi(preds_S[:3], preds_T[:3])
                G_loss += loss
                pi_losses.update(loss.item())

            if self.args.pa:  # pairwise loss
                loss = self.args.lambda_pa * self.criterion_pa(preds_S[3:], preds_T[3:])
                G_loss += loss
                pa_losses.update(loss.item())

            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

            train_losses.update(G_loss.item())
            tbar.set_description('Epoch {}, Train loss: {:.3} = seg {:.3f} + pi {:.3f} + pa {:.3f}'.format(
                epoch, train_losses.avg, segment_losses.avg, pi_losses.avg, pa_losses.avg))

            if evaluation:
                output = F.interpolate(preds_S[0], size=(target.size(1), target.size(2)), mode='bilinear', align_corners=True)
                pred = torch.argmax(output, dim=1)
                self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W

            # 即便 tqdm 有 total，仍然要这样跳出
            if i == self.iters_per_epoch - 1:
                break

        if self.writer:
            self.writer.add_scalars(f'{prefix}/loss', {
                'train': train_losses.avg,
                'segment': segment_losses.avg,
                'pi': pi_losses.avg,
                'pa': pa_losses.avg
            }, epoch)
            if evaluation:
                Acc = self.evaluator.Pixel_Accuracy()
                mIoU = self.evaluator.Mean_Intersection_over_Union()
                print('Epoch: {}, Acc_pixel:{:.3f}, mIoU:{:.3f}'.format(epoch, Acc, mIoU))

                self.writer.add_scalars(f'{prefix}/IoU', {
                    'mIoU': mIoU,
                    # 'mDice': mDice,
                }, epoch)
                self.writer.add_scalars(f'{prefix}/Acc', {
                    'acc_pixel': Acc,
                    # 'acc_class': Acc_class
                }, epoch)

    @torch.no_grad()
    def validation(self, epoch, test=False):
        self.student.eval()
        self.evaluator.reset()  # reset confusion matrix

        if test:
            tbar = tqdm(self.test_dataloader, desc='\r')
            prefix = 'Test'
        else:
            tbar = tqdm(self.val_dataloader, desc='\r')
            prefix = 'Valid'

        # loss
        segment_losses = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['img'], sample['target']
            image, target = image.to(self.device), target.to(self.device)

            output = self.student(image)[0]  # 拿到首个输出
            segment_loss = self.criterion(output, target)
            segment_losses.update(segment_loss.item())
            tbar.set_description(f'{prefix} loss: %.4f' % segment_losses.avg)

            output = F.interpolate(output, size=(target.size()[1:]), mode='bilinear', align_corners=True)
            pred = torch.argmax(output, dim=1)  # pred

            # eval: add batch result
            self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W

        Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        # mDice = self.evaluator.Mean_Dice()
        print('Epoch: {}, Acc_pixel: {:.4f}, mIoU: {:.4f}'.format(epoch, Acc, mIoU))

        if self.writer:
            self.writer.add_scalar(f'{prefix}/loss', segment_losses.avg, epoch)
            self.writer.add_scalars(f'{prefix}/IoU', {
                'mIoU': mIoU,
                # 'mDice': mDice,
            }, epoch)
            self.writer.add_scalars(f'{prefix}/Acc', {
                'acc_pixel': Acc,
                # 'acc_class': Acc_class
            }, epoch)

        if not test:
            if mIoU > self.best_mIoU:
                print('saving model...')
                self.best_mIoU = mIoU
                self.best_pixelAcc = Acc
                self.best_epoch = epoch

                state = {
                    'epoch': self.best_epoch,
                    'state_dict': self.student.state_dict(),  # 方便 test 保持同样结构?
                    'optimizer': self.G_optimizer.state_dict(),
                    'best_mIoU': self.best_mIoU,
                    'best_pixelAcc': self.best_pixelAcc
                }
                self.saver.save_checkpoint(state)
                print('save model at epoch', epoch)

        return mIoU, Acc

    def load_best_checkpoint(self):
        checkpoint = self.saver.load_checkpoint()
        self.student.load_state_dict(checkpoint['state_dict'])
        # self.G_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'=> loaded checkpoint - epoch {checkpoint["epoch"]}')
        return checkpoint["epoch"]
