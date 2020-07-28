import torch
import constants
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.misc import get_learning_rate
from utils.loss import SegmentationLosses
from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import Evaluator
from utils.misc import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from tqdm import tqdm
import torch.nn.functional as F


class Trainer:

    def __init__(self, args, model, train_set, val_set, test_set, class_weights, saver, writer):
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

        if args.optimizer == 'SGD':
            print('Using SGD')
            self.optimizer = torch.optim.SGD(model.parameters(),
                                             lr=args.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay,
                                             nesterov=args.nesterov)
            self.lr_scheduler = LR_Scheduler(mode=args.lr_scheduler, base_lr=args.lr,
                                             lr_step=args.lr_step,
                                             num_epochs=args.epochs,
                                             warmup_epochs=args.warmup_epochs,
                                             iters_per_epoch=self.iters_per_epoch)
        elif args.optimizer == 'Adam':
            print('Using Adam')
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=args.lr,
                                              # amsgrad=True,
                                              weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        self.device = torch.device(f'cuda:{args.gpu_ids}')

        if len(args.gpu_ids) > 1:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
            patch_replication_callback(model)
            print(args.gpu_ids)

        self.model = model.to(self.device)

        # loss
        if args.use_balanced_weights:
            weight = torch.from_numpy(class_weights.astype(np.float32)).to(self.device)
        else:
            weight = None

        self.criterion = SegmentationLosses(mode=args.loss_type, weight=weight, ignore_index=constants.BG_INDEX)

        # evaluator
        self.evaluator = Evaluator(self.num_classes)

        self.best_epoch = 0
        self.best_mIoU = 0.0
        self.best_pixelAcc = 0.0

    def training(self, epoch, prefix='Train', evaluation=False):
        self.model.train()
        if evaluation:
            self.evaluator.reset()

        train_losses = AverageMeter()
        tbar = tqdm(self.train_dataloader, desc='\r', total=self.iters_per_epoch)  # 设置最多迭代次数, 从0开始..

        if self.writer:
            self.writer.add_scalar(f'{prefix}/learning_rate', get_learning_rate(self.optimizer), epoch)

        for i, sample in enumerate(tbar):
            image, target = sample['img'], sample['target']
            image, target = image.to(self.device), target.to(self.device)
            if self.args.optimizer == 'SGD':
                self.lr_scheduler(self.optimizer, i, epoch)  # each iteration

            output = self.model(image)
            loss = self.criterion(output, target)  # multiple output loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item())
            tbar.set_description('Epoch {}, Train loss: {:.3f}'.format(epoch, train_losses.avg))

            if evaluation:
                output = F.interpolate(output[-1], size=(target.size(1), target.size(2)), mode='bilinear', align_corners=True)
                pred = torch.argmax(output, dim=1)
                self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W

            # 即便 tqdm 有 total，仍然要这样跳出
            if i == self.iters_per_epoch - 1:
                break

        if self.writer:
            self.writer.add_scalar(f'{prefix}/loss', train_losses.val, epoch)
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
        self.model.eval()
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

            output = self.model(image)[0]  # 拿到首个输出
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
        print('Epoch: {}, Acc_pixel:{:.4f}, mIoU:{:.4f}'.format(epoch, Acc, mIoU))

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
                    'state_dict': self.model.state_dict(),  # 方便 test 保持同样结构?
                    'optimizer': self.optimizer.state_dict(),
                    'best_mIoU': self.best_mIoU,
                    'best_pixelAcc': self.best_pixelAcc
                }
                self.saver.save_checkpoint(state)
                print('save model at epoch', epoch)

        return mIoU, Acc

    def load_best_checkpoint(self):
        checkpoint = self.saver.load_checkpoint()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'=> loaded checkpoint - epoch {checkpoint["epoch"]}')
        return checkpoint["epoch"]
