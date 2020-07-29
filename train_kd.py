import argument_parser
from pprint import pprint

args = argument_parser.parse_args()
pprint(vars(args))

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"

if len(args.gpu_ids) > 1:
    args.sync_bn = True

from torch.utils.tensorboard import SummaryWriter
from datasets.build_datasets import build_datasets

from model.bisenet import BiSeNet

from utils.saver import Saver
from utils.trainer_kd import Trainer
from utils.misc import AccCaches, get_curtime, print_model_parm_nums, load_state_dict
import numpy as np


def main():
    # dataset
    trainset, valset, testset = build_datasets(args.dataset, args.base_size, args.crop_size)

    # 定义 student/teacher 模型
    student = BiSeNet(trainset.num_classes, context_path='resnet18', in_planes=32)
    teacher = BiSeNet(trainset.num_classes, context_path='resnet101', in_planes=64)
    print_model_parm_nums(student, 'student')  # student: Number of params: 5.66 M
    print_model_parm_nums(teacher, 'teacher')  # teacher: Number of params: 132.92 M

    # 加载 student/teacher 已经训练好的模型
    device = f'cuda:{args.gpu_ids}'
    load_state_dict(student, 'runs/SUNRGBD/res18_inp32_deconv_Jul27_100319/checkpoint.pth.tar', device)
    load_state_dict(teacher, 'runs/SUNRGBD/res101_inp64_deconv_Jul26_205859/checkpoint.pth.tar', device)

    class_weights = None
    if args.use_balanced_weights:  # default false
        class_weights = np.array([  # med_freq
            0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
            0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
            2.589096, 0.980794, 0.920340, 0.667984, 1.172291,  # 15
            0.862240, 0.921714, 2.154782, 1.187832, 1.178115,  # 20
            1.848545, 1.428922, 2.849658, 0.771605, 1.656668,  # 25
            4.483506, 2.209922, 1.120280, 2.790182, 0.706519,  # 30
            3.994768, 2.220004, 0.972934, 1.481525, 5.342475,  # 35
            0.750738, 4.040773  # 37
        ])

    saver = Saver(args, timestamp=get_curtime())
    writer = SummaryWriter(saver.experiment_dir)

    trainer = Trainer(args, student, teacher,
                      trainset, valset, testset, class_weights, saver, writer)

    start_epoch = 0

    miou_caches = AccCaches(patience=5)  # miou
    for epoch in range(start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            miou, pixelAcc = trainer.validation(epoch)
            miou_caches.add(epoch, miou)
            if miou_caches.full():
                print('acc caches:', miou_caches.accs)
                print('best epoch:', trainer.best_epoch, 'best miou:', trainer.best_mIoU)
                _, max_miou = miou_caches.max_cache_acc()
                if max_miou < trainer.best_mIoU:
                    print('end training')
                    break

    print('valid')
    print('best mIoU:', trainer.best_mIoU, 'pixelAcc:', trainer.best_pixelAcc)

    # test
    epoch = trainer.load_best_checkpoint()
    test_mIoU, test_pixelAcc = trainer.validation(epoch, test=True)
    print('test')
    print('best mIoU:', test_mIoU, 'pixelAcc:', test_pixelAcc)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
