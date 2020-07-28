import argparse
import random


def parse_args(params=None):
    parser = argparse.ArgumentParser(description="BiSeNet")

    # model
    parser.add_argument('--context_path', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--in_planes', type=int, default=64,
                        help='resnet in planes (default: 64)')
    parser.add_argument('--dataset', type=str, default='SUNRGBD',
                        help='dataset name (default: SUNRGBD)')
    # default size
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')

    parser.add_argument('--sync-bn', type=bool, default=False,  # multi gpu
                        help='whether to use sync bn (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        help='loss func type (default: ce)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # gpu
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--iters_per_epoch', type=int, default=None,
                        help='iterations per epoch')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: True)')

    # distill params
    parser.add_argument("--pi", action='store_true', default=False, help="is pixel wise loss using or not")
    parser.add_argument("--pa", action='store_true', default=False, help="is pair wise loss using or not")
    parser.add_argument("--ho", action='store_true', default=False, help="is holistic loss using or not")
    parser.add_argument("--lr-g", type=float, default=1e-2, help="learning rate for G")
    parser.add_argument("--lr-d", type=float, default=4e-4, help="learning rate for D")
    parser.add_argument("--lambda-gp", type=float, default=10.0, help="lambda_gp")
    parser.add_argument("--lambda-d", type=float, default=0.1, help="lambda_d")
    parser.add_argument("--lambda-pi", type=float, default=10.0, help="lambda_pi")
    parser.add_argument('--lambda-pa', default=1.0, type=float, help='')

    # optimizer params
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,  # todo: 1e-4 测试集似乎效果不好
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default='False',
                        help='whether use nesterov (default: False)')

    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['step', 'poly', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr-step', type=str, default='35', help='step size for lr-step-scheduler')

    # seed
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    # checking point
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether to resume training')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval (default: 5) - record metrics every Nth iteration')

    # todo: active
    # parser.add_argument('--active-selection-mode', type=str, default='random',
    #                     choices=['random',
    #                              'entropy',
    #                              'error_mask',
    #                              'dropout',
    #                              'coreset'])
    # parser.add_argument('--max-iterations', type=int, default=9,
    #                     help='max active iterations')
    # parser.add_argument('--init-percent', type=int, default=None,
    #                     help='init label data percent')
    # parser.add_argument('--percent-step', type=int,
    #                     help='incremental label data percent (default: 5)')
    # parser.add_argument('--select-num', type=int,
    #                     help='incremental label data percent')
    # parser.add_argument('--hard-levels', type=int, default=9,
    #                     help='incremental label data percent')
    # parser.add_argument('--strategy', type=str, default='diff_score',
    #                     choices=['diff_score', 'diff_entropy'],
    #                     help='error mask strategy')

    args = parser.parse_args(params)

    # manual seeding
    # if args.seed == -1:
    #     args.seed = int(random.random() * 2000)
    # print('Using random seed =', args.seed)
    # print('ActiveSelector:', args.active_selection_mode)

    return args
