# no class weights
# rand 15min
# 设置 iters_per_epoch=200，每个 epoch 降至 5min  -> 20000 iters
# todo: 迭代总次数才是 影响训练效果的主要原因;
# bs=16, 15897MiB, 占太多，别的GPU也无法跑

# todo: 起始准确率差，因为没有 pretrain 参数
# inp=64 可以使用 pretrain 参数，所以一开始性能就好

# res18
# rand
python train.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 400 \
--gpu-ids 2 \
--loss-type mce --use-balanced-weights \
--lr 0.01 --lr-scheduler poly \
--context_path resnet18 \
--in_planes 32 \
--checkname res18_inp32_deconv

# res50
# cit
#
python train.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 400 \
--gpu-ids 0 \
--loss-type mce --use-balanced-weights \
--lr 0.01 --lr-scheduler poly \
--context_path resnet50 \
--in_planes 16 \
--checkname res50_inp16_deconv

# res101
python train.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 300 \
--gpu-ids 0 \
--loss-type mce --use-balanced-weights \
--lr 0.007 --lr-scheduler poly \
--context_path resnet101 \
--in_planes 64 \
--checkname res101_inp64_deconv