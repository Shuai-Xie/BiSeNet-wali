# pi 仅采用 pixel-wise loss 蒸馏训练
# 迭代 10000 iters
# 起始 lr 不要太大，不然 seg loss 会很高

# pi 用在 score map 分类结果
# pa 用在 feature map 比较特征 相似度

python train_kd.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 100 \
--gpu-ids 0 \
--loss-type mce --use-balanced-weights \
--pi \
--lr-g 1e-4 --lr-scheduler poly \
--checkname kd_pi

# rand
python train_kd.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 100 \
--gpu-ids 1 \
--loss-type mce --use-balanced-weights \
--pi \
--lr-g 5e-3 --lr-scheduler poly \
--checkname kd_pi_lr5e-3

# loss 在变，但是值太小
# pa=1000, pa loss 激增 4->40
# Acc_pixel: 0.2555, mIoU: 0.0246 第1次 valid 掉太多
# pa=10, 仍然带动 pa loss 和 seg loss 不断增加
python train_kd.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 100 \
--gpu-ids 1 \
--loss-type mce --use-balanced-weights \
--pa --lambda-pa 10. \
--lr-g 1e-3 --lr-scheduler poly \
--checkname kd_pa10

# 三种 loss 数量级有无影响
# pa=1000, pi=10, pa loss 激增 4->120, pi loss 激增 58 -> 200+
# Acc_pixel: 0.3349, mIoU: 0.0224
python train_kd.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 100 \
--gpu-ids 2 \
--loss-type mce --use-balanced-weights \
--pi --pa \
--lambda-pi 10. --lambda-pa 10. \
--lr-g 1e-3 --lr-scheduler poly \
--checkname kd_pi10_pa10