# pi 仅采用 pixel-wise loss 蒸馏训练
# 迭代 10000 iters
# 起始 lr 不要太大，不然 seg loss 会很高
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

python train_kd.py --dataset SUNRGBD --base-size 512 --crop-size 512 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 --iters_per_epoch 100 \
--gpu-ids 2 \
--loss-type mce --use-balanced-weights \
--pi --pa \
--lr-g 1e-4 --lr-scheduler poly \
--checkname kd_pi+pa