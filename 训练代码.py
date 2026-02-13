通道注意力融合训练代码
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
CUDA_VISIBLE_DEVICES=6 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs --num_classes 2 --crop_size 112 --batch_size 8 \
--instance_length 4 --AUs True --fold 1 --fusionmodel M3DFEL_AUs_att --savename Pretrain_fold_1 \
--resume /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/outputs/M3DFEL_fold_2-[01-1]6-[12:06]/model_best.pth

MEEI数据集训练代码_concat (只有--AUs True)
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
CUDA_VISIBLE_DEVICES=2 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0115_MEEI_val_AUs \
--num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --AUs True --fold 1 --savename M3DFEL_MEEI_1

MEEI数据集训练代码
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
CUDA_VISIBLE_DEVICES=6 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0115_MEEI_val_AUs \
--num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --fold 1 --savename M3DFEL_MEEI_ori_1

--fusionmodel M3DFEL_AUs_att

通道相加训练代码
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
CUDA_VISIBLE_DEVICES=4 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs \
--num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --AUs True --fold 1 --fusionmodel M3DFEL_AUs_add --savename M3DFEL_AUs_add_fold_1

原始图像训练代码
CUDA_VISIBLE_DEVICES=3 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_ori --num_classes 2 --crop_size 112 --batch_size 8 \
--instance_length 4 --fold 1 --savename M3DFEL_ori_fold_1 --epoch 100



单独使用AU训练代码
CUDA_VISIBLE_DEVICES=6 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs \
--num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --AUs True --fold 2 --fusionmodel M3DFEL_AUs_only --savename M3DFEL_AUs_only_fold_2

注意力的预训练
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && CUDA_VISIBLE_DEVICES=6 \
/home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs --num_classes 2 --crop_size 112 \
--batch_size 8 --instance_length 4 --fold 3  --savename Pretrain_fold_3 \
--resume /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/outputs/M3DFEL_MEEI_2-[01-17]-[14:48]/model_best.pth

FiLM融合训练代码
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
CUDA_VISIBLE_DEVICES=7 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs \
--num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --AUs True --fold 1 --fusionmodel M3DFEL_AUs_FiLM2 --savename M3DFEL_AUs_FiLM2_1

FiLM融合训练代码
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
CUDA_VISIBLE_DEVICES=7 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs \
--num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --AUs True --fold 4 --fusionmodel M3DFEL_AUs_FiLM --savename M3DFEL_FiLM_fold_4

新增防止过拟合的代码
我的图像_dropout训练代码
dropout
CUDA_VISIBLE_DEVICES=6 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val --num_classes 2 --crop_size 112 --batch_size 8 \
--instance_length 4 --fusionmodel M3DFEL_dropout --fold 1 --savename M3DFEL_dropout_fold_1

调整learningrate
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && CUDA_VISIBLE_DEVICES=6 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val --num_classes 2 --crop_size 112 --batch_size 8 \
--instance_length 4  --fold 1 --savename M3DFEL_3e4_fold_1 --lr 3e-4

原始代码
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && CUDA_VISIBLE_DEVICES=7 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val --num_classes 2 --crop_size 112 --batch_size 8 \
--instance_length 4  --fold 1 --savename M3DFEL_fold_1

focal loss
cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && CUDA_VISIBLE_DEVICES=7 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
--root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val --num_classes 2 --crop_size 112 --batch_size 8 \
--instance_length 4  --focal_loss True --fold 1 --savename M3DFEL_focal_loss_fold_1
