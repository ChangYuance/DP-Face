#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

PYTHON=/home/chang_yuance/.conda/envs/tface/bin/python
MAIN=/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py
ROOT=/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val

for FOLD in 1 2 3 4 5
do
    echo "===================================="
    echo " Running Fold ${FOLD} on GPU 6"
    echo "===================================="

    $PYTHON $MAIN \
        --root $ROOT \
        --num_classes 2 \
        --crop_size 112 \
        --batch_size 32 \
        --instance_length 4 \
        --fold $FOLD \
        --savename M3DFEL_lr_3e-4_fold_${FOLD} \
        --lr 3e-4

    echo " Fold ${FOLD} finished."
done
# cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && CUDA_VISIBLE_DEVICES=6 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
# --root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val --num_classes 2 --crop_size 112 --batch_size 8 \
# --instance_length 4  --fold 1 --savename M3DFEL_3e4_fold_1 --lr 3e-4