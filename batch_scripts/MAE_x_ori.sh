#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

PYTHON=/home/chang_yuance/.conda/envs/tface/bin/python
MAIN=/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py
ROOT=/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_ori

for FOLD in  1 2 3 4 5
do
    echo "===================================="
    echo " Running Fold ${FOLD} on GPU 6"
    echo "===================================="

    $PYTHON $MAIN \
        --root $ROOT \
        --num_classes 2 \
        --crop_size 224 \
        --batch_size 8 \
        --instance_length 4 \
        --fold $FOLD \
        --fusionmodel MAE_only_x \
        --savename M3DFEL_MAE_x_ori_fold_${FOLD} \
        --epochs 100 \

    echo " Fold ${FOLD} finished."
done