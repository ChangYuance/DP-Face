#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

PYTHON=/home/chang_yuance/.conda/envs/tface/bin/python
MAIN=/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py
ROOT=/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs

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
        --AUs True \
        --center_loss True \
        --fusionmodel M3DFEL_AUs_only_center_loss \
        --savename M3DFEL_only_AUs_fold_${FOLD} \

    echo " Fold ${FOLD} finished."
done

# cd /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL && \
# CUDA_VISIBLE_DEVICES=4 /home/chang_yuance/.conda/envs/tface/bin/python /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/main.py \
# --root /home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/0107dataset_val_AUs \
# --num_classes 2 --crop_size 112 --batch_size 8 --instance_length 4 --AUs True --fold 1 --fusionmodel M3DFEL_AUs_att --savename M3DFEL_AUs_att_fold_1