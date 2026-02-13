#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

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
        --crop_size 224 \
        --batch_size 32 \
        --instance_length 1 \
        --fold $FOLD \
        --num_frames 1 \
        --fusionmodel FP_VGGFace \
        --savename M3DFEL_FP_VGGFace_fold_${FOLD} \

    echo " Fold ${FOLD} finished."
done