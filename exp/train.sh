#!/bin/bash

pwd_dir=$pwd
cd ../

eval "$(conda shell.bash hook)"
conda activate usst

GPU_ID=$1
NUM_WORKERS=$2
TAG=$3

# --validate
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --num_workers ${NUM_WORKERS} \
    --config config/${TAG}.yml \
    --tag ${TAG}

cd $pwd_dir
echo "Training finished!"