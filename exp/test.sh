#!/bin/bash

pwd_dir=$pwd
cd ../

source activate egopat3d

GPU_ID=$1
NUM_WORKERS=$2
TAG=$3

CFG_FILE="$(cut -d'[' -f1 <<<${TAG})".yml  # remove [] part

# --validate
CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py \
    --num_workers ${NUM_WORKERS} \
    --config config/${CFG_FILE} \
    --tag ${TAG}

cd $pwd_dir
echo "Testing finished!"