#!/bin/bash

pwd_dir=$pwd
cd ../

source activate usst

GPU_ID=$1
TAG=$2

CFG_FILE="$(cut -d'[' -f1 <<<${TAG})".yml  # remove [] part

# --validate
CUDA_VISIBLE_DEVICES=${GPU_ID} python demo.py \
    --config config/${CFG_FILE} \
    --tag ${TAG}

cd $pwd_dir
echo "Demo finished!"