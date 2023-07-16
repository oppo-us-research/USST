#!/bin/bash

pwd_dir=$pwd
cd ../

eval "$(conda shell.bash hook)"
conda activate usst

GPU_ID=$1
NUM_WORKERS=$2
TAG=$3
PHASE=$4

if [ $PHASE == 'train' ]
then
    CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_h2o.py \
        --num_workers ${NUM_WORKERS} \
        --config config/h2o/${TAG}.yml \
        --tag ${TAG}
elif [ $PHASE == 'eval' ]
then
    CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_h2o.py \
        --num_workers ${NUM_WORKERS} \
        --config config/h2o/${TAG}.yml \
        --tag ${TAG} \
        --eval
fi

cd $pwd_dir
echo "${PHASE} finished!"