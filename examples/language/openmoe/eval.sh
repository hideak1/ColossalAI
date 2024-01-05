#!/bin/bash

set -xue

NUM_GPU=1
MODEL="test"
SEQ_LENGTH=2048
BATCH_SIZE=1
LR=0.00001

# ep zero
torchrun --standalone --nproc_per_node $NUM_GPU eval.py \
    --num_epoch 1 \
    --model_name $MODEL \
    --plugin "ep_zero" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --zero_stage 0 \
    --dp_size 1

# ep
# torchrun --standalone --nproc_per_node $NUM_GPU train.py \
#     --num_epoch 1 \
#     --model_name $MODEL \
#     --plugin "ep_zero" \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --zero_stage 1

# hybrid
# torchrun --standalone --nproc_per_node $NUM_GPU train.py \
#     --num_epoch 1 \
#     --model_name $MODEL \
#     --plugin "hybrid" \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --zero_stage 1 \
#     --pp_size 2 \
#     --dp_size 1 \
#     --ep_size 2 \
