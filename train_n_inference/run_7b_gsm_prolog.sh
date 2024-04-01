#!/bin/bash

# adjust depending on the number of the nodes

wandb login ""
export WANDB_PROJECT=""

# XXX: edit me
GPUS_PER_NODE=2
NNODES=1

# Note that usually LoRA needs to use larger learning rate
DATA_PATH=""
MODEL_PATH=""
TEST_SIZE=100
MICRO_BATCH=4
TOTAL_BATCH=128
EVAL_STEPS=20
SAVE_STEPS=40
WARMUP_STEPS=20
EPOCHS=6
OUTPUT=""
RESUME_FROM_CHECKPOINT=""
mkdir -p $OUTPUT

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_port 31415 \
    "

export CMD=" \
    `pwd`/finetune1.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT \
    --micro_batch $MICRO_BATCH \
    --total_batch $TOTAL_BATCH \
    --eval_steps $EVAL_STEPS \
    --log_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --test_size $TEST_SIZE \
    --epochs $EPOCHS \
    --wandb \
    "

# # clear old checkpoint as it'd mismatch while we sort things out
#     rm -rf $SAVE_CHECKPOINT_PATH

echo $CMD

export CUDA_VISIBLE_DEVICES=2,3
# to debug - add echo (it exits and prints what it would have launched)
eval $LAUNCHER $CMD
