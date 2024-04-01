#!/bin/bash


# adjust depending on the number of the nodes

BASE_MODEL=""
LORA_PATH=""
LORA_ROOT_PATH=""
DATA_PATH=""
OUTPUT_PATH=""
OUTPUT_NAME=""
USE_LOCAL=1 # 1: use local model, 0: use huggingface model
PART=0
mkdir -p $OUTPUT_PATH

if [[ USE_LOCAL -eq 1 ]] && [[ LORA_ROOT_PATH != "None" ]]
then
cp $LORA_ROOT_PATH/adapter_config.json $LORA_PATH
fi

export LAUNCHER="python"

export CMD=" \
    `pwd`/generate2.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH  \
    --data_path $DATA_PATH  \
    --output_path $OUTPUT_PATH  \
    --use_local $USE_LOCAL \
    --part $PART
    --output_name $OUTPUT_NAME \
    "

echo $CMD

export CUDA_VISIBLE_DEVICES=1

# eval $LAUNCHER $CMD &> output.llama_2_7B_hf.gsm_prolog_permutation_12_trial_3_reinstructed_fixed_noval.generation.txt
eval $LAUNCHER $CMD
