#!/bin/bash

GPUS_PER_NODE=8
NNODES=${SLURM_NNODES}
NODE_RANK=${SLURM_NODEID}
MASTER_ADDR=localhost
MASTER_PORT=6009

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export WANDB_ENTITY="brophie"
export WANDB_PROJECT="GREEN"

MODEL="StanfordAIMI/RadPhi-2"
TRAIN_DATA="/dataNAS/people/sostm/data/GREEN_V3/train.json"
VAL_DATA="/dataNAS/people/sostm/data/GREEN_V3/val.json"
output_dir=/dataNAS/people/sostm/checkpoints/green_models/radphi2_green_v3_justin

torchrun $DISTRIBUTED_ARGS training/finetune.py \
    --model_name_or_path $MODEL \
    --data_path "$TRAIN_DATA" \
    --eval_data_path "$VAL_DATA" \
    --fp16 False \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --report_to "wandb" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --group_by_length True \
    --dataloader_num_workers 4 \
    --deepspeed training/ds_config_zero3.json \
    # --resume_from_checkpoint ${output_dir}/checkpoint-1000 \