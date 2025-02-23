#!/bin/bash

torchrun --nproc_per_node=2 --master_port=12345 /dataNAS/people/sostm/GREEN/inference/run_get_green_analysis_on_rexval.py \
    --model_name /dataNAS/people/sostm/checkpoints/green_models/radphi2_green_v3_justin/checkpoint-3040