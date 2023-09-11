#!/bin/bash
max_length=512
output_dir=checkpoint/$(date +%Y_%m_%d_%H_%M_%S)
dataset=dataset

CUDA_LAUNCH_BLOCKING=1 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    general_train_seqcls.py \
        --output_dir ${output_dir} \
        --train_data_path ${dataset}/train.txt \
        --save_inputs ${dataset}/${max_length}_inputs.txt \
        --max_length ${max_length} \
        --num_train_epochs 5 \
        --batch_size 40 \
        --eval_step 100 \
        --DDP 


python predict_bert.py \
    --dataset ${dataset} \
    --load_path ${output_dir} \
    --max_length ${max_length}
