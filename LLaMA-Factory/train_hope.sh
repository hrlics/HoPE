#!/bin/bash
set -x -e
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export $WORLD_SIZE=2
export $GPU_NUM=8

export WANDB_DISABLED=true
export full_batch_size=128
export batch_size=1
export gradient_accumulation_steps=$(($full_batch_size/($batch_size*$GPU_NUM*$WORLD_SIZE)))
export CPUS_PER_TASK=20
export DECORD_EOF_RETRY_MAX=20480

export JOB_NAME=hope
export MODEL_SIZE=7B

export output_dir=cache/Qwen2-VL_${JOB_NAME}_${MODEL_SIZE}
export model_name_or_path=cache/Qwen2-VL-${MODEL_SIZE}-Instruct-with-Qwen2

torchrun \
    --nnodes $WORLD_SIZE  \
    --nproc_per_node $GPU_NUM  \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --model_name_or_path $model_name_or_path \
    --stage sft \
    --total_pixels 6272000 \
    --video_maxlen 128 \
    --do_train true \
    --finetuning_type full \
    --dataset llava_video_178k_filtered \
    --template qwen2_vl \
    --cutoff_len 8200 \
    --overwrite_cache true \
    --tokenized_path cache/pre_tokenized \
    --preprocessing_num_workers 128 \
    --output_dir $output_dir \
    --num_train_epochs 1.0 \
    --logging_steps 1 \
    --save_steps 400 \
    --save_total_limit 7 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 2.0e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --which_rope ${JOB_NAME} \
    --report_to none

