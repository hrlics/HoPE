MODEL_WEIGHT_CACHE=/cache/to/your/checkpoint
ROPE_TYPES=(hope vanilla_rope mrope videorope)
CHECKPOINT=specify_your_checkpoint_as_{num_steps}


for ROPE_TYPE in "${ROPE_TYPES[@]}"; do

    MODEL_NAME=Qwen2-VL-${ROPE_TYPE}_7B

    accelerate launch --num_processes 8 --config_file  easy_context/accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 \
    vision_niah/eval_vision_niah.py \
    --model  $MODEL_WEIGHT_CACHE/$MODEL_NAME/checkpoint-${CHECKPOINT} \
    --needle_embedding_dir ./vision_niah/data/needle_embeddings/needle_1-hour_3000-frames_144-tokens \
    --haystack_dir ./vision_niah/data/haystack_embeddings/haystack_1-hour_3000-frames_144-tokens \
    --needle_dataset ./vision_niah/needle_datasets/dataset.json \
    --output_path ./vision_niah/niah_output/$MODEL_NAME/checkpoint-${CHECKPOINT} \
    --prompt_template qwen2 \
    --max_frame_num 3000 \
    --min_frame_num  100\
    --frame_interval 200 \
    --rope_type $ROPE_TYPE \
    --image_tokens 144 \
    --depth_interval 0.2

done


