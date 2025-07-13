pip3 install qwen_vl_utils
export DECORD_EOF_RETRY_MAX=20480

rope_types=("hope" "vanilla_rope" "m_rope" "videorope")
checkpoint=your/checkpoint
total_pixels_array=(6272000 12544000 25088000 50176000)
MODEL_SIZE=7B

for total_pixels in "${total_pixels_array[@]}"; do
    
    max_frames=$(( ($total_pixels / 6272000) * 56 ))

    for rope_type in "${rope_types[@]}"; do

        model_path="../cache/Qwen2-VL_${rope_type}_${MODEL_SIZE}/checkpoint-${checkpoint}"
        echo "Starting evaluation with ${rope_type}, checkpoint=${checkpoint}, total_pixels=${total_pixels}, max_frames=${max_frames}"
        
        accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
            --model qwen2_vl \
            --model_args=pretrained=$model_path,which_rope=$rope_type,use_flash_attention_2=True,total_pixels=$total_pixels,fps=2.0,max_frames=$max_frames \
            --tasks mlvu_dev,longvideobench_val_v,videomme \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix qwen2_vl \
            --output_path ./eval_logs/${rope_type}/${MODEL_SIZE}/${max_frames}
        
        echo "Completed evaluation with ${rope_type}, checkpoint=${checkpoint}, total_pixels=${total_pixels}, max_frames=${max_frames}"

    done
done

echo "All evaluations completed!"