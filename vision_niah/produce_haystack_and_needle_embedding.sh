python produce_haystack_embedding.py \
--model Qwen/Qwen2-VL-7B-Instruct \
--video_path ./data/haystack_videos/movie.mp4 \
--output_dir ./data/haystack_embeddings/haystack_1-hour_3000-frames_144-tokens \
--sampled_frames_num 3000 \
--pooling_size 2 

python produce_needle_embedding.py \
--model Qwen/Qwen2-VL-7B-Instruct \
--output_dir ./data/needle_embeddings/needle_1-hour_3000-frames_144-tokens \
--needle_dataset ./needle_datasets/dataset.json