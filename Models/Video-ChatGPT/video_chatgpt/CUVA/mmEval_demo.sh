python mmEval_demo.py \
        --model-name "path for LLaVA-7B-Lightening-v1-1" \
        --vision_tower_path "path for openai/clip-vit-large-patch14" \
        --projection_path "path for video_chatgpt-7B.bin" \
        --video_dir_path "path for video_dir" \
        --task "task" \
        --gpu_id 0 \
        --mod 0
# mod 1: with importance curve