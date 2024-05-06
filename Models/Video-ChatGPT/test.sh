source ~/.bashrc
#python scripts/apply_delta.py \
#        --base-model-path /home/dh/zsc/Video-ChatGPT/llama-7b-hf \
#        --target-model-path LLaVA-Lightning-7B-v1-1 \
#        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
#
#python video_chatgpt/demo/video_demo.py \
#        --model-name /home/dh/zsc/Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
#        --projection_path /home/dh/zsc/Video-ChatGPT/video_chatgpt-7B.bin

export PYTHONPATH="./:$PYTHONPATH"

python /home/dh/pythonProject/AnomalyDataset/Video-ChatGPT/video_chatgpt/demo/cmd_demo.py \
        --model-name /home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
        --projection_path /home/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin

CUDA_VISIBLE_DEVICES=1 python /home/dh/pythonProject/AnomalyDataset/Video-ChatGPT/video_chatgpt/demo/test_aqa.py \
        --model-name /home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
        --projection_path /home/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin