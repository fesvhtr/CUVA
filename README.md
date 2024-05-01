# Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly
The official repo for Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly [CVPR2024].  
**This repository is still under maintenance. Please wait for the final version.**
## Introduction
## Get Start
```
git clone https://github.com/fesvhtr/CUVA.git
conda create -n cuva python=3.8
pip install -r requirements.txt
conda activate cuva
```
## CUVA Benchmark
### Dataset
Coming soon
### Inference with Video-ChatGPT + A-Guardian
```
export PYTHONPATH="./:$PYTHONPATH"
cd /CUVA/Video-ChatGPT/video_chatgpt/CUVA
python inference_CUVA.py \
        --model-name /home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1
        --gpu_id 0 \
        --mod 0
```
### Classic Evaluation
Refer to repo [QA-Eval](https://github.com/fesvhtr/QA-Eval.git)
```
git clone https://github.com/fesvhtr/QA-Eval
python eval.py
```
### Evaluation with MMEval 
```
export PYTHONPATH="./:$PYTHONPATH"
cd /CUVA/Video-ChatGPT/video_chatgpt/CUVA
python inference_CUVA.py \
        --model-name /home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1
        --gpu_id 0 \
        --mod 0
```
## Cite
If you find our work useful for your research, please consider citing:
```
Release with CVPR2024
```
## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>  
CUVA is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).