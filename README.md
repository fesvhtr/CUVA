# Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly
[![paper](https://img.shields.io/badge/cs.AI-2405.00181-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2405.00181)   <a href="https://huggingface.co/datasets/fesvhtr/CUVA"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/dataset-on-hf-md-dark.svg" alt="Dataset on hf">  

The official repo for Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly [CVPR2024].  
This repository is still under maintenance. The code for the partial ablation experiment on A-Guardian is still being organized.  
If you have any questions please contact [7597892@bupt.edu.cn]().

## Update  
Now CUVA dataset can be easily evaluated by [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval) by using the task name **cuva_test**.

## Introduction
We present a comprehensive benchmark for **Causation Understanding of Video Anomaly** (CUVA). 
We also introduce **MMEval**, a novel evaluation metric designed to better align with human preferences for CUVA.
Then we propose a novel **prompt-based method** that can serve as a baseline approach for the challenging CUVA.
## Get Start
```
git clone https://github.com/fesvhtr/CUVA.git
cd CUVA
conda create -n cuva python=3.8
conda activate cuva
pip install -r requirements.txt
```
## CUVA Benchmark
### CUVA Dataset
Please download the dataset from [hf](https://huggingface.co/datasets/fesvhtr/CUVA). There are 4 zip files and 1 json file in the dataset, unzip them and put them in the `data` folder.  

### Inference with Video-ChatGPT + A-Guardian
```
export PYTHONPATH="./:$PYTHONPATH"
cd /CUVA/Models/Video-ChatGPT/video_chatgpt/CUVA
./inference_CUVA.sh
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
cd /CUVA/Models/Video-ChatGPT/video_chatgpt/CUVA
./mmEval_demo.sh
```
### Multiple reasoning and evaluation
Modify and run CUVA.py and mmEval.py in the `CUVA` folder.
## Acknowledgement
Sincere thanks to Video-chatGPT, VideoChat, mPlug, Otter, VideoLLaMA, Univtg and others for their excellent work.
## Cite
If you find our work useful for your research, please consider citing:
```
@INPROCEEDINGS{CUVA,
  author={Du, Hang and Zhang, Sicheng and Xie, Binzhu and Nan, Guoshun and Zhang, Jiayang and Xu, Junrui and Liu, Hangyu and Leng, Sicong and Liu, Jiangming and Fan, Hehe and Huang, Dajiu and Feng, Jing and Chen, Linli and Zhang, Can and Li, Xuhuan and Zhang, Hao and Chen, Jianhang and Cui, Qimei and Tao, Xiaofeng},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Uncovering what, why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly}, 
  year={2024},
  volume={},
  number={},
  pages={18793-18803},
  keywords={Measurement;Annotations;Surveillance;Natural languages;Benchmark testing;Traffic control;Pattern recognition;Anomaly Video;Large Language Model},
  doi={10.1109/CVPR52733.2024.01778}}

```
## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>  
CUVA is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).
