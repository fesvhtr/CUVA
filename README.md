# Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly
[![paper](https://img.shields.io/badge/cs.AI-2405.00181-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2405.00181)   <a href="https://huggingface.co/datasets/fesvhtr/CUVA"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/dataset-on-hf-md-dark.svg" alt="Dataset on hf">  

The official repo for Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly [CVPR2024].  

## Updates  
- [2026.04] The data format of the CUVA dataset on Hugging Face has been restructured into a more user-friendly version.  
- [2025.11] We are working on finetuning some latest models to replace the out-of-date VideoChatGPT.
- [2025.11] https://github.com/Dulpy/ECVA can access the new version of CUVA. We also appreciate your citation of this paper.
- [2025.10] Now CUVA dataset can be easily evaluated by [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval) by using the task name **cuva_test**.

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
#### Load annotation from Hugging Face Hub

```python
from datasets import load_dataset

full_ds = load_dataset("fesvhtr/CUVA", split="full")
test_ds = load_dataset("fesvhtr/CUVA", split="test")
```

#### Download Video Archives

For the original video files, download the archives explicitly from the dataset repository.

Download one archive into the Hugging Face cache:

```python
from huggingface_hub import hf_hub_download

zip_path = hf_hub_download(
    repo_id="fesvhtr/CUVA",
    repo_type="dataset",
    filename="raw/group_0.zip",
)
print(zip_path)
```

Download the whole repository snapshot:

```python
from huggingface_hub import snapshot_download

repo_dir = snapshot_download(
    repo_id="fesvhtr/CUVA",
    repo_type="dataset",
)
print(repo_dir)
```

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
