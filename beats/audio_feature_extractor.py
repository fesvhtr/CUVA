# @FileName  :test.py
# @Time      :2023/9/22 11:36
# @Author    :Duh
import torch
import torchaudio
from BEATs import BEATs, BEATsConfig
from pydub import AudioSegment
from pydub.utils import make_chunks
import os, re
import numpy as np
import h5py

# load the fine-tuned checkpoints
checkpoint = torch.load('data/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()


def extract_beats_feature(audio_dir):
    for each in os.listdir(audio_dir):
        audio_tensor, sample_rate = torchaudio.load(os.path.join(audio_dir, each))
        print("sample rate of audio is %s" % sample_rate)
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
        padding_mask = torch.zeros_like(audio_tensor)
        audio_feature = BEATs_model.extract_features(audio_tensor, padding_mask=padding_mask)[0]
        return audio_feature

# audio_input_16khz = torch.randn(3, 10000)
# padding_mask = torch.zeros(3, 10000).bool()
# audio_feature = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
# audio_file = h5py.File("/home/dh/pythonProject/AnomalyDataset/beats/audio_feature.h5", "w")
# f = open('/home/dh/pythonProject/AnomalyDataset/beats/audio_feature.json', 'wb')

if __name__ == "__main__":
    save_dir = "/home/dh/pythonProject/AnomalyDataset/Aist/anomaly/"
    # for each in os.listdir("/home/dh/pythonProject/AnomalyDataset/Aist/anomaly"):  # 循环目录
    #     filename = re.findall(r"(.*?)\.mp3", each)  # 取出.mp3后缀的文件名
    #     if filename:
    #         song = AudioSegment.from_file('/home/dh/pythonProject/AnomalyDataset/Aist/anomaly/{}'.format(each), "mp3")  # 打开mp3文件
    #         time_duration = song.duration_seconds
    #         print(time_duration)
    #         segment = np.linspace(0, time_duration, num=9, retstep=True)[1] * 1000 # 每一段音频时长
    #         chunks = make_chunks(song, segment)  # 将文件切割为10s一块
    #         for i, chunk in enumerate(chunks):
    #             name = "%s%s-%d.mp3" % (save_dir, filename[0], i)
    #             chunk.export(name, format="mp3")

    extract_beats_feature(save_dir)
