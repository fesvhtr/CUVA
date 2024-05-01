# @FileName  :test.py
# @Time      :2023/9/22 11:36
# @Author    :Duh
import torch
from BEATs import BEATs, BEATsConfig
from pydub import AudioSegment
from pydub.utils import make_chunks
import os, re


# load the fine-tuned checkpoints
checkpoint = torch.load('data/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

# predict the classification probability of each class
audio_input_16khz = torch.randn(3, 10000)
padding_mask = torch.zeros(3, 10000).bool()
audio_feature = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]



# # 循环目录下所有文件
for each in os.listdir("D:/PycharmProjects/拾音器/"):  # 循环目录
    filename = re.findall(r"(.*?)\.mp3", each)  # 取出.mp3后缀的文件名
    print(each)
    if each:
        song = AudioSegment.from_file('D:/PycharmProjects/'.format(each), "mp3")  # 打开mp3文件
        time_duration = song.duration_seconds
        size = time_duration / 8  # 每一段音频时长
        chunks = make_chunks(song, size)  # 将文件切割为10s一块

        for i, chunk in enumerate(chunks):
            audio_feature = BEATs_model.extract_features(chunk, padding_mask=padding_mask)[0]

            # chunk_name = "{}-{}.mp3".format(each.split(".")[0], i)    # 也可以自定义名字
            # chunk.export('D:/PycharmProjects//{}'.format(chunk_name), format="mp3")  # 新建的保存文件夹


if __name__ == "__main__":

    print(audio_feature)