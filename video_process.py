# @FileName  :video_process.py
# @Time      :2023/9/6 16:02
# @Author    :Duh
# !/usr/bin/python
# -*- coding:utf8 -*-

import os
from shutil import copyfile
import cv2, json

from huggingface_hub import snapshot_download

names = ['.mp4']  # 需要随机替换的后缀名列表
count = 0


def test(path):
    """get video frame information"""
    global count
    files = os.listdir(path)  # 获取当前目录的所有文件及文件夹

    res = {}
    for file in files:
        if os.path.isdir(file):
            continue
        video = cv2.VideoCapture(os.path.join(path, file))
        count += 1
        print(file)
        size = {"width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        print(size)
        res[file] = size
        video.release()

    p = open("/home/dh/pythonProject/AnomalyDataset/train_mist_v1/frame_size.json", "w")
    json.dump(res, p)
    p.close()

if __name__ == '__main__':
    test("/home/dh/combine_dataset")
    print(count)
