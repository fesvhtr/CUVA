import cv2
import numpy as np
import json
import sys
import os
import yaml
import torch
from PIL import Image

from chatcaptioner.video_chat import set_openai_key,caption_for_video
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import RandomSampledDataset, plot_img, print_info
from chatcaptioner.video_reader import read_video_with_timestamp, read_video_sampling,read_video_with_timestamp_key_frame


# VIDEO_FOLDER=sys.argv[1]   # msvd_data/videos
# CAPTION_FILE = sys.argv[2]   # msvd_data/caption.txt
# OUTPUT_FOLDER=sys.argv[3]    # output/
# VIDEO_LIMIT=int(sys.argv[4])     # 7


VIDEO_FOLDER = "msvd_data/videos"   # msvd_data/videos
CAPTION_FILE = "msvd_data/caption.txt"   # msvd_data/caption.txt
OUTPUT_FOLDER = "./output/"    # output/
VIDEO_LIMIT = 1    # 7


blip2s = {
    'FlanT5 XL': Blip2('FlanT5 XL', device_id=0, bit8=False)
}


video_files = []

def iterate_files(folder_path):
    """
    This function iterates through all the files in a folder and prints their names.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):    # 检查是否是一个存在的文件
            video_files.append(file_path)

iterate_files(VIDEO_FOLDER)



# add the video caption pairs
video_caption = {}
with open(CAPTION_FILE,"r") as f:
    for line in f.readlines():   # example: 1QVn0FffExM_14_24 a lion is licking his chops
        data = line.split(" ")
        video_id = data[0]
        caption = " ".join(data[1:]).replace("\n","")

        if video_id not in video_caption:
            video_caption[video_id] = [caption]
        else:
            video_caption[video_id].append(caption)



# extract the video frames with uniform sampling
video_list = []
for video_path in video_files[:VIDEO_LIMIT]:
    video_id = video_path.split("/")[-1].replace(".avi","")   # default code
    if video_id in video_caption.keys():
        new_json_file = {}
        new_json_file["video_id"] = video_id
        new_json_file["video_path"] = video_path
        new_json_file["annotation"] = video_caption[video_id]

        sampled_frames = read_video_sampling(video_path, num_frames=8)
        new_json_file["features"] = sampled_frames
        video_list.append(new_json_file)


for sample in video_list:
    video_id = sample["video_id"]
    features = sample["features"]
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    with open(OUTPUT_FOLDER+video_id+".txt","w") as f:
        sub_summaries = caption_for_video(blip2s['FlanT5 XL'], features, print_mode="chat", n_rounds=10, model='text-davinci-003') # default n_rounds=30
        caption = sample["annotation"]
        for cap in caption:
            f.write("ground truth: "+ cap+"\n")
        f.write("chatCaptioner: " + sub_summaries["ChatCaptioner"]["caption"]+"\n\n\n")
        f.write("chat log:\n")
        for element in sub_summaries["ChatCaptioner"]["chat"]:
            f.write(element["content"]+"\n")
