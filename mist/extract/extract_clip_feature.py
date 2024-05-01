# @FileName  :test.py
# @Time      :2023/9/22 16:37
# @Author    :Duh
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import tqdm
import h5py
import clip
import torch
from torch.utils.data import DataLoader
from dataloader_video import VideoCLIPDataset

import math
import urllib.request
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


if __name__ == "__main__":

    frame_num = 32
    dataset = VideoCLIPDataset(None, frame_num, r"/home/dh/combine_dataset/*.mp4")
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=8,
        shuffle=False
    )
    data_iter = iter(dataloader)

    # Load CLIP model.
    clip_model = ".././ViT-B-32.pt"
    # clip_model = "ViT-B/32" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)

    dataset_feats = h5py.File("/home/dh/pythonProject/AnomalyDataset/mist/data/feats/anomaly/clip_patch_feat_all.h5","w")
    dataset_feats.create_dataset("features", (len(dataset), 32, 17, 512))
    dataset_feats.create_dataset("ids", (len(dataset), ), 'S20')
    # dataset_feats.close()

    global_index = 0
    video_ids = {}
    for batch in tqdm.tqdm(data_iter):
        video_path = batch['video_path']
        batch_size = batch['video'].shape[0]
        for i in range(batch_size):
            for j in range(frame_num):
                with torch.no_grad():
                    image_features = model.encode_image(batch['video'][i][j].cuda())
                dataset_feats['features'][global_index, j] = image_features.detach().cpu().numpy()
            dataset_feats['ids'][global_index] = batch['vid'][i].encode("ascii", "ignore")
            global_index += 1
    dataset_feats.close()



