import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

from tqdm import tqdm, trange
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
    files = os.listdir("/home/dh/combine_dataset")
    for file in tqdm(files):
        file_name = file.split('.')[0]
        print(file_name)
        file_path = os.path.join("/home/dh/combine_dataset", file)
        frame_num = 32
        dataset = VideoCLIPDataset(None, frame_num, file_path)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False
        )
        data_iter = iter(dataloader)
        # Load CLIP model.
        clip_model = ".././ViT-B-32.pt" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(clip_model, device=device, jit=False)
        h5pyfile = "/home/disk1/anomaly/clip_patch_feat_%s.h5" % file_name
        dataset_feats = h5py.File(h5pyfile, "w")
        dataset_feats.create_dataset("features", (len(dataset), 32, 17, 512))
        dataset_feats.create_dataset("ids", (len(dataset),), 'S20')
        global_index = 0
        data_iter = iter(dataloader)
        for batch in data_iter:
            try:
                video_path = batch['video_path']
                batch_size = batch['video'].shape[0]
                for i in range(batch_size):
                    for j in range(frame_num):
                        with torch.no_grad():
                            image_features = model.encode_image(batch['video'][i][j].cuda())   # 0 video 0-j frame
                        dataset_feats['features'][global_index, j] = image_features.detach().cpu().numpy()
                    # print(batch['vid'][i].encode("ascii", "ignore"))
                    dataset_feats['ids'][global_index] = batch['vid'][i].encode("ascii", "ignore")
            except Exception:
                pass
        dataset_feats.close()
