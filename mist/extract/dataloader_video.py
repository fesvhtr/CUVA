from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import re
import cv2
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, ConvertImageDtype
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import pandas as pd
import json
import random

import decord
from decord import cpu, gpu
import clip

from PIL import Image


class VideoCLIPDataset(Dataset):
    def __init__(
            self,
            video_path_file,
            frame_num,
            video_dir=None,
            segment_file=None,
            raw_frame=False,
            frame_size=224,
            input_format="video",
            grid_num=4,
    ):
        if segment_file:
            self.video_paths, self.segment = self.load_segment(segment_file, video_dir)
        else:
            if video_path_file:
                self.meta_data = json.load(open(video_path_file))    # for Env-QA
                self.video_paths = list(self.meta_data.keys())
            else:
                self.video_paths = glob.glob(video_dir)       # for NExT-QA, AGQA
            self.segment = None

        self.input_format = input_format
        self.video_dir = video_dir
        self.raw_frame = raw_frame
        self.frame_num = frame_num  # 32
        self.frame_size = frame_size
        self.preprocess = preprocess_clip(self.frame_size)
        self.preprocess_patch = preprocess_patch(self.frame_size, grid_num=grid_num)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            video_id = video_path.split('/')[-1].replace('.mp4', '') if self.input_format == "video" else video_path
            video_id = video_id.split('/')[-1]
            question_id, segment = self.segment[idx] if self.segment else (video_id, None)

            # sample frames
            if self.input_format == "video":
                frames, frame_idxs, vlen = read_frames_decord(video_path, self.frame_num, mode='uniform',
                                                              image_size=None, begin_end_time=segment) # frames rgb
                frames = frames.permute(0, 3, 1, 2)  #
            else:
                frames, frame_idxs, vlen = self.read_frames_pil(video_path, self.frame_num, self.video_dir, mode='uniform')   # frames rgb

            # if self.raw_frame:
            raw_frames = frames

            # get patches
            video = self.preprocess(frames)   # [frame_num, 3, img_size, img_size]

            video_patch = self.patchify(frames)  # [frame_num, patch_num, 3, img_size, img_size]

            video = torch.cat([video.unsqueeze(dim=1), video_patch], dim=1)

            return {
                "video": video,
                "video_path": video_path,
                'raw_frame': raw_frames,
                'vid': video_id,
                'qid': question_id,
                'video_patch': video_patch
            }
        except Exception:
            print(video_path)
            return {
                "video_path": video_path,
            }

    @staticmethod
    def load_segment(segment_file, video_dir):
        if segment_file is None:
            return None
        elif ".json" in segment_file:
            return json.load(open(segment_file))
        elif ".csv" in segment_file:
            raw_video_segment = []
            with open(segment_file, "r") as file_handle:
                for line in file_handle.readlines():
                    raw_video_segment.append(line.strip('\n').split(','))
            # keys = raw_video_segment[0]
            # keys: qid, vid, start, end
            video_paths = [os.path.join(video_dir, d[1] + '.mp4') for d in raw_video_segment[1:]]
            video_segment = [(d[0], (float(d[2]), float(d[3]))) for d in raw_video_segment[1:]]
            return video_paths, video_segment

    def patchify(self, raw_frames, patch_stride=None):
        frame_num = raw_frames.shape[0]
        frames = self.preprocess_patch(raw_frames)  # [N, 3, width, height]
        patch_size = self.frame_size
        if patch_stride is None:
            patch_stride = patch_size
        frame_patches = frames.unfold(
            2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        frame_patches = frame_patches.reshape(frame_num, 3, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4)
        return frame_patches  # frame_num, patch_num, 3, img_size, img_size

    def read_frames_pil(self, video_id, num_frames, video_dir, mode='uniform'):
        def frame_id2path(f_id, v_dir):
            env, vid = re.findall("(FloorPlan.*?_physics)_(.*?)_img", f_id)[0]
            out_path = os.path.join(v_dir, env, vid, f_id + ".png")
            return out_path

        frame_ids = self.meta_data[video_id]['frame_ids']
        vlen = len(frame_ids)
        sampled_idxs = sample_frames(num_frames, vlen, sample=mode)  # sample: "random" or "uniform"
        frames = []
        for idx in sampled_idxs:
            frame_id = frame_ids[idx]
            frame_path = frame_id2path(frame_id, v_dir=video_dir)
            if os.path.exists(frame_path):
                frames.append(pil_to_tensor(Image.open(frame_path)))
            else:
                print(frame_id)
                continue

        if frames:
            frames = torch.stack(frames, dim=0)
        else:
            frames = torch.zeros([1, 3, 224, 224])

        return frames, sampled_idxs, vlen


def preprocess_patch(n_px=224, grid_num=4):
    '''
    preprocess for extract clip patch feature, only resize size is different with the original clip
    '''
    img_size = n_px * grid_num
    return Compose([
        Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(img_size),
        ConvertImageDtype(torch.float),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def preprocess_clip(n_px=224):
    '''
    clip preprocess:
    Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
    CenterCrop(size=(224, 224))
    <function _convert_image_to_rgb at 0x7f87918a64d0>
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )
    '''
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(n_px),
        ConvertImageDtype(torch.float),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def read_frames_decord(video_path, num_frames, mode='uniform', fix_start=None, image_size=(512, 512),
                       begin_end_time=None):
    # print("video path: {}".format(video_path))

    if image_size:
        width, height = image_size
        video_reader = decord.VideoReader(video_path, width=width, height=height, num_threads=1, ctx=cpu(0))
    else:
        video_reader = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
    # video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    if begin_end_time:
        fps = get_fps(video_path)
        begin_end_frame = [min(t * fps, vlen) for t in begin_end_time]
    else:
        begin_end_frame = None

    frame_idxs = sample_frames(num_frames, vlen, sample=mode,
                               fix_start=fix_start, begin_end_frame=None)  # sample: "random" or "uniform"
    if begin_end_frame:
        cur_num_frames = 5  # curve sample other frames
        frame_idxs_c = sample_frames(cur_num_frames, vlen, sample=mode,
                               fix_start=fix_start, begin_end_frame=begin_end_frame)  # sample: "random" or "uniform"
        frame_idxs = frame_idxs + frame_idxs_c
    frames = video_reader.get_batch(frame_idxs).byte()  # frames rgb
    #     frames = frames.permute(0, 3, 1, 2)  # [t, channel, , ]
    return frames, frame_idxs, vlen


def sample_frames(num_frames, vlen, sample='rand', fix_start=None, begin_end_frame=None):
    acc_samples = min(num_frames, vlen)
    begin, end = begin_end_frame if begin_end_frame else (0, vlen)
    intervals = np.linspace(start=begin, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def process_batch(batch, device):
    batch['img'] = [batch['img'][0].sum(dim=0).to(device)]
    for i, img_meta in enumerate(batch['img_metas'][0]):
        for k, v in img_meta.items():
            if isinstance(v, list):
                batch['img_metas'][0][i][k] = v[0]
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    batch['img_metas'][0][i][k][kk] = vv[0].to(device)
            elif isinstance(v, torch.Tensor):
                batch['img_metas'][0][i][k] = v[0].to(device)
    return batch


def viz_patches(x, figsize=(20, 20), patch_idx=None, topk=None, t=5):
    # x: num_patches, 3, patch_size, patch_size
    n = x.shape[0]
    nrows = int(math.sqrt(n))
    _, axes = plt.subplots(nrows, nrows, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        im = x[i].permute(1, 2, 0).numpy()
        im = (im * 255.).round().astype(np.uint8)
        if patch_idx is not None and i == patch_idx:
            im[0:t] = (255, 0, 0)
            im[im.shape[0]-t:] = (255, 0, 0)
            im[:, 0:t] = (255, 0, 0)
            im[:, im.shape[1]-t:] = (255, 0, 0)
        if topk is not None:
            if i in topk and i != patch_idx:
                im[0:t] = (255, 255, 0)
                im[im.shape[0]-t:] = (255, 255, 0)
                im[:, 0:t] = (255, 255, 0)
                im[:, im.shape[1]-t:] = (255, 255, 0)
        ax.imshow(im)
        ax.axis("off")
    plt.show()



