import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
import cv2
import sys
import random
import json
sys.path.append('/home/dh/xujunrui/Video-ChatGPT')

from video_chatgpt.model import VideoChatGPTLlamaForCausalLM
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.constants import *
import torch
def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs
def read_json(path):
    with open(path, "r", encoding='utf-8') as f:
        load_data = json.load(f)
    return load_data


def mod_load_video(vis_path, n_clips=1, num_frm=100, add_frm=200, load_mod=None):
    """
    Load video frames from a video file.
    Load video name from a json file called Voting.json.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """
    print("use mod sampler")
    # 从参数中提取出video的编号
    datas = read_json('/home/dh/xujunrui/Video-ChatGPT/data/Voting_new.json')
    Video_ID = int(vis_path.split('/')[-1].split('.')[0])
    # print(Video_ID)
    # 获取voting的json文件中的重要区间

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # print("The total frame is:",total_frame_num)#打印一下总帧数
    vc = cv2.VideoCapture(vis_path)
    fps = vc.get(cv2.CAP_PROP_FPS)  # 获取fps帧率
    # print("The fps of this video is:",round(fps))

    moments = []
    lst = []
    for data in datas:
        if (data != None):
            for i in data:
                if (i["video_name"] == Video_ID):
                    # print(i)
                    temp = [int(j) * int(fps) for j in i["answer"]]
                    lst.append(temp)
    # 区间去重
    [moments.append(i) for i in lst if i not in moments]
    # print(moments)

    # Currently, this function supports only 1 clip
    assert n_clips == 1
    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)

    # Get indices of frames to extract
    old_frame_idx = get_seq_frames(total_frame_num, total_num_frm)

    # print("The origin frames is:", old_frame_idx)
    # print("The frames length is:",len(old_frame_idx))

    if (load_mod == 1):  # 固定帧数算法
        imp_moment = []
        temp = []
        # 找出所有在重要区间里面的
        for moment in moments:
            for x in old_frame_idx:
                if (x >= moment[0] and x <= moment[1]):
                    temp.append(x)
        [imp_moment.append(i) for i in temp if i not in imp_moment]
        temp = []
        imp_moment.sort()
        # print("Important moments:",imp_moment)

        mod_lst = []
        for i in old_frame_idx:
            if (i not in imp_moment):
                mod_lst.append(i)
        # print("Mod lst is:",mod_lst)

        add_list = []
        for moment in moments:
            temp.append(round(moment[0] * 0.17 + moment[1] * 0.83))  # 确定一个重要区间替换几个帧
            temp.append(round(moment[0] * 0.33 + moment[1] * 0.67))  # 确定一个重要区间替换几个帧
            temp.append(round((moment[1] + moment[0]) * 0.5))  # 确定一个重要区间替换几个帧
            temp.append(round(moment[0] * 0.67 + moment[1] * 0.33))  # 确定一个重要区间替换几个帧
            temp.append(round(moment[0] * 0.83 + moment[1] * 0.17))  # 确定一个重要区间替换几个帧

        [add_list.append(i) for i in temp if i not in add_list]
        add_list.sort()
        # print("frame we add:", add_list)

        # 随机抽取add_list长度的不在重要区间里面的数值,生成相应数量即可
        pop_list = []
        pop_list = random.sample(mod_lst, min(len(add_list), len(mod_lst)))
        pop_list.sort()
        # print("frame we remove:", pop_list)

        for i in range(len(pop_list)):
            old_frame_idx.pop(old_frame_idx.index(pop_list[i]))
            old_frame_idx.append(add_list[i])

        old_frame_idx.sort()
        # print("The final frame:", old_frame_idx)
        # Extract frames as numpy array
        img_array = vr.get_batch(old_frame_idx).asnumpy()
        # Set target image height and width
        target_h, target_w = 224, 224
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

            # Reshape array to match number of clips and frames
            img_array = img_array.reshape(
                (n_clips, len(old_frame_idx), img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
            # Convert numpy arrays to PIL Image objects
            clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(len(old_frame_idx))]

    elif (load_mod == 2):
        # 固定帧率采样,但是这样的话下面的array的系数就必须改成len(frame_idx)
        for moment in moments:
            # print(moment)
            # 定义需要多抽的帧数，以往全部一秒一帧采样，现在一秒二帧试一试
            for i in range(round((moment[1] - moment[0]) / fps)):
                # append到frame_idx中
                start = int(np.round(moment[0] + fps * i))
                end = int(np.round(moment[0] + fps * (i + 1)))
                # 该区间为一秒区间，基本上间隔30帧，在这里插入多少帧数决定密集采样频率
                old_frame_idx.append(start + round(fps * 0.6))  # 一秒中间插入一帧
                old_frame_idx.append(start + round(fps * 0.3))  # 一秒中间插入一帧
        old_frame_idx.sort()
        frame_idx = []
        [frame_idx.append(i) for i in old_frame_idx if i not in frame_idx]  # 去重
        # print("The mod frames ID is:", frame_idx)
        # print("The mod frames length is:", len(frame_idx))

        # Extract frames as numpy array
        img_array = vr.get_batch(frame_idx).asnumpy()
        # Set target image height and width
        target_h, target_w = 224, 224
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

            # # Reshape array to match number of clips and frames
            # img_array = img_array.reshape(
            #     (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
            # # Convert numpy arrays to PIL Image objects
            # clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

            # Reshape array to match number of clips and frames
            img_array = img_array.reshape(
                (n_clips, len(frame_idx), img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
            # Convert numpy arrays to PIL Image objects
            clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(len(frame_idx))]

    # 不变帧数采样
    # 想法是，将非重要区间的随机帧移动到重要区间的给定区域帧数，然后移动的帧数默认为两帧，这两帧数

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def initialize_model(model_name, projection_path=None):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """

    # Disable initial torch operations
    disable_torch_init()

    # Convert model name to user path
    model_name = os.path.expanduser(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                         use_cache=True)

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    model = model.cuda()

    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # Set video token length
    video_token_len = 356

    return model, vision_tower, tokenizer, image_processor, video_token_len

if __name__ == "__main__":
    # print(load_video("/home/dh/VideoData/00016.mp4"))#原始抽帧函数，返回PIL Image objects
    # print(len(load_video("/home/dh/VideoData/00016.mp4")))#100 frames

    mod_load_video("/home/dh/VideoData/00124.mp4",load_mod=1)#原始抽帧函数，返回PIL Image objects
    #print('----')
    #mod_load_video("/home/dh/VideoData/00014.mp4",load_mod=2)#原始抽帧函数，返回PIL Image objects

