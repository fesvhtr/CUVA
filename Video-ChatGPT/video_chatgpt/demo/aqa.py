import sys,os
import argparse
import datetime
import json
import time
sys.path.append("/home/dh/pythonProject/AnomalyDataset/Video-ChatGPT")

from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.demo.chat import Chat
from tqdm import tqdm
import torch
import logging
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("cmd_demo")

headers = {"User-Agent": "Video-ChatGPT"}

def read_mist_feature(h5_file):
    video_mist_features = {}
    with h5py.File(h5_file) as fp:
        vids = fp['ids']
        feats = fp['features']
        # print(feats.shape)  # v_num, clip_num, feat_dim
        for id, (vid, feat) in enumerate(zip(vids, feats)):
            vid = vid.decode('utf-8')
            video_mist_features[vid] = feat
    return video_mist_features


def upload_video(chat, video_path, mod=0):
    """
    upload video to the chat class
    :param chat:  chat class
    :param video_path:  the path of the video
    :param mod: sampling mod (None,1,2) respectively refers to three sampling method
    :return: sample images
    """
    if video_path is None:
        print('Invalid video path')
    first_run = True
    img_list = []
    img_list = chat.upload_video(video_path, img_list, mod=mod)
    return img_list, first_run

def upload_text(state, text, image, first_run):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return state, ""
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return state, moderation_msg

    text = text[:3336]  # Hard cut-off
    if first_run:
        text = text[:3000]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
            state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state, ""

def test_AIST(chat, video_path, question, mist_feature=None, mod=0):
    img_list = []
    state = default_conversation.copy()
    temperature = args.temp
    max_output_tokens = args.max_tokens
    img_list, first_run = upload_video(chat, video_path, mod)
    state, moderation_msg = upload_text(state, question, video_path, first_run)
    result_generator = chat.answer(state, img_list, temperature, max_output_tokens, first_run, mist_feature)
    for result in result_generator:
        state, _, img_list, first_run, _, *_ = result
    ans = state.messages[-1][-1]
    return ans

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model-name", type=str, default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1")
    parser.add_argument('--vision_tower_path', type=str, default="/home/disk1/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff")
    parser.add_argument("--projection_path", type=str, required=False, default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin")
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=4096)
    # need customize
    parser.add_argument("--question", type=str, default="causes")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt-cause")
    parser.add_argument("--mod", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--output_path", type=str, default='output_anomaly_cause.json')
    parser.add_argument("--cas_path", type=str, default="/home/dh/pythonProject/AnomalyDataset/train_mist_v1/Causev1_ori.json")
    parser.add_argument("--gt_path", type=str, default="/home/dh/pythonProject/AnomalyDataset/train_mist_v1/gt_Causev1.json")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    logger.info(args)
    disable_torch_init()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(args.model_name, args.projection_path, args.vision_tower_path)

    # Create replace token, this will replace the <video> in the prompt.
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    # Create chat for the demo
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)

    h5_file_path = "/home/disk1/anomaly_mist/mist_features"
    video_mist_features = read_mist_feature(h5_file_path)
    print('Initialization Finished')

    gt_path = args.gt_path
    candidate_path = args.cas_path
    output_dir = '/home/dh/pythonProject/AnomalyDataset/train_mist_v1/output'
    video_dir = '/home/dh/combine_dataset'
    err_id = []

    with open(gt_path, 'r') as f:
        output_data = json.load(f)
    with open(candidate_path, 'r') as f:
        ac_data = json.load(f)
    for i in tqdm(output_data):
        video = i['video_id']
        question = i['question']
        prompt = "Here are three answers (A,B,C) which are the descriptions of the {} of this anomaly event. Please rank the answers in the following choices according to their correctness. \
         Note: Each answer will start with a capital letter and a colon. The final output format will be a string of uppercase letters separated by commas, with the more correct answers appearing earlier.\
         and provide corresponding reasons\n.".format(args.question)
        try:
            candidate_answers = ac_data[video]
            for order, ac_ans in enumerate(candidate_answers):
                if order == 3:
                    continue
                order = chr(67 - order)
                ac_ans = ac_ans.replace('\n', ' ')
                prompt += ("{}:{} \n".format(order, ac_ans))
            mist_feature = video_mist_features[video]
            ans = test_AIST(chat, os.path.join(video_dir, video), prompt, mod = args.mod)
            i['output'] = ans
        except Exception as e:
            print(e)
            i['output'] = ''
            err_id.append(video)
            continue
    print('err_id:', err_id)
    with open(os.path.join(output_dir, args.output_path), 'w') as f:
        json.dump(output_data, f)
