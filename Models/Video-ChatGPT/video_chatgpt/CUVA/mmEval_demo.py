import sys, os
import argparse
import datetime
import json
import time

sys.path.append("/home/dh/pythonProject/CUVA/Models/Video-ChatGPT")

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


def upload_video(chat, video_path, mod=None):
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


def mmEval(chat, video_path, question, mist_feature=None, mod=None):
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
    parser.add_argument("--model-name", type=str,
                        default="/mnt/new_disk/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1")
    parser.add_argument('--vision_tower_path', type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False,
                        default="/mnt/new_disk/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin")
    parser.add_argument("--video_dir_path", type=str, default="/home/dh/combine_dataset")
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=4096)
    # need customize
    parser.add_argument("--task", type=str, default="Des")
    parser.add_argument("--mod", type=int, default=None)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    logger.info(args)
    disable_torch_init()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    video_dir = args.video_dir_path
    task = args.task

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(args.model_name, args.projection_path, args.vision_tower_path)

    # Create replace token, this will replace the <video> in the prompt.
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    # Create chat for the demo
    chat_description = Chat(args.model_name, "MMEval_des_v1", tokenizer, image_processor, vision_tower, model,
                            replace_token)
    chat_cause = Chat(args.model_name, "MMEval_cause", tokenizer, image_processor, vision_tower, model, replace_token)
    chat_result = Chat(args.model_name, "MMEval_result", tokenizer, image_processor, vision_tower, model, replace_token)
    chat_map = {'Des': chat_description, 'Cau': chat_cause, 'Res': chat_result}

    # h5_file_path = "/home/disk1/anomaly_mist/mist_features"
    # video_mist_features = read_mist_feature(h5_file_path)
    print('Initialization Finished')



    prompts = {
        'Des': '''Here I had a model describe the anomaly in the video and here are the answers he gave and the standard answer.
            You need to understand the video and rate the model's answer. The scoring range is 0-10.
            You need to evaluate the answer to this model in 5 aspects, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
            You need to analyze carefully, be demanding and give rich assessments and boldly give low scores.You need to give marks for each of the five areas in the following format: Score: x/10. ''',
        'Cau': '''Here will be an [model's answer] which are the root cause of an anomaly event.
        Please compare it with the [reference answer] and refer to the events in the video,then give the [model's answer] a score in 0 to 10 to evaluate the correctness of their reasoning.
        You need to evaluate the answer to this model in several ways, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
        You need to analyze carefully, be demanding and give rich assessments and boldly give low scores.You need to give marks for each of the five areas in the following format: Score: x/10.''',
        'Res': '''Here will be an [model's answer] which is the final result of an anomaly event. Please compare it with the [reference answer] and refer to the events in the video,
        then give the [model's answer] a score in 0 to 10 to evaluate the correctness of the reasoning and summarization of the results of this anomaly event. 
        You need to evaluate the answer to this model in 5 aspects, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
        You need to analyze carefully, be demanding and give rich assessments and boldly give low scores.You need to give marks for each of the five areas in the following format: Score: x/10.''',

    }
    while True:
        video_path = input("Please input the video path: ")
        if video_path == 'exit':
            break
        print('Task:', task)
        gt = input("Please input the ground truth: ")
        answer = input("Please input the model answer: ")
        if task == 'exit' or gt == 'exit':
            break
        prompt = prompts[task]
        prompt += ("[Reference: {}],\n[Model answer: {}]".format(gt, answer))
        try:
            # mist_feature = video_mist_features[video]
            score = mmEval(chat_map[task], video_path, prompt, mod=1)
        except Exception as e:
            print(e)
            continue
