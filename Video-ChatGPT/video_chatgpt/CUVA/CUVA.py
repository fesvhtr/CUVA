import sys, os
import argparse
import datetime
import json
import time
import timeout_decorator

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


def CUVA(chat, video_path, question, mist_feature=None, mod=0):
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
                        default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1")
    parser.add_argument('--vision_tower_path', type=str,
                        default="/home/disk1/cuva/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False,
                        default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin")
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=4096)
    # need customize
    parser.add_argument("--question", type=str, default="causes")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt-cause")
    parser.add_argument("--mod", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--output_path", type=str, default='output_anomaly_cause.json')
    parser.add_argument("--cas_path", type=str,
                        default="/home/dh/pythonProject/AnomalyDataset/train_mist_v1/Causev1_ori.json")
    parser.add_argument("--gt_path", type=str,
                        default="/home/dh/pythonProject/AnomalyDataset/train_mist_v1/gt_Causev1.json")
    args = parser.parse_args()

    return args


@timeout_decorator.timeout(120)  # 120 seconds (2 minutes)
def inference_with_timeout(chat, video_path, prompt, mod):
    return inference_CUVA(chat, video_path, prompt, mod=mod)


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
    chat_des = Chat(args.model_name, 'cuva_des', tokenizer, image_processor, vision_tower, model, replace_token)
    chat_cause = Chat(args.model_name, 'cuva_cause', tokenizer, image_processor, vision_tower, model, replace_token)
    chat_result = Chat(args.model_name, 'cuva_result', tokenizer, image_processor, vision_tower, model, replace_token)
    chat_default = Chat(args.model_name, 'conv_video_chatgpt_v1', tokenizer, image_processor, vision_tower, model,
                        replace_token)


    prompts = {
        'des': 'Watch the video and describe any anomaly events you see in the order they happen. Focus on what is different from normal, like who or what is involved and their actions.',
        'cause': 'Explain why the anomaly in the video are happening. Use what you see in the video to make logical reasoning about the root reasons behind these anomalies.Please ensure that your response is logically rigorous and directly related to the abnormal events in the video and the potential reasons behind them.',
        'result': 'Figure out what results and effect these anomalies have. Link the anomaly directly to their outcomes, like how they affect people or the environment. Your answer should be as clear and specific as possible, avoiding generalities and focusing directly on the video rather than summarizing the impact of a type of event on society.',
        'Detection': '''Anomalies represent occurrences or scenarios that deviate from the norm, defying expectations and straying 
                     from routine conditions. e.g. crimes such as robberies, fights, vandalism, etc. such as robberies, fights, crimes
                      such as vandalism, car accidents or moving violations, accidents such as fires, sudden health problems of people, etc., 
                     can be defined as anomalies. Is there any unusual event in the video, answer YES or NO!
                     ''',
        'Classification': '''You will be presented with a video clip. After watching the video, please identify and categorize any anomalies or unusual events based on the categories provided below. Ensure you are specific in your description and refer to the categories and sub-categories accurately.
                        **Categories for Anomaly Detection**:
                        1. **Fighting**
                        2. **Animals Hurting People**
                        3. **Water Incidents**
                        4. **Vandalism**
                        5. **Traffic Accidents**
                        6. **Robbery**
                        7. **Theft**
                        8. **Pedestrian Incidents**
                        9. **Fire**
                        10. **Traffic Violations**
                        11. **Forbidden to Burn**
                        After viewing, please write down the category number,the output format is as follows: [XX]''',
        'Timestamp': 'Locate the position of the anomalous segment in the video, e.g. how many s to how many s, given as [xxxx,xxxx]',
    }
    chat_map = {
        'des': chat_des,
        'cause': chat_cause,
        'result': chat_result,
        'default': chat_default
    }
    h5_file_path = "/home/disk1/anomaly_mist/mist_features"
    video_mist_features = read_mist_feature(h5_file_path)
    print('Initialization Finished')

    video_dir = '/home/disk1/combine_dataset'
    for video_name in tqdm(os.listdir(video_dir)[0:20]):
        video_path = os.path.join(video_dir, video_name)
        prompt = prompts['Timestamp']
        try:
            # mist_feature = video_mist_features[video]
            score = CUVA(chat_map['default'], video_path, prompt, mod=args.mod)
        except Exception as e:
            print(e)
            continue
