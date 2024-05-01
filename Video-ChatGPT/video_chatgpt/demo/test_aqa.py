import os
import argparse
import datetime
import json
import time
import sys
sys.path.append("/home/dh/pythonProject/AnomalyDataset/Video-ChatGPT")
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.demo.gradio_patch import Chatbot as grChatbot
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *
from tqdm import tqdm
import logging
import h5py


logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("cmd_demo")

headers = {"User-Agent": "Video-ChatGPT"}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def cmd_add_text(state, text, image, first_run):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return state, ""
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return state, moderation_msg

    text = text[:2536]  # Hard cut-off
    if first_run:
        text = text[:2200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, image)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state, ""



def cmd_upload_image(chat, video_path, state, mod):
    if video_path is None:
        print('Invalid video path')
        return state, None, False
    state = default_conversation.copy()
    img_list = []
    first_run = True
    img_list = chat.upload_video(video_path, img_list, mod = mod)
    return state, img_list, first_run


def test(chat, data_path):
    with open(os.path.join(data_path, 'instruction', 'gt.json'), 'r') as f:
        data_test = json.load(f)
    for ins in tqdm(data_test):
        video_path = os.path.join(data_path, 'video', ins['visual_input'])
        ans = test_single_video(chat, video_path, ins['instruction'])
        ins['output'] = ans
    with open(os.path.join(data_path, 'output', 'output_Video_ChatGPT.json'), 'w') as f:
        json.dump(data_test, f)
    print('Output saved')


def test_single_video(chat, video_path, question):
    state = None
    img_list = []
    temperature = args.temp
    max_output_tokens = args.max_tokens
    state, img_list, first_run = cmd_upload_image(video_path, state)
    state, moderation_msg = cmd_add_text(state, question, video_path, first_run)
    result_generator = chat.answer(state, img_list, temperature, max_output_tokens, first_run)
    for result in result_generator:
        state, _, img_list, first_run, _, *_ = result
    ans = state.messages[-1][-1]
    return ans


def build_cmd_demo(chat, video_path):
    state = None
    img_list = []
    temperature = args.temp
    max_output_tokens = args.max_tokens
    state, img_list, first_run = cmd_upload_image(video_path, state)
    while True:
        question = input('------Enter Question:')
        if question == 'next':
            return
        state, moderation_msg = cmd_add_text(state, question, video_path, first_run)
        result_generator = chat.answer(state, img_list, temperature, max_output_tokens, first_run)
        for result in result_generator:
            state, _, img_list, first_run, _, *_ = result
        print(DEFAULT_VIDEO_TOKEN)


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


def test_AIST(chat, video_path, question, mist_feature=None, mod=None):
    state = None
    img_list = []
    temperature = args.temp
    max_output_tokens = args.max_tokens
    state, img_list, first_run = cmd_upload_image(chat, video_path, state, mod)
    state, moderation_msg = cmd_add_text(state, question, video_path, first_run)
    result_generator = chat.answer(state, img_list, temperature, max_output_tokens, first_run, mist_feature)
    for result in result_generator:
        state, _, img_list, first_run, _, *_ = result
    ans = state.messages[-1][-1]
    return ans


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model-name", type=str, default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1")
    parser.add_argument('--vision_tower_path', type=str, default="/home/disk1/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt-cause")
    parser.add_argument("--projection_path", type=str, required=False, default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    return args

#todo: add prompt to compare
if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    logger.info(args)
    disable_torch_init()

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

    data_path = '/home/dh/pythonProject/AnomalyDataset/train_mist_v1/gt_Causev1.json'
    gt_path = '/home/dh/pythonProject/AnomalyDataset/Video-ChatGPT/video_chatgpt'
    ac_path = '/home/dh/pythonProject/AnomalyDataset/train_mist_v1/Causev1_ori.json'
    output_path = '/home/dh/pythonProject/AnomalyDataset/train_mist_v1/output/Causev1_ori.json'
    video_dir = '/home/dh/combine_dataset'
    err_id = []

    with open(data_path, 'r') as f:
        output_data = json.load(f)
    with open(ac_path, 'r') as f:
        ac_data = json.load(f)
    for i in tqdm(output_data):
        video = i['video_id']
        question = i['question']
        prompt = "Here are four answers (A,B,C,D) which are descriptions of the root cause of this anomaly event. Please rank the answers in the following choices according to their correctness. \
         Note: Each answer will start with a capital letter and a colon. The final output format will be a string of uppercase letters separated by commas, with the more correct answers appearing earlier.\
         and provide corresponding reasons\n"
        try:
            candidate_answers = ac_data[video]
            for order, ac_ans in enumerate(candidate_answers):
                order = chr(68 - order)
                ac_ans = ac_ans.replace('\n', ' ')
                prompt += ("{}:{} \n".format(order, ac_ans))
            mist_feature = video_mist_features[video]
            ans = test_AIST(chat, os.path.join(video_dir, video), prompt, mod = 1)
            i['output'] = ans
        except Exception as e:
            print(e)
            i['output'] = ''
            err_id.append(video)
            continue
    print('err_id:', err_id)
    with open(os.path.join(output_path, 'output_anomaly_rank.json'), 'w') as f:
        json.dump(output_data, f)
