import os
import argparse
import datetime
import json
import time
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.demo.gradio_patch import Chatbot as grChatbot
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *
from tqdm import tqdm
import logging

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

    text = text[:1536]  # Hard cut-off
    if first_run:
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, image)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state, ""


def cmd_upload_image(image, state):
    if image is None:
        print('Invalid video path')
        return state, None, False
    state = default_conversation.copy()
    img_list = []
    first_run = True
    llm_message = chat.upload_video(image, img_list)
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


def test_AIST(chat, video_path, question):
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
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt_v1")
    parser.add_argument("--projection_path", type=str, required=False, default="")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    logger.info(args)
    disable_torch_init()

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(args.model_name, args.projection_path)

    # Create replace token, this will replace the <video> in the prompt.
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    # Create chat for the demo
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)
    print('Initialization Finished')

    data_path = '/home/disk1/funqa/test'
    output_path = '/home/dh/zsc/FunQA_test/output'
    err_id = []

    with open('/home/dh/zsc/FunQA_test/funqa_test.json', 'r') as f:
        output_data = json.load(f)
    for i in tqdm(output_data):
        video = i['visual_input']
        question = i['instruction']
        try:
            ans = test_AIST(chat, os.path.join(data_path, video), question)
            i['output'] = ans
        except Exception as e:
            print(e)
            i['output'] = ''
            err_id.append(video)
            continue
    print('err_id:', err_id)
    with open(os.path.join(output_path, 'output_funqa_VCG.json'), 'w') as f:
        json.dump(output_data, f)
