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


def test(chat,data_path):
    with open(os.path.join(data_path,'instruction','gt.json'),'r') as f:
        data_test = json.load(f)
    for ins in tqdm(data_test):
        video_path = os.path.join(data_path,'video',ins['visual_input'])
        ans = test_single_video(chat,video_path,ins['instruction'])
        ins['output'] = ans
    with open(os.path.join(data_path,'output','output_Video_ChatGPT.json'),'w') as f:
        json.dump(data_test,f)
    print('Output saved')


def test_single_video(chat,video_path,question):
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


def build_cmd_demo(chat,video_path):
    state = None
    img_list = []
    temperature = args.temp
    max_output_tokens = args.max_tokens
    state, img_list, first_run = cmd_upload_image(video_path, state)
    while True:
        question = input('------Enter Question:')
        if question =='next':
            return
        state, moderation_msg = cmd_add_text(state,question,video_path,first_run)
        result_generator = chat.answer(state, img_list, temperature, max_output_tokens, first_run)
        for result in result_generator:
            state, _, img_list, first_run, _, *_ = result
        print(DEFAULT_VIDEO_TOKEN)


def test_AIST(chat,video_path,data):
    state = None
    img_list = []
    temperature = args.temp
    max_output_tokens = args.max_tokens
    state, img_list, first_run = cmd_upload_image(video_path, state)
    questions = {'Detection':'''Anomalies represent occurrences or scenarios that deviate from the norm, defying expectations and straying 
                 from routine conditions. e.g. crimes such as robberies, fights, vandalism, etc. such as robberies, fights, crimes
                  such as vandalism, car accidents or moving violations, accidents such as fires, sudden health problems of people, etc., 
                 can be defined as anomalies. Is there any unusual event in the video, answer YES or NO!
                 ''',
                 'Classification':'''
                    You will be presented with a video clip. After watching the video, please identify and categorize any anomalies or unusual events based on the categories provided below. Ensure you are specific in your description and refer to the categories and sub-categories accurately.
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
                    After viewing, please write down the category number,the output format is as follows: [XX]
                    ''',
                 'Timestamp':'Locate the position of the anomalous segment in the video, e.g. how many s to how many s, given as [xxxx,xxxx]',
                 'Description':'Give a detailed description of the anomalous segment in the video. Please remember to describe the details of the incident',
                 'Cause':'Please reason logically and give the root cause of the anomalies in the video in detail',
                 'Result':'Please reason logically and give a detailed and organized account of the final outcome caused by the unusual events in the video',
                 }

    state, moderation_msg = cmd_add_text(state,questions['Detection'],video_path,first_run)
    result_generator = chat.answer(state, img_list, temperature, max_output_tokens, first_run)
    for result in result_generator:
        state, _, img_list, first_run, _, *_ = result
    ans =state.messages[-1][-1]
    video_name = video_path.split('/')[-1]
    output4video = {
        'visual_input':video_name,
        'output':ans,
        'task':key,
    }
    data.append(output4video)


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
    parser.add_argument("--model-name", type=str,
                        default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/LLaVA-7B-Lightening-v1-1")
    parser.add_argument('--vision_tower_path', type=str,
                        default="/home/disk1/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff")
    parser.add_argument("--projection_path", type=str, required=False,
                        default="/home/dh/zsc/VideoBench/model/Video-ChatGPT/video_chatgpt-7B.bin")
    parser.add_argument("--conv-mode", type=str, default="conv_video_chatgpt_v1")
    parser.add_argument("--temp", type=float,default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--data_path", type=str,default='/home/disk1/')
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

    data_path = args.data_path
    output_path = args.output_path
    output_json_data = []
    # build demo
    # while True:
    #     video_path = input('------Enter video path:')
    #     build_cmd_demo(chat,video_path)
    wrong_video = []
    video_list = os.listdir(data_path)

    for video in tqdm(video_list[0:30]):
        try:
            test_AIST(chat,os.path.join(data_path,video),output_json_data)
        except Exception as e:
            print(e)
            wrong_video.append(video.split('/')[-1])
            continue
    print('wrong_video:',wrong_video)
    with open(os.path.join(output_path,'output_VCG_part3.json'),'w') as f:
        json.dump(output_json_data,f)
