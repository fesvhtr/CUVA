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

def mmaEval(chat, video_path, question, mist_feature=None, mod=None):
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
    parser.add_argument("--h5_file_path", type=str, default="/home/disk1/anomaly_mist/mist_features")
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=4096)
    # need customize
    parser.add_argument("--question", type=str, default="causes")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt-cause")
    parser.add_argument("--mod", type=int, default=None)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--output_path", type=str, default='output_anomaly_cause.json')
    parser.add_argument("--cas_path", type=str, default="/home/dh/pythonProject/AnomalyDataset/train_mist_v1/Causev1_ori.json")
    parser.add_argument("--gt_path", type=str, default="/home/dh/pythonProject/AnomalyDataset/train_mist_v1/gt_Causev1.json")
    parser.add_argument("--sub_path", type=str)
    args = parser.parse_args()

    return args

def chk_file(submission_file, answer_file):
    with open(submission_file) as f:
        submission = json.load(f)
    with open(answer_file) as f:
        answer = json.load(f)

    chk_answer = []
    for data in answer:
        chk_answer.append({
            'task': data['task'],
            'visual_input': data['visual_input'],
            'ID': data['ID']
        })

    diff = False
    for data in submission:
        if {
            'task': data['task'],
            'visual_input': data['visual_input'],
            'ID': data['ID']
        } not in chk_answer:
            print(data)
            diff = True
            break

    assert not diff, 'Submission file is not valid'
    print('File is valid! Loading File...')

    submission = sorted(submission, key=lambda x: x['ID'])
    answer = sorted(answer, key=lambda x: x['ID'])

    # 假设 submission 和 answer 已经是排序过的列表
    submission = sorted(submission, key=lambda x: x['ID'])
    answer = sorted(answer, key=lambda x: x['ID'])

    # 创建一个集合来存储相同的 ID 和 output
    duplicate_id_output = set()

    # 找出 submission 和 answer 中相同的 ID 和 output
    for sub_data in submission:
        for ans_data in answer:
            if sub_data['ID'] == ans_data['ID'] and sub_data['output'] == ans_data['output']:
                duplicate_id_output.add((sub_data['ID'], sub_data['output']))

    # 过滤掉 submission 和 answer 中存在于 duplicate_id_output 的项
    filtered_submission = [data for data in submission if (data['ID'], data['output']) not in duplicate_id_output]
    filtered_answer = [data for data in answer if (data['ID'], data['output']) not in duplicate_id_output]

    # 更新 submission 和 answer 列表
    submission = filtered_submission
    answer = filtered_answer
    submission = sorted(submission, key=lambda x: x['ID'])
    answer = sorted(answer, key=lambda x: x['ID'])
    print('submission: ',len(submission),'answer: ',len(answer))
    return submission,answer

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
    chat_description = Chat(args.model_name, "MMEval_des_v1", tokenizer, image_processor, vision_tower, model, replace_token)
    chat_cause = Chat(args.model_name, "MMEval_cause", tokenizer, image_processor, vision_tower, model, replace_token)
    chat_result = Chat(args.model_name, "MMEval_result", tokenizer, image_processor, vision_tower, model, replace_token)
    chat_map = {"Description": chat_description, "Cause": chat_cause, "Result": chat_result}

    h5_file_path = args.h5_file_path
    video_mist_features = read_mist_feature(h5_file_path)
    print('Initialization Finished')

    gt_file = args.gt_path
    submission_file = args.sub_path
    output_path = args.output_path

    model_name = submission_file.split('/')[-1].split('test_AQA_')[-1].split('.')[0]
    output_file = os.path.join(output_path, 'mmaEval_output_{}.json'.format(model_name))
    video_dir = '/home/dh/combine_dataset'
    err_id = []
    output_data = []

    answer_data, gt_data = chk_file(submission_file, gt_file)

    prompts = {
        'Cause': '''Here will be an [model's answer] which are the root cause of an anomaly event.
Please compare it with the [reference answer] and refer to the events in the video,then give the [model's answer] a score in 0 to 10 to evaluate the correctness of their reasoning.
You need to evaluate the answer to this model in several ways, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
You need to analyze carefully, be demanding and give rich assessments and boldly give low scores.You need to give marks for each of the five areas in the following format: Score: x/10.''',

        'Result': '''Here will be an [model's answer] which is the final result of an anomaly event. Please compare it with the [reference answer] and refer to the events in the video,
then give the [model's answer] a score in 0 to 10 to evaluate the correctness of the reasoning and summarization of the results of this anomaly event. 
You need to evaluate the answer to this model in 5 aspects, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
You need to analyze carefully, be demanding and give rich assessments and boldly give low scores.You need to give marks for each of the five areas in the following format: Score: x/10.''',

        'Description': '''Here I had a model describe the anomaly in the video and here is the [model's answer] he gave and the standard answer.
You need to understand the video and rate the model's answer. The scoring range is 0-10.
You need to evaluate the answer to this model in 5 aspects, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
You need to analyze carefully, be demanding and give rich assessments and boldly give low scores.You need to give marks for each of the five areas in the following format: Score: x/10. '''
    }
    for i in tqdm(range(len(answer_data))):
        video = answer_data[i]['visual_input']
        answer = answer_data[i]['output']
        reference = gt_data[i]['output']
        task = answer_data[i]['task']
        if task in ['Detection','Classification','Timestamp']:
            continue
        print(task)
        prompt = prompts[task]
        prompt += ("[Reference: {}],\n[Model answer: {}]".format(reference, answer))

        try:
            mist_feature = video_mist_features[video]
            score = mmaEval(chat_map[task], os.path.join(video_dir, video), prompt, mod = args.mod, mist_feature=mist_feature)
        except:
            print('Error ID: ', answer_data[i]['ID'])
            err_id.append(answer_data[i]['ID'])
            continue
        output_data.append({
            'visual_input': video,
            'task': task,
            'prompt': prompt,
            'output': score,
        })
    print('Error ID: ', err_id)
    with open(os.path.join(output_file), 'w') as f:
        json.dump(output_data, f)
