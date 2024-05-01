import os
from PIL import Image
from chatcaptioner.blip2 import Blip2
import openai
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from utils import cal_text_emb
import torch
import matplotlib.pyplot as plt

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]
warnings.filterwarnings("ignore")

# keywords = ["Several cars collided, causing chaos", " A rear-end collision occurred", "One car hit", "emitting smoke"]
frames = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900, 930, 960, 990, 1020, 1050]
dest = r'/home/dh/pythonProject/AnomalyDataset/Data/capture_image/'
BLIP2DICT = {
    'FlanT5 XL': '/home/dh/zjy/ChatCaptioner-main/Video_ChatCaptioner/blip2-flan-t5-xl',
}
blip2 = Blip2('FlanT5 XL', 0)

Answer_Instruction = "Answer questions and follow the following restrictions: (1) If you are not sure about the answer, say you DO NOT KNOW honestly.  (2) DO NOT IMAGINE any contents that are NOT in the image. "
SUB_ANSWER_INSTRUCTION = 'Answer: '
Questions = ['describe the image in details.', 'Determine whether there are any abnormal events in the image.']


GPT_Detection_Instruction = "Based on the results of the Q&A, determine whether there are any abnormalities in the image. " \
                            "If there are abnormalities, return 'yes'; if there are no abnormalities, return 'no'; if you are unsure yourself, reply 'uncertain'." \
                            "Please ensure to return 'Yes' or 'No' or 'Uncertain' and do not add any extra content"\
                            "Following are the question and answer results:\n"
SUB_GPT_Detection_Instruction = "Your opinion is:"

GPT_Detection_Context = "Frame1:\n" \
                        "Question: describe the image in details.\n" \
                        "Answer: A car crashes into a truck.\n" \


def get_chat_log(questions, answers, last_n=0):
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n == 0:  # for blip2
        questions = questions[-1]
        chat_log = chat_log + 'Question: {}. '.format(questions)
    elif last_n == -1:  # for summary and gpt
        for i in range(len(answers)):
            chat_log = chat_log + template.format(questions[i], answers[i])
    return chat_log

def ask_questions(img, blip2, question):
    blip2_prompt = Answer_Instruction + '\n' + question + '\n' + SUB_ANSWER_INSTRUCTION
    answer = blip2.ask(img, blip2_prompt)
    return answer

def call_gpt3(gpt3_prompt, max_tokens=40, model="text-davinci-003"):  # 'text-curie-001' does work at all to ask questions
    response = openai.Completion.create(model=model, prompt=gpt3_prompt, max_tokens=max_tokens)  # temperature=0.6,
    reply = response['choices'][0]['text']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

f2img = []
detection_res = []

for frame in frames:
    path = os.path.join(dest, str(frame)+".jpg")
    img = Image.open(path)
    f2img.append(img)

for i, img in enumerate(f2img):
    frame_id = frames[i]
    GPT_Detection_Context = "Frame" + str(frame_id) + ":\n"
    temp = []
    for question in Questions:
        GPT_Detection_Context += "Question: " + question + "\n"
        answer = ask_questions(img, blip2, question)
        GPT_Detection_Context += "Answer: " + answer + "\n"
        GD_prompt = GPT_Detection_Instruction + GPT_Detection_Context + SUB_GPT_Detection_Instruction
        reply, _ = call_gpt3(GD_prompt)
        reply = reply.strip()
        temp.append(reply)
    print(temp)
    detection_res.append(temp)




# print(summary_prompt)
# print("ok")
# summary, _ = call_gpt3(summary_prompt)
# print(summary)
#
#
#
# GD_prompt = GPT_Detection_Instruction + GPT_Detection_Context + SUB_GPT_Detection_Instruction
# print(GD_prompt)
# reply, _ = call_gpt3(GD_prompt)
# reply = reply.strip()
# print(reply)











