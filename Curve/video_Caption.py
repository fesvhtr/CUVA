import os
from PIL import Image
from chatcaptioner.blip2 import Blip2, Blip2Processor
import re
import warnings
import openai
import os

os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"

openai.api_key = os.environ["OPENAI_API_KEY"]
warnings.filterwarnings("ignore")

Question_Instruction = "You are now playing a role in abnormal video comprehension. "\
    "You are able to understand a abnormal video by ASKING a lot of questions WITH OUT SEEING THE VIDEO."\
    "You need to try your best to identify the cause of the abnormality."\
    "An expert will then answer your question."\
    "Note that you cannot ask questions with very similar semantics or very simple questions."\
    "Try to diversify your questions and obtain as much information as possible within a limited number of rounds"\

SUB_Question_Instruction = "Thought: what does this video describe? What is the cause of the abnormality?"\
    "Action: ask one questions to guess the contents of the video."\
    "Restrictions: (1) You MUST ask questions from Frame 1 to Frame %s, all frames will ultimately involve " \
    "(2) One question at one round, the question format MUST be Frame_id: question, e.g. Frame_1: Describe it in details. "\
    "(3) Cannot ask the same question for the same frame" \
    "(4) AVOID asking yes/no questions. " \
    "(5) CANNOT continuously ask question about the same frame"\
    "(6) The probability of all frames being questioned should be equal"\
    "Questions: "

General_Question = 'Frame_1: Describe it in details.'

Answer_Instruction = "Answer given questions with the following restrictions. (1) If you are not sure about the answer, say you DO NOT KNOW honestly.  (2) DO NOT IMAGINE any contents that are NOT in the image. "

Summary_Instruction = "Based on the provided information, Please SUMMARIZE the contents of the video. Pay attention to the following restrictions"\
    "Restrictions: (1) DO NOT add information. "\
    "(2) DO NOT describe each frame individually and DO NOT mention the frame. "\
    "(3) DO NOT summarize negative or uncertain answers. "\
    "video summarization: "

SUB_ANSWER_INSTRUCTION = 'Answer: '

screen_shots_path_list = []
screen_shots = []
ScreenShots_LIMIT = 10
BLIP2DICT = {
    'FlanT5 XL': '/home/dh/zjy/ChatCaptioner-main/Video_ChatCaptioner/blip2-flan-t5-xl',
}

def iterate_files(folder_path):
    """
    This function iterates through all the files in a folder and prints their names.
    """
    for filename in os.listdir(folder_path):
        if filename == '30.jpg' or filename == '60.jpg' or filename == '90.jpg':
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):    # 检查是否是一个存在的文件
                screen_shots_path_list.append(file_path)

def find_digit(input):
    regex = r"Frame_(\d+)"
    index = -1  # 如果没匹配到，默认返回index为-1

    # Use re.search() to find the match in the sentence
    match = re.search(regex, input)

    # Extract the index from the match object
    if match:
        index = match.group(1)
        # print("Index found:", index)
    else:
        print("input: "+input)
        print("No index found in sentence.")

    return index

def get_chat_log(questions, answers, last_n=0):
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n == 0:
        questions = questions[-1]
        chat_log = chat_log + 'Question: {}. '.format(questions)
    elif last_n == -1:
        for i in range(len(answers)):
            chat_log = chat_log + template.format(questions[i], answers[i])

    return chat_log

def prepare_gpt_prompt(task_prompt, questions, answers, sub_prompt, flag=0):

    gpt_prompt = '\n'.join([task_prompt,
                             get_chat_log(questions, answers, flag),
                             sub_prompt])
    return gpt_prompt


def call_gpt3(gpt3_prompt, max_tokens=40, model="text-davinci-003"):  # 'text-curie-001' does work at all to ask questions
    response = openai.Completion.create(model=model, prompt=gpt3_prompt, max_tokens=max_tokens)  # temperature=0.6,
    reply = response['choices'][0]['text']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def ask_questions(screen_shots, blip2, gpt_model, n_rounds=5, n_blip2_context=0):
    questions, answers = [], []
    SUB_Question_Instruction_Fill = SUB_Question_Instruction % str(len(screen_shots))

    print('--------Chat Starts----------')
    for i in range(n_rounds):
        if i == 0:
            question = General_Question
            questions.append(question)
        else:
            gpt3_prompt = prepare_gpt_prompt(Question_Instruction,questions, answers,SUB_Question_Instruction_Fill)
            question, _ = call_gpt3(gpt3_prompt)
            question.strip()
            questions.append(question)

        print('GPT 3: ' + question)

        frame_id = find_digit(question)
        img_info = screen_shots[int(frame_id)-1]
        chat_log = get_chat_log(questions, answers, last_n=n_blip2_context)
        blip2_prompt = Answer_Instruction + '\n' + chat_log + '\n' + SUB_ANSWER_INSTRUCTION
        answer = blip2.ask(img_info, blip2_prompt)
        answers.append(answer)

        print('BLIP2: ' + answer)
    return questions, answers

def summary_chat(questions, answers, gpt_model, max_gpt_tokens=100):
    summary_prompt = prepare_gpt_prompt(
        Summary_Instruction,
        questions, answers,
        '', -1)
    summary, n_tokens = call_gpt3(summary_prompt, model=gpt_model, max_tokens=max_gpt_tokens)
    summary = summary.replace('\n', ' ').strip()
    return summary

if __name__ == '__main__':
    folder_path = r"/home/dh/pythonProject/AnomalyDataset/Data/capture_image/"
    iterate_files(folder_path)
    blip2_processor = Blip2Processor.from_pretrained(BLIP2DICT['FlanT5 XL'])
    for path in screen_shots_path_list:
        img = Image.open(path)
        screen_shots.append(img)

    if len(screen_shots) > ScreenShots_LIMIT:
        screen_shots = screen_shots[:ScreenShots_LIMIT]

    blip2 = Blip2('FlanT5 XL', 0)
    questions, answers = ask_questions(screen_shots, blip2, "text-davinci-003", n_rounds=20, n_blip2_context=0)

    summary = summary_chat(questions, answers, "text-davinci-003")
    print("Summary: " + summary)
