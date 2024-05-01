import dataclasses
from enum import auto, Enum
from typing import List
from video_chatgpt.eval.xjr_model_utils import load_video, mod_load_video


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    CHOICE = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    user_description = None

    skip_next: bool = False
    text: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.CHOICE:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            if self.user_description == None:
                return None
            for i, desc in enumerate(self.user_description):
                desc = desc.replace('\n', ' ')
                ret += ("user" + str(i)+ ": " + desc + "\n")
            ret += seps[i % 2]
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def append_desc(self, desc):
        self.user_description = desc

    # def get_text(self):
    #     return self.text

    def get_video_frames(self, n_clips=1, num_frm=100):
        video_frames = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, video_path = msg

                    clip_imgs = load_video(video_path, n_clips, num_frm)

                    for image in clip_imgs:
                        video_frames.append(image)
        return video_frames

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image = msg
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        # Hack to make the demo work
        try:
            if '<video>' in ret[0][0]:
                ret[0][0] = ret[0][0].replace("<video>", "")
        except Exception as e:
            pass

        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
         "Renewable energy sources are those that can be replenished naturally.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
conv_video_chatgpt_v1 = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_video_chatgpt_result = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, the video showcases the entire process of an anomaly event."
           "Here are several answers which are descriptions of the results of this anomaly event. Please rank the answers in the following choices according to their correctness. "
            "Note: Each answer will start with a capital letter and a colon. The final output format will be a string of uppercase letters separated by commas, with the more correct answers appearing earlier.",
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(
        ("USER", '''Here are three answers (A,B,C) which are descriptions of the results of this anomaly event. Please rank the answers in the following choices according to their correctness. Also give the reason for your ranking. 
        A: The given descriptions do not provide any specific details that would qualify as an anomaly.  As such, I cannot provide a detailed and organized account of the final outcome caused by the unusual events in the video.  If more specific details were provided, such as a person driving aggressively or a car parked in an unusual location, I would be able to provide a more detailed and specific explanation for the outcome.
        B: The unusual events depicted in the video, such as the robbery, can have a significant impact on the community and the individuals involved. The robbery, in particular, can lead to a range of negative consequences, including: 1. Financial Losses: The individuals involved in the robbery may lose valuable possessions, such as money, electronics, or other personal belongings
        C: Windows were damaged and property was stolen.Clothes and debris were taken out and scattered everywhere.There is a lot of dirt on the thief's shoes, making the floor dirty as well,
         '''),
        ("ASSISTANT",
         "Rank: C,B,A Because C is the most reasonable answer with some details. B cannot provide a detailed and organized account of the final outcome. A has base logic with a lot of wrong details")
    ),
    #D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter.  # but B has worse logic with wrong details.
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)
conv_video_chatgpt_description = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, the video showcases the entire process of an anomaly event."
           "Here are several answers which are descriptions of the results of this anomaly event. Please rank the answers in the following choices according to their correctness. "
            "Note: Each answer will start with a capital letter and a colon. The final output format will be a string of uppercase letters separated by commas, with the more correct answers appearing earlier.",
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(
        ("USER", '''Here are three answers (A,B,C) which are descriptions of the results of this anomaly event. Please rank the answers in the following choices according to their correctness. Also give the reason for your ranking. 
        A: It is not clear from the given descriptions what the specific categories or sub-categories are.  The descriptions are mostly general or vague and do not provide any specific details that would qualify an event as an anomaly. Please refer to the categories provided for more details.
        B: In the video, there is a man talking about a robbery that occurred in the area. The segment starts at 0:00 and lasts for 0:05. The man is standing in front of a building and is giving a detailed description of the incident, which involves an altercation between two individuals.
        C: Three masked thieves entered the house through the backyard to steal, damaging the building and stealing property, making the house chaotic and dirty,
         '''),
        ("ASSISTANT",
         "Rank: C,B,A. Because C is the most reasonable description with some details. B cannot provide a detailed and organized account of the video description. A has base logic with some wrong details")
    ),
    #D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter.  # but B has worse logic with wrong details.
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)
# "You are Video-ChatGPT, a large vision-language assistant. "
#        "You are able to understand the video content that the user provides, the video showcases the entire process of an anomaly event."
#        "Your task is to rank the following candidate answers so that the most relevant or suitable answer regarding the cause of the anomaly event is placed at the top. "
#        "Please provide a list of answers arranged in the order you consider most appropriate"
#        "You need to understand my above requirements in conjunction with the following examples and complete the task as instructed."
#        "Input: A: A masked robber in a pullover entered a grocery store, pointed a gun at the cashier and carried out a robbery, threatening to kill him"
#        "B: First, we see a man standing in front of a convenience store at night. He is wearing a blue shirt and has a blue hoodie on. Next, we see a man standing in front of a convenience store at night. The store is dark, and there are lights shining outside."
#        "C: The root cause of the anomalies in the video can be attributed to the actions of the man in the yellow jacket. His decision to use a gun to intimidate the man with the cigarettes is a significant deviation from the norm, as guns are typically associated with violence and criminal activity."
#        "D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter."
#        "output: [A,B,C,D]",

conv_video_chatgpt_reason = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, the video showcases the entire process of an anomaly event."
           "Here are several answers which are descriptions of the root cause of this anomaly event. Please rank the answers in the following choices according to their correctness. "
            "Note: Each answer will start with a capital letter and a colon. The final output format will be a string of uppercase letters separated by commas, with the more correct answers appearing earlier.",
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(
        ("USER", '''Here are three answers (A,B,C) which are descriptions of the root cause of this anomaly event. Please rank the answers in the following choices according to their correctness. Also give the reason for your ranking. 
        A: A masked robber in a pullover entered a grocery store, pointed a gun at the cashier and carried out a robbery, threatening to kill him"
        B: First, we see a man standing in front of a convenience store at night. He is wearing a blue shirt and has a blue hoodie on. Next, we see a man standing in front of a convenience store at night. The store is dark, and there are lights shining outside.
        C: The root cause of the anomalies in the video can be attributed to the actions of the man in the yellow jacket. His decision to use a gun to intimidate the man with the cigarettes is a significant deviation from the norm, as guns are typically associated with violence and criminal activity.
         '''),
        ("ASSISTANT",
         "Rank: C,B,A. Because C is the most reasonable answer with some details. B's logic is good but loss some detail. A has base logic with some wrong details")
    ),
    #D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter.  # but B has worse logic with wrong details.
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)


mmeval_des_score_v1 = Conversation(
    system="You are MMEval, a large vision-language model Evaluation assistant. "
           "You are able to understand the video content, the video showcases the entire process of an anomaly event."
           "Also you can understand the model answer and the reference answer and give a score to evaluate the correctness of the model answer. "
           "Here will be an [model's answer] which is a description of an anomaly event. Please compare it with the [reference answer] and refer to the events in the video,"
           " then give the [model's answer] a score in 0 to 10 to evaluate the correctness. "
            '''You need to evaluate the answer to this model in several ways, with 2 marks for each: [Consistency],[Details],[Relevance],[Temporal Coherence],[Anomaly recognition].
            1. [Consistency]: if the subject of the event matches exactly.
            2. [Details]: description of the details and correct
            3. [Relevance]: description of the information are helpful to understand the event
            4. [Temporal Coherence]: Answer describes events in chronological order, does not jump or confuse, and is not fully integrated.
            5. [Anomaly recognition]: The model is describing an anomalous event. rather than an unimportant event in the video.    
            If the subject of the event is just wrong, then the score should be lowered from the standard base.
            You need to give marks after evaluating the five areas in the following format: Score: x/10''',
        # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(
        ("USER", '''[Reference: A white pickup truck was rear-ended by a white SUV behind it on the roadway, and the pickup truck slammed left into a guardrail, followed by a gray car that lost control and slammed into the guardrail}],\n
        [Model answer:There's a lot of collisions on the road. Vehicles are stopped in the middle of the road. ]'''),
        ("ASSISTANT",
         "Score: 5/10. [Consistency]: 1/2, [Details]: 1/2,[Relevance]: 2/2,[Temporal Coherence]: 0/2, [Anomaly recognition]: 1/2"),
        ("USER",'''[Reference: A white pickup truck was rear-ended by a white SUV behind it on the roadway, and the pickup truck slammed left into a guardrail, followed by a gray car that lost control and slammed into the guardrail],\n
        [Model answer: There's a lot of cars on the road. It's crowded.]'''),
        ("ASSISTANT",
        "Score: 1/10. [Consistency]: 0/2, [Details]: 0/2,[Relevance]: 1/2,[Temporal Coherence]: 0/2, [Anomaly recognition]: 0/2"),
        ("USER",'''[Reference: A white pickup truck was rear-ended by a white SUV behind it on the roadway, and the pickup truck slammed left into a guardrail, followed by a gray car that lost control and slammed into the guardrail],\n
        [Model answer:There was an anomaly in the road, and the anomaly was pedestrians walking on the road ]'''),
        ("ASSISTANT",
        "Score: 3/10. [Consistency]: 1/2, [Details]:0/2,[Relevance]: 1/2,[Temporal Coherence]: 0/2, [Anomaly recognition]: 1/2"),
        ("USER",'''[Reference: A white pickup truck was rear-ended by a white SUV behind it on the roadway, and the pickup truck slammed left into a guardrail, followed by a gray car that lost control and slammed into the guardrail],\n
        [Model answer: There was a rear-end collision with a vehicle on the road, and the vehicle collision damaged the vehicle and the guardrail]'''),
        ("ASSISTANT",
        "Score: 8/10. [Consistency]: 2/2, [Details]:1/2,[Relevance]: 2/2,[Temporal Coherence]: 1/2, [Anomaly recognition]: 2/2"),
        ("USER",'''[Reference: A white pickup truck was rear-ended by a white SUV behind it on the roadway, and the pickup truck slammed left into a guardrail, followed by a gray car that lost control and slammed into the guardrail],\n
        [Model answer:A vehicle was involved in a rear-end collision on the roadway and a pickup truck rear-ended the vehicle before changing direction and hitting a guardrail.]'''),
        ("ASSISTANT",
        "Score: 10/10. [Consistency]: 2/2, [Details]:2/2,[Relevance]: 2/2,[Temporal Coherence]: 2/2, [Anomaly recognition]: 2/2"),
    ),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)

mmeval_cause_score = Conversation(
    system="You are MMEval, a large vision-language model Evaluation assistant. "
           "You are able to understand the video content, the video showcases the entire process of an anomaly event."
           "Also you can understand the model answer and the reference answer and give a score to evaluate the correctness of the model answer. "
           "Here will be an [model's answer] which are the root cause and effect of an anomaly event. Please compare it with the [reference answer] and refer to the events in the video,"
           " then give the [model's answer] a score in 0 to 10 to evaluate the correctness of this causal reasoning. "
            '''You need to evaluate the answer to this model in several ways, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
            1. [Consistency]: The cause described are accurately aligned with the video and reference answers
            2. [Causal Explanation]: The answer explains clearly and in detail the root cause of the anomaly event, contains common sense reasoning
            3. [Evidence Support]: The answer is based on strong evidence or sound reasoning that allows for a convincing acceptance of the reasons given.
            4. [Logical Structure]: Answers are presented in a clear logical structure so that the reasons for the occurrence of unusual events are presented in a logical order.
            5. [Clarity]: Reasoning should be concise and strong, not long-winded assumptions. 
            Marks should be reduced appropriately if the answer is very lengthy and heavily speculative.
            You need to give marks after evaluating the five areas in the following format: Score: x/10''',
        # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(
        ("USER", '''[Reference: In an unmanned self-service store in Chengdu, five children carrying backpacks entered the store to destroy the self-service water dispenser and obtain drinks inside, then shared the loot and left],\n
        [Model answer: {A few men entered a store}]'''),
        ("ASSISTANT",
         "Score: 3/10. [Consistency]: 1/2, [Causal Explanation]: 0/2,[Evidence Support]: 1/2,[Logical Structure]: 0/2, [Clarity]: 1/2"),

        ("USER",'''[Reference: In an unmanned self-service store in Chengdu, five children carrying backpacks entered the store to destroy the self-service water dispenser and obtain drinks inside, then shared the loot and left],\n
        [Model answer: Several men entered a store and took some merchandise, they broke the door of the store, which is theft, so it caused an anomaly]'''),
        ("ASSISTANT",
         "Score: 10/10. [Consistency]: 2/2, [Causal Explanation]: 2/2,[Evidence Support]: 2/2,[Logical Structure]: 2/2, [Clarity]: 2/2"),

        ("USER",'''[Reference: In an unmanned self-service store in Chengdu, five children carrying backpacks entered the store to destroy the self-service water dispenser and obtain drinks inside, then shared the loot and left],\n
        [Model answer: Several people go shopping and then leave the store, the exception may be because they had an argument with the owner of the store, or the owner sold goods that were not what they expected, it is not correct to sell bad goods, the exception is because of the goods that were sold or the argument between them]'''),
        ("ASSISTANT",
         "Score: 2/10. [Consistency]: 0/2, [Causal Explanation]:0/2,[Evidence Support]: 0/2,[Logical Structure]: 1/2, [Clarity]: 1/2"),

        ("USER",'''[Reference: In an unmanned self-service store in Chengdu, five children carrying backpacks entered the store to destroy the self-service water dispenser and obtain drinks inside, then shared the loot and left],\n
        [Model answer: A couple of guys went to the store to get their stuff and then left the store, the door to the store was broken and they went right into the store]'''),
        ("ASSISTANT",
         "Score: 6/10. [Consistency]: 1/2, [Causal Explanation]: 1/2,[Evidence Support]: 1/2,[Logical Structure]: 1/2, [Clarity]: 2/2"),
    ),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)

mmeval_result_score = Conversation(
    system="You are MMEval, a large vision-language model Evaluation assistant. "
           "You are able to understand the video content, the video showcases the entire process of an anomaly event."
           "Also you can understand the model answer and the reference answer and give a score to evaluate the correctness of the model answer. "
           "Here will be an [model's answer] which is the final result of an anomaly event. Please compare it with the [reference answer] and refer to the events in the video,"
           " then give the [model's answer] a score in 0 to 10 to evaluate the correctness of the reasoning and summarization of the results of this anomaly event. "
            '''You need to evaluate the answer to this model in several ways, with 2 marks for each: [Consistency],[Causal Explanation],[Evidence Support],[Logical Structure],[Clarity].
            1. [Consistency]: The result described are accurately aligned with the video and reference answers
            2. [Causal Explanation]: The answer explains clearly and in detail the final result of the anomaly event, covering of all serious consequences of accidents
            3. [Evidence Support]: The answers refer to some of the results and impacts that have actually occurred
            4. [Logical Structure]: Answers are presented in a clear logical structure so that the results of the anomalous event are presented in a logical sequence
            5. [Clarity]: Outcomes and impacts should be clear, not generalized
            You need to give marks after evaluating the five areas in the following format: Score: x/10''',
        # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(
        ("USER", '''[Reference: The shooting process record shows that the man graffiti on the white door with red spray paint, with the content of PioDE. After the graffiti was completed, the man left the scene. ],\n
        [Model answer: The result of this anomaly is damage to the homeowner's property. ]'''),
        ("ASSISTANT",
         "Score: 3/10.[Consistency]: 1/2, [Causal Explanation]: 0/2,[Evidence Support]: 1/2,[Logical Structure]: 0/2, [Clarity]: 1/2"),
        ("USER",'''[Reference: The shooting process record shows that the man graffiti on the white door with red spray paint, with the content of PioDE. After the graffiti was completed, the man left the scene. ],\n
        [Model answer: The exterior of the home was harmed by the man and the homeowner received financial damages as a result.]'''),
        ("ASSISTANT",
        "Score: 5/10.[Consistency]: 1/2, [Causal Explanation]: 1/2,[Evidence Support]: 1/2,[Logical Structure]: 1/2, [Clarity]: 1/2"),
        ("USER",'''[Reference: The shooting process record shows that the man graffiti on the white door with red spray paint, with the content of PioDE. After the graffiti was completed, the man left the scene. ],\n
        [Model answer: The white wall turns red when sprayed by the man, causing an unusual event the man escapes.]'''),
        ("ASSISTANT",
        "Score: 10/10. [Consistency]: 2/2, [Causal Explanation]: 2/2,[Evidence Support]: 2/2,[Logical Structure]: 2/2, [Clarity]: 2/2"),
        ("USER",'''[Reference: The shooting process record shows that the man graffiti on the white door with red spray paint, with the content of PioDE. After the graffiti was completed, the man left the scene. ],\n
        [Model answer: A man goes over a wall and hits a sidewalker, and the man falls to the ground.]'''),
        ("ASSISTANT",
        "Score: 0/10. [Consistency]: 0/2, [Causal Explanation]: 0/2,[Evidence Support]: 0/2,[Logical Structure]: 0/2, [Clarity]: 0/2"),
    ),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)

conv_cuva_des = Conversation(
    system='''You are CVUA, an anomaly detection assistant. You possess advanced image and video analysis capabilities, 
    enabling you to identify and describe unconventional events in videos. When asked about the content of a video, 
    your task is to carefully observe the events within the video, particularly those that deviate from the norm. 
    In responding to questions, you should focus on describing the abnormal behaviors, objects, or situations observed, 
    rather than providing reasons or background explanations for these anomalies. Ensure that your responses are detailed, 
    accurate, and directly related to the specific details observed in the video. You need to consider the following categories of anomalies, 
    check if the video contains one of them.
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
    ''',
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(),
    #D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter.  # but B has worse logic with wrong details.
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)

conv_cuva_cause = Conversation(
    system='''You are CVUA, an anomaly detection assistant. You possess advanced image and video analysis capabilities, 
    enabling you to identify and describe unconventional events in videos. When asked about the content of a video, 
    your task is to carefully observe the events within the video, particularly those that deviate from the norm. 
    'What are the reasons for the anomalies appearing in the video? When analyzing this video, 
    please focus directly on providing the causes of the abnormal events. Based on your observations of the video content, 
    use logical reasoning to determine the specific reasons behind these anomalies. Your task is to build a reasonable inference 
    process starting from the clues and details in the video, clearly indicating which factors have led to the observed abnormalities.
     Please ensure that your response is logically rigorous and directly related to the abnormal events in the video and the potential 
     reasons behind them.
    ''',
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(),
    #D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter.  # but B has worse logic with wrong details.
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)

conv_cuva_result = Conversation(
    system='''You are CVUA, an anomaly detection assistant. You possess advanced image and video analysis capabilities, 
    enabling you to identify and describe unconventional events in videos. When asked about the content of a video, 
    your task is to carefully observe the events within the video, particularly those that deviate from the norm. 
    Please, as a vision model, observe the abnormal events in this video and describe in detail the specific results caused 
    by these events. Your description needs to clearly show the direct connection between the abnormal events and their outcomes, 
    including impacts on the environment, people, or the overall situation. Ensure that your response is structurally clear and 
    specifically detailed, directly related to the events in the video, in order to meet high evaluation standards. 
    Your description should be as clear and specific as possible, avoiding generalities and focusing directly on the 
    video rather than summarizing the impact of a type of event on society. Try to describe some of the severe consequences
     that are clearly presented in the video.''',
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=(),
    #D: The man who had bought an umbrella from the store was not wearing it on his back when he made his way to the cashier counter.  # but B has worse logic with wrong details.
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="###",
    sep2="</s>",
)

default_conversation = conv_video_chatgpt_reason
conv_templates = {
    "default": conv_v1_2,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "video-chatgpt-cause": conv_video_chatgpt_reason,
    "video-chatgpt-result": conv_video_chatgpt_result,
    "MMEval_des_v1": mmeval_des_score_v1,
    "MMEval_cause": mmeval_cause_score,
    "MMEval_result": mmeval_result_score,
    "cuva_des" :conv_cuva_des,
    "cuva_cause" :conv_cuva_cause,
    "cuva_result":conv_cuva_result,
    "conv_video_chatgpt_v1": conv_video_chatgpt_v1,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
