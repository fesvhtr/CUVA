# @FileName  :Clip_curve.py
# @Time      :2023/9/6 15:40
# @Author    :Duh
import torch, clip
import cv2, os, shutil
import matplotlib.pyplot as plt
from PIL import Image
# import openai


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def generate_frames(video_src, frame_interval=30):  # 帧间隔默认30

    dest = r'/home/dh/pythonProject/AnomalyDataset/Data/capture_image'  # todo:不同测试视频文件夹分开
    # 使用shutil.rmtree删除目标文件夹及其内容，然后使用os.mkdir重新创建该文件夹。
    shutil.rmtree(dest)
    os.mkdir(dest)

    count = 0  # 抽取帧数
    vc = cv2.VideoCapture(video_src)
    rval, frame = vc.read()  # 初始化,并读取第一帧,rval表示是否成功获取帧,frame是捕获到的图像
    fps = vc.get(cv2.CAP_PROP_FPS)      # 获取视频fps
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取每个视频帧数
    # 获取视频总帧数
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    # 统计当前帧
    frame_count = 1

    while rval:
        rval, frame = vc.read()
        # 隔n帧保存一张图片
        if frame_count % frame_interval == 0:
            # 当前帧不为None，能读取到图片时

            if frame is not None:
                cv2.imwrite("/home/dh/pythonProject/AnomalyDataset/Data/capture_image/" + str(frame_count) + '.jpg', frame)
                count += 1
        frame_count += 1
    vc.release()  # 关闭视频文件
    print("[INFO] 总共抽帧：{}张图片\n".format(count))
    return dest


def get_gpt_score(text_list=None):
    os.environ["http_proxy"] = "http://127.0.0.1:19180"

    openai.api_key = "YOUR_API_KEY"

    completion = openai.ChatCompletion.create(  # 创建对话
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What should I do if i feel headache"}
        ]
    )
    print(completion.choices[0].message)


def cal_img_emb(img_src):
    imgs = os.listdir(img_src)
    imgs.sort(key=lambda x: int(x[0:-4]))
    img_emb_list = []
    for img in imgs:
        img_emb = preprocess(Image.open(os.path.join(img_src,img))).unsqueeze(0).to(device)
        img_feature = model.encode_image(img_emb)
        img_emb_list.append(img_feature)
    img_emb = torch.stack(img_emb_list, dim=0).squeeze(1)
    return img_emb


def cal_text_emb(text_list):
    text = clip.tokenize(text_list).to(device)
    score = torch.tensor([0.5, 0.4, 0.3, 0.2]).to(device)
    text_emb = model.encode_text(text)
    return text_emb, score


def cal_sim(img_emb, text_emb, score):
    img_emb /= img_emb.norm(dim=-1, keepdim=True)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)
    similarity = (100.0 * img_emb @ text_emb.T).softmax(dim=-1).T
    similarity /= similarity.norm(dim=-1, keepdim=True)
    score = score.unsqueeze(1)   
    similarity = similarity * score
    importance_value = torch.sum(similarity, dim=0)
    return importance_value


def draw(importance_value):
    x = torch.linspace(1, importance_value.shape[0]+1, steps=importance_value.shape[0])
    y = importance_value.detach().cpu()
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    src = r'/home/dh/pythonProject/AnomalyDataset/Data/00016.mp4'   # todo:整个文件夹里只有一个测试视频
    img_emb = cal_img_emb(generate_frames(src))
    text_list = ['Police car comes', 'Crash', 'Smoke', 'People injured', 'Sires sound']
    text_emb, score = cal_text_emb(text_list)
    imp_value = cal_sim(img_emb, text_emb, score)
    draw(imp_value)
    # get_gpt_score()