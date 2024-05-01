import torch
import matplotlib.pyplot as plt
import shutil
import os
import cv2
import numpy as np
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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
    text_emb = model.encode_text(text)
    return text_emb

def draw(importance_value, peak_list):
    x = torch.linspace(0, importance_value.shape[0], steps=importance_value.shape[0])
    y = importance_value.detach().cpu()
    # plt.scatter(peak_list, y[peak_list], c='r')

    plt.plot(x, y)
    plt.show()

def draw_curve(params):
    fig_num = len(params)
    fig, axs = plt.subplots(fig_num, 1, sharex=True, sharey=True)
    if fig_num == 1:
        axs = [axs]
    for idx, param in enumerate(params):
        x = [i/param['all_video_frames'] for i in param['data_x']]
        px = [i/param['all_video_frames'] for i in param['peak_frames']]
        axs[idx].plot(x, param['data_y'])
        axs[idx].scatter(px, param['data_y'][param['peak_list']], c='r')
        axs[idx].set_title(param['title'])
    plt.show()

def draw_one_curve(data_y, peak_frames, data_x, peak_list, all_video_frames):
    x = [i/all_video_frames for i in data_x]
    px = [i/all_video_frames for i in peak_frames]
    plt.plot(x, data_y)
    plt.scatter(px, data_y[peak_list], c='r')
    plt.show()

def draw_test(data1_y, data2_y, peak_frames1, peak_frames2, data1_x, data2_x, peak_list1, peak_list2, all_video_frames):
    x1 = [i/all_video_frames for i in data1_x]
    x2 = [i/all_video_frames for i in data2_x]
    px1 = [i/all_video_frames for i in peak_frames1]
    px2 = [i/all_video_frames for i in peak_frames2]
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    axs[0].plot(x1, data1_y)
    axs[0].scatter(px1, data1_y[peak_list1], c='r')
    axs[0].set_title('coarse sampling')

    axs[1].plot(x2, data2_y)
    axs[1].scatter(px2, data2_y[peak_list2], c='r')
    axs[1].set_title('fine sampling')

    plt.show()



def find_peak(importance_value, threshold=0.35):     # todo:后面看用什么方式确定一下threshold
    peak_list = []   # 选出的符合条件的波峰
    previous, present, latter = 0, 0, 0
    for idx, value in enumerate(importance_value):
        if idx != len(importance_value)-1:
            present, latter = value, importance_value[idx+1]
            if present > previous and present > latter and present > threshold:
                peak_list.append(idx)
            previous = present
        else:   # 最后一个value要对数组越界做一些处理
            present = value
            if present > previous and present > threshold:
                peak_list.append(idx)
    return peak_list

def generate_coarse_frames(video_src, dest, frame_interval=30):
    if os.path.exists(dest):
        # 使用shutil.rmtree删除目标文件夹及其内容，然后使用os.mkdir重新创建该文件夹。
        shutil.rmtree(dest)
    os.mkdir(dest)

    count = 0  # 抽取帧数
    vc = cv2.VideoCapture(video_src)
    rval, frame = vc.read()  # 初始化,并读取第一帧,rval表示是否成功获取帧,frame是捕获到的图像
    fps = vc.get(cv2.CAP_PROP_FPS)      # 获取视频fps
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取每个视频帧数
    # 获取视频总帧数
    print("[INFO] 视频粗采样阶段")
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    # 统计当前帧
    frame_count = 1
    saved_frame_list = []

    while rval:
        rval, frame = vc.read()
        # 隔n帧保存一张图片
        if frame_count % frame_interval == 0:
            # 当前帧不为None，能读取到图片时

            if frame is not None:
                path = os.path.join(dest, str(frame_count) + '.jpg')
                cv2.imwrite(path, frame)
                count += 1
                saved_frame_list.append(frame_count)
        frame_count += 1
    vc.release()  # 关闭视频文件
    print("[INFO] 粗采样阶段总共抽帧：{}张图片\n".format(count))
    return saved_frame_list, frame_all


def generate_fine_frames(video_src, dest, peak_list, saved_frames_list, peak_time_interval=1, dense_sample=5):
    if os.path.exists(dest):
        # 使用shutil.rmtree删除目标文件夹及其内容，然后使用os.mkdir重新创建该文件夹。
        shutil.rmtree(dest)
    os.mkdir(dest)

    count = 0  # 抽取帧数
    vc = cv2.VideoCapture(video_src)
    rval, frame = vc.read()  # 初始化,并读取第一帧,rval表示是否成功获取帧,frame是捕获到的图像
    fps = vc.get(cv2.CAP_PROP_FPS)      # 获取视频fps
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取每个视频帧数
    # 获取视频总帧数
    print("[INFO] 视频细采样阶段")
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    # 统计当前帧
    frame_count = 1

    selected_frames = []

    for peak in peak_list:
        pre_bound, post_bound = max(0, (peak + 1) * 30 - peak_time_interval * 30), min((peak + 1) * 30 + peak_time_interval * 30, frame_all)
        samples = np.linspace(pre_bound, post_bound, dense_sample * 2)
        int_samples = [int(i) for i in samples]
        selected_frames += int_samples
    selected_frames += saved_frames_list
    selected_frames = list(set(selected_frames))
    selected_frames.sort()

    while rval:
        rval, frame = vc.read()

        if frame_count in selected_frames:
            path = os.path.join(dest, str(frame_count) + '.jpg')
            cv2.imwrite(path, frame)
            count += 1

        frame_count += 1
    vc.release()  # 关闭视频文件
    print("[INFO] 细采样阶段总共抽帧：{}张图片\n".format(count))
    return selected_frames


if __name__ == '__main__':
    # example_origin = torch.tensor([0.1103, 0.0798, 0.1621, 0.2337, 0.2727, 0.1574, 0.1292, 0.2517, 0.2435,
    #     0.2579, 0.1581, 0.1226, 0.1668, 0.1400, 0.1096, 0.2081, 0.2376, 0.3426,
    #     0.2939, 0.1327, 0.1801, 0.2412, 0.2569, 0.2466, 0.2138, 0.2791, 0.2824,
    #     0.2161, 0.1850, 0.2260, 0.2117, 0.2372, 0.1951, 0.1691, 0.1418])
    # example = example_origin.detach().cpu().numpy()
    # peak_list = find_peak(example)
    # draw(example_origin, peak_list)
    pass
