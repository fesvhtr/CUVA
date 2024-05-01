import torch
from blip2 import Blip2
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_coarse_frames, generate_fine_frames, cal_img_emb, cal_text_emb, find_peak, draw_curve


if __name__ == '__main__':
    params = []   # for draw curve
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    src = r'/home/dh/pythonProject/AnomalyDataset/Data/00038.mp4'
    dest = r'/home/dh/pythonProject/AnomalyDataset/Data/capture_image/'

    keywords = ["Several cars collided, causing chaos", " A rear-end collision occurred", "One car hit", "emitting smoke"]
    score = torch.tensor([0.5, 0.4, 0.3, 0.2]).to(device)

    saved_frames_list1, frame_all_num = generate_coarse_frames(src, dest)
    # img_emb = cal_img_emb(dest)
    # text_emb = cal_text_emb(keywords)
    # similarity = cosine_similarity(text_emb.detach().cpu(), img_emb.detach().cpu())
    # similarity = torch.tensor(similarity).to('cuda')
    # score1 = score.unsqueeze(1)
    # imp_value_0 = torch.sum(similarity * score1, dim=0)
    #
    # coarse_peak_list = find_peak(imp_value_0.detach().cpu().numpy())
    # coarse_peak_frames = [saved_frames_list1[i] for i in coarse_peak_list]
    # params_coarse = {'data_y': imp_value_0.detach().cpu().numpy(), 'peak_frames': coarse_peak_frames, 'data_x': saved_frames_list1, 'peak_list': coarse_peak_list, 'all_video_frames': frame_all_num, 'title': 'coarse sampling'}
    # params.append(params_coarse)
    # draw_curve(params)
    #
    # saved_frames_list2 = generate_fine_frames(src, dest, coarse_peak_list, saved_frames_list1)   # 细采样阶段
    #
    # img_emb = cal_img_emb(dest)
    # text_emb = cal_text_emb(keywords)
    # similarity = cosine_similarity(text_emb.detach().cpu(), img_emb.detach().cpu())
    # similarity = torch.tensor(similarity).to('cuda')
    # score2 = score.unsqueeze(1)
    #
    # imp_value_1 = torch.sum(similarity * score2, dim=0)
    # fine_peak_list = find_peak(imp_value_1.detach().cpu().numpy())
    # fine_peak_frames = [saved_frames_list2[i] for i in fine_peak_list]
    # params_fine = {'data_y': imp_value_1.detach().cpu().numpy(), 'peak_frames': fine_peak_frames, 'data_x': saved_frames_list2, 'peak_list': fine_peak_list, 'all_video_frames': frame_all_num, 'title': 'fine sampling'}
    # params.append(params_fine)
    # draw_curve(params)


    # # for test
    # # imp_value_1 = torch.tensor([0.3462, 0.3455, 0.3527, 0.3466, 0.3443, 0.3565, 0.3709, 0.3523, 0.3421,
    # #     0.3483, 0.3551, 0.3605, 0.3587, 0.3590, 0.3526, 0.3560, 0.3605, 0.3555,
    # #     0.3547, 0.3530, 0.3498, 0.3591, 0.3586, 0.3560, 0.3457, 0.3398, 0.3310,
    # #     0.3412, 0.3385, 0.3304, 0.3402, 0.3613, 0.3544, 0.3453, 0.3502, 0.3668,
    # #     0.3481, 0.3603, 0.3537, 0.3434, 0.3556, 0.3501, 0.3496, 0.3540, 0.3590,
    # #     0.3566, 0.3523, 0.3596, 0.3440, 0.3420, 0.3380, 0.3424, 0.3516, 0.3518,
    # #     0.3738, 0.3573, 0.3541, 0.3535, 0.3642, 0.3626, 0.3651, 0.3497, 0.3474,
    # #     0.3353, 0.3371, 0.3408, 0.3485, 0.3464, 0.3417, 0.3435, 0.3556, 0.3569,
    # #     0.3564, 0.3560, 0.3487, 0.3469, 0.3460, 0.3483, 0.3467, 0.3541, 0.3498,
    # #     0.3515, 0.3508, 0.3559, 0.3554, 0.3523, 0.3482, 0.3571, 0.3442, 0.3457,
    # #     0.3521, 0.3535, 0.3388, 0.3598, 0.3528, 0.3556, 0.3422, 0.3278, 0.3271])
    # # fine_peak_list = [2, 6, 11, 13, 16, 21, 31, 35, 37, 40, 44, 47, 54, 58, 60, 71, 79, 81, 83, 87, 91, 93, 95]
    # # fine_peak_frames = [66, 90, 120, 133, 150, 180, 456, 480, 490, 510, 560, 573, 643, 663, 676, 866, 936, 950, 960, 983, 1010, 1020, 1030]
    #
    # # 寻找单峰, 测试时imp_value_1是numpy类型的array
    # single_peaks = []
    # single_peaks_frames = []
    # imp_value_1_tonumpy = imp_value_1.detach().cpu().numpy()
    # peak_values = [imp_value_1_tonumpy[i] for i in fine_peak_list]
    # for idx, value in enumerate(peak_values):
    #     # todo:开头和结尾还没有做特殊处理
    #     if 0 < idx < len(peak_values)-1:
    #         prefix, latter = peak_values[idx-1], peak_values[idx+1]
    #         if (value - prefix) > 0.005 and (value - latter) > 0.005:
    #             single_peaks_frames.append(fine_peak_frames[idx])
    #             dict_to_save = {"idx": fine_peak_list[idx], "value": value, "frame": fine_peak_frames[idx]}
    #             single_peaks.append(dict_to_save)
    #
    # # 合并入原曲线内
    # for d in single_peaks:
    #     if d["frame"] in saved_frames_list1:
    #         continue
    #     saved_frames_list1.append(d["frame"])
    #     saved_frames_list1.sort()
    #     index = saved_frames_list1.index(d["frame"])
    #     imp_value_0 = torch.cat([imp_value_0[0:index], torch.tensor([d["value"]]).to("cuda:0"), imp_value_0[index:]])
    # coarse_peak_list = find_peak(imp_value_0.detach().cpu().numpy())
    # coarse_peak_frames = [saved_frames_list1[i] for i in coarse_peak_list]
    # # draw(imp_value_0, coarse_peak_list)
    #
    # # 寻找峰群
    # peak_group_frames = []
    # temp = []
    # fine_peak_frames_without_singlepeak = [i for i in fine_peak_frames if i not in single_peaks_frames]
    #
    #
    # for idx, value in enumerate(fine_peak_frames_without_singlepeak):
    #     if idx == 0:
    #         temp.append(value)
    #         continue
    #     elif idx == len(fine_peak_frames_without_singlepeak)-1:
    #         if (value - temp[-1]) < 30:
    #             temp.append(value)
    #             peak_group_frames.append(temp)
    #         else:
    #             if len(temp) == 1:
    #                 break
    #             else:
    #                 peak_group_frames.append(temp)
    #
    #     else:
    #         if (value - temp[-1]) < 30:
    #             temp.append(value)
    #         else:
    #             if len(temp) == 1:
    #                 temp = [value]
    #             else:
    #                 peak_group_frames.append(temp)
    #                 temp = [value]
    # # 算group的平均值
    # group_num = len(peak_group_frames)
    # group_mean = []
    #
    # for sub_list in peak_group_frames:
    #     temp = 0
    #     for item in sub_list:
    #         index = fine_peak_frames.index(item)
    #         temp += imp_value_1_tonumpy[index]
    #     mean = temp / len(sub_list)
    #     group_mean.append(mean)
    #
    # # 对应回原曲线中
    # for idx, group in enumerate(peak_group_frames):
    #     for frame in group:
    #         if frame in saved_frames_list1:
    #             continue
    #         else:
    #             saved_frames_list1.append(frame)
    #         saved_frames_list1.sort()
    #         index = saved_frames_list1.index(frame)
    #         imp_value_0 = torch.cat([imp_value_0[0:index], torch.tensor([group_mean[idx]]).to("cuda:0"), imp_value_0[index:]])
    #
    #
    # coarse_peak_list = find_peak(imp_value_0.detach().cpu().numpy())
    # coarse_peak_frames = [saved_frames_list1[i] for i in coarse_peak_list]
    # draw_test(imp_value_0.detach().cpu().numpy(), imp_value_1.detach().cpu().numpy(), coarse_peak_frames,
    #           fine_peak_frames, saved_frames_list1, saved_frames_list2, coarse_peak_list, fine_peak_list, 1067)





















































    # img_src = r"/home/dh/pythonProject/AnomalyDataset/Data/capture_image"
    # imgs = os.listdir(img_src)
    # imgs.sort(key=lambda x: int(x[0:-4]))
    # captions = []
    #
    # model = Blip2('FlanT5 XL', 0)
    # info_list = []
    # for idx, img_url in enumerate(imgs):
    #     if idx in peak_list:
    #         img = Image.open(os.path.join(img_src, img_url))
    #         caption = model.caption(img)
    #         captions.append(caption)
    #         # print(idx, img_url, caption)
    #         info_list.append({'idx': idx, 'img_url': img_url, 'caption': caption})
    #
    # caption_text_embed, _ = cal_text_emb(captions)  # shape: [8, 512]
    # cap_kw_similarity = cosine_similarity(caption_text_embed.detach().cpu(), text_emb.detach().cpu())   # shape:[8, 5], type: ndarray
    # rel_kw_idx = cap_kw_similarity.argmax(axis=1)
    # print(cap_kw_similarity)
    # for idx, d in enumerate(info_list):
    #     d.update({"kw": text_list[rel_kw_idx[idx]]})
    #
    # for d in info_list:
    #     print(d.values())

    # print(cap_kw_similarity)
    # print(cap_kw_similarity.max(axis=1))
    # print(cap_kw_similarity.argmax(axis=1))

