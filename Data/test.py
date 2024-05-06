# @FileName  :test.py
# @Time      :2023/10/18 10:24
# @Author    :Duh

import json
import os

from decord import VideoReader
from decord import cpu, gpu
from PIL import Image
import h5py

anomaly_train_candidate = {"00016": [
    "In the traffic flow, a motorcycle for two people drove between the two cars.The car on the left began to change lanes to the left",
    "A single ponytail blonde woman wearing a black jumpsuit snipes enemies in the forest with a gun, while a suit bald black man remotely controls a robot to help. Armed helicopters fly over the sky.",
    "Four masked people broke into the room and were discovered by the owner.Thieves break windows, break doors and cameras during the day, and enter the room",
    "A black man found two children in a house when a lion appeared behind him. The man decided to distract the lion to protect the child from being attacked and eaten by the lion"],
    "00041": [
        "A car was driving normally on an uphill highway when the last car on the left suddenly changed lanes and collided directly in front of it",
        "Driving on rainy roads, it rained heavily and the windshield was almost covered with rainwater. At an intersection, a tricycle crosses the road from the sidewalk",
        "A chemical tanker truck was driving in the left lane of the road when suddenly the tire skidded and the truck shifted to the left, causing the vehicle to lose control.",
        "A black man found two children in a house when a lion appeared behind him. The man decided to distract the lion to protect the child from being attacked and eaten by the lion"
    ]}

anomaly_train = [{"video_id": "00016", "question": "What is the reason for the occurrence of this anomaly event?",
                  "answer": "Two cars collide on the highway,The vehicles behind collided with another two vehicles due to a traffic accident ahead.Large scale car collisions have caused other vehicles to detour one after another",
                  "question_id": "CAUSE",
                  "answer_type": "free-text"},
                 {"video_id": "00041", "question": "What is the reason for the occurrence of this anomaly event?",
                  "answer": "Two cars collide on the highway,The vehicles behind collided with another two vehicles due to a traffic accident ahead.Large scale car collisions have caused other vehicles to detour one after another",
                  "question_id": "CAUSE",
                  "answer_type": "free-text"}]

anomaly_test = [{"video_id": "00041", "question": "What is the reason for the occurrence of this abnormal event?",
                 "answer": "A car was driving normally on an uphill highway when the last car on the left suddenly changed lanes and collided directly in front of it",
                 "question_id": "CAUSE",
                 "answer_type": "free-text"}]  # val.json

anomaly_test_candidate = {"00041": [
    "The open space was filled with red firecrackers, and the square was surrounded by a group of people holding mobile phones to shoot. Two men lit the wires of the firecrackers and began.",
    "When the vehicle suddenly turns at the intersection, the opposing motorcycle is frightened and severely affected, resulting in a collision between the car and motorcycle",
    "The movie stars, including men and women, and those who cooperate with men and women, smoke in various scenes, with smoke drifting from their hands, flowing slowly from their mouths, and exchanging somke with each other",
    "A white man wearing a white cowboy hat and glasses climbed onto a truck and threw garbage into a bucket surrounded by an iron mesh"
]}  # val_candidate.json

if __name__ == "__main__":
    # with open("/home/dh/pythonProject/AnomalyDataset/mist/data/datasets/anomaly/anomaly_train_candidates.json",
    #           "w") as f:
    #     json.dump(anomaly_train_candidate, f)
    #
    # with open("/home/dh/pythonProject/AnomalyDataset/mist/data/datasets/anomaly/anomaly_train.json", "w") as f:
    #     json.dump(anomaly_train, f)
    #
    # with open("/home/dh/pythonProject/AnomalyDataset/mist/data/datasets/anomaly/anomaly_val.json", "w") as f:
    #     json.dump(anomaly_train, f)
    #
    # with open("/home/dh/pythonProject/AnomalyDataset/mist/data/datasets/anomaly/anomaly_val_candidates.json", "w") as f:
    #     json.dump(anomaly_train_candidate, f)
    #
    # vr = VideoReader("/home/dh/pythonProject/AnomalyDataset/mist/anomaly/00041.mp4", ctx=cpu(0))
    #
    # frame = vr.get_batch([1])  # get the first frame
    #
    # anomaly_frame_size = {"00016": {"width": frame.shape[2], "height": frame.shape[1]},  # frame.shape (1, 720, 1280, 3)
    #                       "00041": {"width": frame.shape[2], "height": frame.shape[1]}}
    #
    # with open("/home/dh/pythonProject/AnomalyDataset/mist/data/datasets/anomaly/anomaly_frame_size.json", "w") as f:
    #     json.dump(anomaly_frame_size, f)
    # def getDataFromH5py(fileName, target, start, length):
    #     with h5py.File(fileName, 'r') as h5f:
    #         if not h5f.__contains__(target):
    #             res = []
    #         elif (start + length >= h5f[target].shape[0]):
    #             res = h5f[target].value[start:h5f[target].shape[0]]
    #         else:
    #             res = h5f[target].value[start:start + length]
    #     return res

    json_file = "/home/dh/pythonProject/AnomalyDataset/train_mist_v1/Causev1.json"

    encoding = 'utf-8'

    frame_feats = {}

    # with h5py.File(app_feat_file, 'r') as fp:
    #     vids = fp['ids']
    #     feats = fp['features']
    #     print(feats.shape)  # v_num, clip_num, feat_dim
    #     for id, (vid, feat) in enumerate(zip(vids, feats)):
    #         vid = vid.decode(encoding)
    #         frame_feats[vid] = feat
    #
    # for i in range(1, 100):
    #     try:
    #         feature = frame_feats["%05d" % (i)]
    #     except Exception:
    #         print("%05d" % (i))
    #
    # app_feat_dir = "/home/dh/pythonProject/AnomalyDataset/mist/data/feats/anomaly"
    # frame_feats = {}
    # for fp in os.listdir(app_feat_dir):
    #     if fp.split('_')[-1] in ("100", "200", "300", "400", "500", "all"):
    #         continue
    #     filepath = os.path.join(app_feat_dir, fp)
    #     file = h5py.File(filepath, 'r')
    #     vid, feat = file['ids'].decode(encoding), file['features']
    #     frame_feats = {vid: feat}
