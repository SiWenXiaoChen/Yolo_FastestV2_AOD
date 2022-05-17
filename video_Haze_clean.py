# python
# 陈涛
# 开发时间：$[DATE] $[TIME]
# python
# 陈涛
# 开发时间：$[DATE] $[TIME]
# python
# 陈涛
# 开发时间：$[DATE] $[TIME]
import os
import cv2
import time
import argparse

import numpy as np

import model.detector
import torch
import yuyin
import bofang

import utils.utils

# 指定训练配置文件
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/traffic_haze.data',
                    help='Specify training profile *.data')
parser.add_argument('--weights', type=str, default='weights/traffic_haze-280-epoch-0.748718ap-model.pth',
                    help='The path of the .pth model to be transformed')
parser.add_argument('--img', type=str, default='img/017.jpg', help='The path of test image')
parser.add_argument('--vid', type=str, default='Lar/Video_haze_same.avi', help='The path of test image')


opt = parser.parse_args()
cfg = utils.utils.load_datafile(opt.data)
assert os.path.exists(opt.weights), "请指定正确的模型路径"


# 模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True).to(device)
model.load_state_dict(torch.load(opt.weights, map_location=device), False)

# sets the module in eval node
model.eval()


# 1 加载视频文件
capture = cv2.VideoCapture(opt.vid)
t=0
# 2 读取视频
ret, frame = capture.read()
while ret:
    t = t + 1
    start = time.time()
    # 3 ret 是否读取到了帧，读取到了则为True

    ret, frame = capture.read()

    # 数据预处理
    img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # 模型推理
    preds, clean_img = model(img)
    frame = clean_img.detach().numpy().reshape(3, 352, 352).transpose(1, 2, 0).astype(np.uint8).copy()
    # 特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.4, iou_thres=0.4)

    # 加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = frame.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    # 绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        if category == "Traffic_Light_go":
            cv2.putText(frame, category, (x1, y1 - 25), 0, 0.5, (0, 255, 0), 2)
        elif category == "Traffic_Light_stop":
            cv2.putText(frame, category, (x1, y1 - 25), 0, 0.5, (0, 0, 255), 2)
        elif category == "Traffic_Light_ambiguous":
            cv2.putText(frame, category, (x1, y1 - 25), 0, 0.5, (255, 0, 0), 2)

    end = time.time()
    print("fpn={}".format(int(1/(end-start))))
    cv2.imshow("video", frame)
    # 4 若键盘按下q则退出播放
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# 5 释放资源
capture.release()
# 6 关闭所有窗口
cv2.destroyAllWindows()