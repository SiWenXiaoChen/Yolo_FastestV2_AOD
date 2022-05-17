import os
import cv2
import time
import argparse

import numpy as np
import torch
from torch import nn

import model.detector
import utils.utils

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/traffic_haze.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='weights/traffic_haze-280-epoch-0.748718ap-model.pth',
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='img/frame_011159.jpg',
                        help='The path of test image')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device), False)


    # sets the module in eval node
    model.eval()

    # 数据预处理
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # 模型推理
    start = time.perf_counter()
    preds, clean_img = model(img)
    clean_img = clean_img.detach().numpy().reshape(3, 352, 352).transpose(1, 2, 0)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time:%fms" % time)

    # 特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

    # 加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    # 绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("test_result.png", clean_img)
