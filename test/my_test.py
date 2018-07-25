import os
import sys
import numpy as np
import logging
import cv2
import random

import torch
import torch.nn as nn


''' 配置表 '''
yolo_class_num = 80
pretrained_model_dir = "../weights/yolov3_weights_pytorch.pth"
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes_name_path = "../data/coco.names"
img_h, img_w = 416, 416

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.YOLO_loss import YOLOLoss
from common.utils import non_max_suppression

def MyTest():
    net = ModelMain(is_training=False)
    net.train(False)

    net = nn.DataParallel(net)
    net = net.cuda()

    state_dict = torch.load(pretrained_model_dir)
    net.load_state_dict(state_dict)

    print('\n - - - - - - MyTest Run! - - - - - - \n')

MyTest()