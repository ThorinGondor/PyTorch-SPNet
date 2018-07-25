# coding='utf-8'
import os
import sys
import numpy as np
import logging
import cv2
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn

''' 配置表 '''
yolo_class_num = 80
trained_model_dir = "../weights/model@0_3000.pth"
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes_name_path = "../data/coco.names"
img_h, img_w = 416, 416

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
''' 配置表 '''

from nets.model_main import ModelMain
from nets.YOLO_loss import YOLOLoss
from common.utils import non_max_suppression

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

def test():
    is_traning = False # 不训练，测试
    # Load and initialize network
    net = ModelMain(is_training=is_traning)
    net.train(is_traning)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if trained_model_dir:
        logging.info("load checkpoint from {}".format(trained_model_dir))
        state_dict = torch.load(trained_model_dir)
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(anchors[i], yolo_class_num, (img_h, img_w)))

    # prepare the images path
    images_name = os.listdir("./images/")
    images_path = [os.path.join("./images/", name) for name in images_name]
    print('images_name：', images_name)
    print('images_path：', len(images_path), images_path)
    if len(images_path) == 0:
        raise Exception("no image found in {}".format("./images/"))

    # Start inference
    batch_size = 16
    for step in range(0, len(images_path), batch_size): # range(0, 4, 16) step = 0, 4, 8, 12
        logging.info('Batch_size:{}'.format(batch_size))
        # preprocess
        images = [] # 输入网络图片组
        images_origin = [] # 原始图片组
        for path in images_path[step*batch_size:(step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            logging.info(" √ Successfully Processed! √")
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image) # 预处理完毕，加入原始图片组
            # 进一步将图片处理为网络可以接受的数据类型（resize、归一化等）
            image = cv2.resize(image, (img_h, img_w), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image) # 归一化完毕，加入输入网络图片组
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        logging.info("\nImages Convert to Tensor of CUDA Done!")
        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(
                               prediction=output, num_classes=yolo_class_num, conf_thres=0.5)
        logging.info("\nNet Detection Done!\n")

        # write result images: Draw BBox
        classes = open(classes_name_path, 'r').read().split("\n")[:-1] # 读取coco.names
        if not os.path.isdir("./output/"):
            os.makedirs("./output/")
        for idx, detections in enumerate(batch_detections):
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin[idx])
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                # print('Final Detections: ', detections)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = img_h, img_w  # 416, 416
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                             edgecolor=color,
                                             facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                             verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('output/{}_{}.jpg'.format(step, idx), bbox_inches='tight', pad_inches=0.0)
            plt.close()
    logging.info("All the Test Process Succeed! Enjoy it!")


test()
