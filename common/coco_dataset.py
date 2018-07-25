import os
import numpy as np
import logging
import cv2

import torch
import torch.utils.data as DATA
from torch.utils.data import Dataset

from common import data_transform

class COCODataset(Dataset):
    # 需要传入的参数为："../data/coco/trainvalno5k.txt" (416, 416) True
    def __init__(self, list_path, img_size, is_training, is_debug=False):
        with open(list_path, 'r') as file:
            # 所有参与训练的images的路径 (img_files)：
            self.img_files =file.readlines()
        # 所有参与训练的labels的路径 (label_files)：
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt')
                            .replace('jpg', 'txt') for path in self.img_files]
        # 训练图片的尺寸
        self.img_size = img_size # (w, h) (416, 416)
        self.max_objects = 50
        self.is_debug = is_debug # False

        # data transform and data augmentation
        self.transforms = data_transform.Compose()
        if is_training:
            self.transforms.add(data_transform.ImageBaseAug())
        self.transforms.add(data_transform.ResizeImage(self.img_size)) # eg: 416, 416
        self.transforms.add(data_transform.ToTensor(self.max_objects, self.is_debug)) # 50, False

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip() # 获取图片路径
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # 读取图片
        if img is None:
            raise Exception("Read images error: {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_files)

#  use for test dataloader
if __name__ == "__main__":
    train_data_loader = DATA.DataLoader(
        dataset=COCODataset("../data/coco/trainvalno5k.txt", (416, 416),True, True),
        batch_size=2,
        shuffle=True,
        pin_memory=False)
    for step, sample in enumerate(train_data_loader):
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("step{}_{}.jpg".format(step, i), image)
        # only one batch
        break
