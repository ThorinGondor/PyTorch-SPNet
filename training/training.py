import os
import sys
import time
import logging
logging.getLogger().setLevel(logging.INFO)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DATA

# from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__)) # 获取本文件所处的绝对路径，解决相对路径的问题
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from nets.model_main import ModelMain
from nets.YOLO_loss import YOLOLoss
from common.coco_dataset import COCODataset


''' 配置表 '''
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes = 80

batch_size = 10
train_path = "../data/coco/trainvalno5k.txt"
epochs = 100
img_h, img_w = 416, 416
lr_freeze_backbone = False
other_lr = 0.01
backbone_lr = 0.001
optimizer_type = "sgd"
optimizer_weight_decay = 4e-05
lr_decay_step = 20
lr_decay_gamma = 0.1
pretrain_snapshot = ""#"F:/DeepLearningPytorch/M9_YOLOv3_Full_Reproduce/checkpoint/run0/model@0_9000.pth"
evaluate_type = ""
export_onnx = False
# checkpoint_path = "F:/DeepLearningPytorch/M9_YOLOv3_Full_Reproduce/checkpoint/model.pth"
''' 配置表 '''


def train():
    global_step = 0
    is_training = True

    # Load and Initialize Network
    net = ModelMain(is_training)
    net.train(is_training)

    # Optimizer and Lr
    optimizer = _get_optimizer(net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_step, #20
        gamma=lr_decay_gamma) # 0.1

    # Set Data Paraller:
    net = nn.DataParallel(net)
    net = net.cuda()
    logging.info("Net of Cuda is Done!")

    # Restore pretrain model 从预训练模型中恢复
    if pretrain_snapshot:
        logging.info("Load pretrained weights from {}".format(pretrain_snapshot))
        state_dic = torch.load(pretrain_snapshot)
        net.load_state_dict(state_dic)

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(anchors[i], classes, (img_w, img_h)).cuda())
    print('YOLO_Losses: \n', yolo_losses)

    # DataLoader
    train_data_loader = DATA.DataLoader(dataset=COCODataset(train_path, (img_w, img_h), is_training=True),
                                        batch_size=batch_size,shuffle=True, pin_memory=False)
    # Start the training loop
    logging.info("Start training......")
    for epoch in range(epochs):
        for step, samples in enumerate(train_data_loader):
            images, labels = samples['image'].cuda(), samples["label"].cuda()
            start_time = time.time()
            global_step += 1

            # Forward & Backward
            optimizer.zero_grad()
            outputs = net(images)
            losses_name = ["total_loss", "x","y", "w", "h", "conf", "cls"]
            losses = [[]] * len(losses_name) # [[]] ---> [[], [], [], [], [], [], []]
            for i in range(3): # YOLO 3 scales
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    # print('j: ', j, 'l: ', l) j: index(0-6); l内容: 总loss, x, y, w, h, conf, cls
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0] # losses[0]为总Loss
            conf = losses[5]
            loss.backward()
            optimizer.step()

            if step>0 and step % 10 == 0:
                _loss = loss.item()
                _conf = conf.item()
                duration = float(time.time()-start_time) # 总用时
                example_per_second = batch_size / duration # 每个样本用时
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.2f conf = %.2f example/sec = %.3f lr = %.5f " %
                    (epoch, step, _loss, _conf, example_per_second, lr)
                )
            if step >= 0 and step % 1000 == 0:
                # net.train(False)
                _save_checkpoint(net.state_dict(), epoch, step)
                # net.train(True)

        lr_scheduler.step()

    _save_checkpoint(net.state_dict(), 100, 9999)
    logging.info("Bye~")

def _save_checkpoint(state_dict, epoch, step, evaluate_func = None):
    checkpoint_path = '%s%s%s%s%s' % ('F:/DeepLearningPytorch/M9_YOLOv3_Full_Reproduce/checkpoint/model@', str(epoch),'_', str(step), '.pth')
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)

def _get_optimizer(net):
    logging.info("Get optimizer: ")
    optimizer = None

    # Assign different lr for each layer
    params =None
    base_params = list(map(id, net.backbone.parameters()))
    logits_params = filter(lambda p: id(p) not  in base_params, net.parameters())

    if not lr_freeze_backbone: # Default: False
        params = [
            {"params": logits_params, "lr": other_lr}, # 0.01
            {"params": net.backbone.parameters(), "lr": backbone_lr}, # 0.001
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": other_lr},
        ]
    # Initialize optimizer class
    if optimizer_type == "adam":
        optimizer = optim.Adam(params, weight_decay=optimizer_weight_decay)
    elif optimizer_type == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=optimizer_weight_decay, amsgrad=True)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=optimizer_weight_decay)
    else:
        logging.info("Using SGD Optimizer.")
        optimizer = optim.SGD(params, momentum=0.9, weight_decay=optimizer_weight_decay,
                              nesterov=False)
    return optimizer

if __name__ == "__main__":
    train()