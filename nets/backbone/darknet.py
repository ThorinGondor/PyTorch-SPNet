import torch
import torch.nn as nn
import math
from collections import OrderedDict

''' (1) Darknet网络结构的最小残差单元 '''
class BasicBlock(nn.Module):
    def __init__(self, in_depth, planes): # planes[0] 隐藏深度\通道数 planes[1] 输出深度\通道数
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_depth, out_channels=planes[0],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes[0])
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2 = nn.Conv2d(in_channels=planes[0], out_channels=planes[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes[1])
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out +=residual
        return out

''' (2) Darknet: 多尺度输出（未合并/CONCAT） & 残差网络 '''
class Darknet(nn.Module):
    def __init__(self, layers): # layers: [1, 2, 8, 8, 4]
        super(Darknet, self).__init__()
        self.in_depth = 32 # 【注】 此处的in_depth=32来源于: 3 --> 32, 最初的3通道输入在下一行
        # pre layers 先对三通道图片进行预处理 3 --> 32
        self.conv1 = nn.Conv2d(3, self.in_depth, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_depth)
        self.relu1 = nn.LeakyReLU(0.1)

        # 添加连续的残差模块
        self.layer1 = self._make_layers([32, 64], layers[0])
        self.layer2 = self._make_layers([64, 128], layers[1])
        self.layer3 = self._make_layers([128, 256], layers[2])
        self.layer4 = self._make_layers([256, 512], layers[3])
        self.layer5 = self._make_layers([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, planes, blocks):
        ''' layers：先降采样downsample，再连续添加最小残差模块BasicBlocks '''
        layers = []

        # downsample 先降采样，使用卷积滑步2来降采样，代替原先的maxpool2d
        layers.append(("ds_conv", nn.Conv2d(self.in_depth, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # add BasicBlocks
        self.in_depth = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.in_depth, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = Darknet([1, 2, 8, 8, 4])
    print('\nINFO: model successfully get from darknet!')
    if pretrained:
        if isinstance(pretrained, str): # 参数 pretrained 是否为str类型
            model.load_state_dict(torch.load("../weights/darknet53_weights_pytorch.pth"))
            print('\nINFO: Pretrained Net: ../weights/darknet53_weights_pytorch.pth')
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model