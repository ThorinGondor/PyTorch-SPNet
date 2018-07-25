import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np
from nets.backbone.darknet import darknet53

class ModelMain(nn.Module):
    def __init__(self, is_training=True):
        super(ModelMain, self).__init__()
        self.training = is_training

        # backbone
        self.backbone = darknet53("backbone_pretrained") # 训练时，传str非空；训练时，传str空
        _out_filters = self.backbone.layers_out_filters

        # embedding0
        final_out_filter0 = 3 * (5 + 80) # out_filters 3x(5+classes)
        self.embedding0 = self._make_embeding([512, 1024], _out_filters[-1], final_out_filter0) # filters_list, in_filters, out_filter

        # embedding1
        final_out_filter1 = 3 * (5 + 80)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embeding([256, 512], _out_filters[-2]+256, final_out_filter1) # CONCAT

        # embedding2
        final_out_filter2 = 3 * (5 + 80)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embeding([128, 256], _out_filters[-3]+128, final_out_filter2) # CONCAT

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks -1)//2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embeding(self, filters_list, in_filters, out_filters):
        model = nn.ModuleList([
            self._make_cbl(_in=in_filters, _out=filters_list[0], ks=1),
            self._make_cbl(_in=filters_list[0], _out=filters_list[1], ks=3),
            self._make_cbl(_in=filters_list[1], _out=filters_list[0], ks=1),
            self._make_cbl(_in=filters_list[0], _out=filters_list[1], ks=3),
            self._make_cbl(_in=filters_list[1], _out=filters_list[0], ks=1),
            self._make_cbl(_in=filters_list[0], _out=filters_list[1], ks=3)
        ])
        model.add_module("conv_out", nn.Conv2d(filters_list[1], out_filters, kernel_size=1,
                                               stride=1, padding=0, bias=True))
        return model

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        return out0, out1, out2

    def load_darnet_weights(self, weight_path):
        # Open the weight file:
        fp = open(weight_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        # Needed to write header when saving weights
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        print("total len weights = ", weights.shape)
        fp.close()

        ptr = 0
        all_dict = self.state_dict()
        all_keys = self.state_dict().keys()
        print(all_keys)
        last_bn_weight = None
        last_conv = None
        for i, (k, v) in enumerate(all_dict.items()):
            if 'bn' in k:
                if 'weight' in k:
                    last_bn_weight = v
                elif 'bias' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("bn_bias: ", ptr, num_b, k)
                    ptr += num_b
                    # weight
                    v = last_bn_weight
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("bn_weight: ", ptr, num_b, k)
                    ptr += num_b
                    last_bn_weight = None
                elif 'running_mean' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("bn_mean: ", ptr, num_b, k)
                    ptr += num_b
                elif 'running_var' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("bn_var: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
                else:
                    raise Exception("Error for bn")
            elif 'conv' in k:
                if 'weight' in k:
                    last_conv = v
                else:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("conv bias: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
        print("Total ptr = ", ptr)
        print("real size = ", weights.shape)

'''
config = {"model_params": {"backbone_name": "darknet_53"}}
net = ModelMain()
net.cuda()
print(net)
x = torch.randn(1, 3, 416, 416).cuda()
y0, y1, y2 = net(x)
print(y0.size())
print(y1.size())
print(y2.size())
'''