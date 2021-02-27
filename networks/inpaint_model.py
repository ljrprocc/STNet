import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as T

from networks.baselines import *
from utils.inpaint_utils import ContextualAttention

# from utils.inpaint_utils import flow_to_image

class GateConv2d(nn.Conv2d):
    def __init__(self, activation=None, *args, **kwargs):
        super(GateConv2d, self).__init__(*args, **kwargs)
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, x.shape[1] // 2, dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x


class GatedCoarse2FineModel(nn.Module):
    def __init__(self, hidden_channels=24):
        super(GatedCoarse2FineModel, self).__init__()
        self.gen_relu = nn.ELU(inplace=True)
        self.hidden_channels = hidden_channels
        self.relu = nn.ReLU(inplace=True)
        self.build_inpaint_model()

    def build_inpaint_model(self):
        self.conv1s = []
        self.bn1s = []
        self.conv1s.append(GateConv2d(self.gen_relu, 3, self.hidden_channels * 2, 3, 2, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv1s.append(GateConv2d(self.gen_relu, self.hidden_channels, self.hidden_channels * 4, 3, 2, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv1s = nn.ModuleList(self.conv1s)
        self.bn1s = nn.ModuleList(self.bn1s)
        self.conv2s = []
        self.bn2s = []
        self.conv2s.append(GateConv2d(self.gen_relu, 3, self.hidden_channels * 2, 3, 2, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv2s.append(GateConv2d(self.relu, self.hidden_channels, self.hidden_channels * 4, 3, 2, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv2s.append(ContextAttention(ksize=3, stride=1, rate=2))
        self.conv2s = nn.ModuleList(self.conv2s)
        self.bn2s = nn.ModuleList(self.bn2s)
        self.total_conv = []
        self.total_bn = []
        self.total_conv.append(GateConv2d(self.gen_relu, self.hidden_channels * 4, self.hidden_channels * 4, 3, 1, padding=1))
        self.total_bn.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.total_conv.append(GateTransposed2d(self.gen_relu, self.hidden_channels * 2, self.hidden_channels * 2, 3, 2, padding=1))
        self.total_bn.append(nn.InstanceNorm2d(self.hidden_channels))
        self.total_conv.append(GateTransposed2d(self.gen_relu, self.hidden_channels, self.hidden_channels, 3, 2, padding=1))
        self.total_bn.append(nn.InstanceNorm2d(self.hidden_channels // 2))
        self.total_conv.append(GateConv2d(None, self.hidden_channels // 2, 3, 3, 1, padding=1))
        self.total_conv = nn.ModuleList(self.total_conv)
        self.total_bn = nn.ModuleList(self.total_bn)
    
    def forward(self, x, xori, mask=None):
        x1 = x * mask.repeat(1,3,1,1) + xori * (1. - mask.repeat(1,3,1,1))
        xnow = x1
        for i, conv in enumerate(self.conv1s):
            # print(x1.shape)
            x1 = conv(x1)
            x1 = self.bn1s[i](x1)
            # x1 = self.gen_relu(x1)
            # print(x1.shape)
        x2 = xnow
        offsets = None
        mask_s = F.interpolate(mask, size=(x1.shape[2], x1.shape[3]))
        for i, conv in enumerate(self.conv2s):
            if i == len(self.conv2s) - 1:
                x2, offsets = conv(x2, x2, mask=mask_s)
            else:
                x2 = conv(x2)
                x2 = self.bn2s[i](x2)
            # x2 = self.gen_relu(x2) if i != 1 else self.relu(x2)
        x = torch.cat([x1, x2], 1)
        for i, conv in enumerate(self.total_conv):
            x = conv(x)
            if i < len(self.total_conv) - 1:
                x = self.total_bn[i](x)
                # print(x.shape)
                # x = self.gen_relu(x)
        x = torch.tanh(x)
        return x, offsets

class TinyCoarse2FineModel(nn.Module):
    def __init__(self, hidden_channels=48):
        super(TinyCoarse2FineModel, self).__init__()
        self.gen_relu = nn.ELU(inplace=True)
        self.hidden_channels = hidden_channels
        self.relu = nn.ReLU(inplace=True)
        self.build_inpaint_model()
    
    def build_inpaint_model(self):
        self.conv1s = []
        self.bn1s = []
        self.conv1s.append(nn.Conv2d(3, self.hidden_channels, 3, 2, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv1s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, 3, 2, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv1s = nn.ModuleList(self.conv1s)
        self.bn1s = nn.ModuleList(self.bn1s)
        self.conv2s = []
        self.bn2s = []
        self.conv2s.append(nn.Conv2d(3, self.hidden_channels, 3, 2, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv2s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, 3, 2, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv2s.append(ContextAttention(ksize=3, stride=1, rate=2))
        self.conv2s = nn.ModuleList(self.conv2s)
        self.bn2s = nn.ModuleList(self.bn2s)
        self.total_conv = []
        self.total_bn = []
        self.total_conv.append(nn.Conv2d(self.hidden_channels * 4, self.hidden_channels * 2, 3, 1, padding=1))
        self.total_bn.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.total_conv.append(nn.ConvTranspose2d(self.hidden_channels * 2, self.hidden_channels, 4, 2, padding=1))
        self.total_bn.append(nn.InstanceNorm2d(self.hidden_channels))
        self.total_conv.append(nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels // 2, 4, 2, padding=1))
        self.total_bn.append(nn.InstanceNorm2d(self.hidden_channels // 2))
        self.total_conv.append(nn.Conv2d(self.hidden_channels // 2, 3, 3, 1, padding=1))
        self.total_conv = nn.ModuleList(self.total_conv)
        self.total_bn = nn.ModuleList(self.total_bn)
    
    def forward(self, x, xori, mask=None):
        x1 = x * mask.repeat(1,3,1,1) + xori * (1. - mask.repeat(1,3,1,1))
        xnow = x1
        for i, conv in enumerate(self.conv1s):
            x1 = conv(x1)
            x1 = self.bn1s[i](x1)
            x1 = self.gen_relu(x1)
        x2 = xnow
        offsets = None
        mask_s = F.interpolate(mask, size=(x1.shape[2], x1.shape[3]))
        for i, conv in enumerate(self.conv2s):
            if i == len(self.conv2s) - 1:
                x2, offsets = conv(x2, x2, mask=mask_s)
            else:
                x2 = conv(x2)
                x2 = self.bn2s[i](x2)
            x2 = self.gen_relu(x2) if i != 1 else self.relu(x2)
        x = torch.cat([x1, x2], 1)
        for i, conv in enumerate(self.total_conv):
            x = conv(x)
            if i < len(self.total_conv) - 1:
                x = self.total_bn[i](x)
                x = self.gen_relu(x)
        x = torch.tanh(x)
        return x, offsets


class Coarse2FineModel(nn.Module):
    def __init__(self, hidden_channels=48, dilation_depth=0):
        super(Coarse2FineModel, self).__init__()
        # Stage1 model
        self.hidden_channels = hidden_channels
        self.dilation_depth = dilation_depth
        self.gen_relu = nn.ELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        # self.last_act = nn.Tanh()
        self.build_inpaint_model()

    def build_inpaint_model(self):
        # Define Coarse-to-Fine Network
        # Stage 2, conv branch
        self.conv_1s = []
        self.bn1s = []
        self.conv_1s.append(nn.Conv2d(3, self.hidden_channels, 5, 1, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 2, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels*2, 3, 1, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*2, 3, 2, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*4, 3, 1, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        for i in range(self.dilation_depth):
            self.conv_1s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, dilation=2 ** (i + 1), padding=2 ** (i + 1)))
            self.bn1s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.conv_1s = nn.ModuleList(self.conv_1s)
        # Stage 2, attention branch
        self.conv_2s = []
        self.bn2s = []
        self.conv_2s.append(nn.Conv2d(3, self.hidden_channels, 5, 1, padding=2))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 2, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels, 2*self.hidden_channels, 3, 1, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*2, 3, 2, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*4, 3, 1, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        # context attention
        self.conv_2s.append(ContextualAttention(ksize=3, stride=1, rate=2, use_cuda=True))
        self.bn2s.append(None)
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.bn2s.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.conv_2s = nn.ModuleList(self.conv_2s)
        # total merged branch
        self.totals = []
        self.total_bns = []
        self.totals.append(nn.Conv2d(self.hidden_channels*8, self.hidden_channels*4, 3, 1, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.totals.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels * 4))
        self.totals.append(nn.ConvTranspose2d(self.hidden_channels*4, self.hidden_channels*2, 4, 2, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.totals.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*2, 3, 1, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.totals.append(nn.ConvTranspose2d(self.hidden_channels*2, self.hidden_channels*2, 4, 2, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels * 2))
        self.totals.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels, 3, 1, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels))
        self.totals.append(nn.Conv2d(self.hidden_channels, self.hidden_channels // 2, 3, 1, padding=1))
        self.total_bns.append(nn.InstanceNorm2d(self.hidden_channels // 2))
        self.totals.append(nn.Conv2d(self.hidden_channels // 2, 3, 3, 1, padding=1))
        self.totals = nn.ModuleList(self.totals)

    def forward(self, x, xori, mask=None):
        x1 = x * mask.repeat(1,3,1,1) + xori * (1. - mask.repeat(1,3,1,1))
        xnow = x1
        for i, conv in enumerate(self.conv_1s):
            x1 = conv(x1)
            x1 = self.bn1s[i](x1)
            x1 = self.gen_relu(x1)
            # print(x1.shape)
        # print(mask.shape)
        # print(x1.shape)
        # x2 = x1 * mask + x * (1. - mask)
        x2 = xnow
        offsets = None
        for i, conv in enumerate(self.conv_2s):
            # print(torch.isnan(x2).int().sum(), i)
            if i == 6:
                # print(x2.shape)
                x2, offsets = conv(x2, x2, mask=mask)
                # print(x2.shape)
            else:
                # print(x2.shape)
                x2 = conv(x2)
                x2 = self.bn2s[i](x2)
                # offsets = None
            x2 = self.gen_relu(x2) if i != 5 else self.relu(x2)
        # print(x1.shape, x2.shape)
        x = torch.cat([x1, x2], 1)
        for i, conv in enumerate(self.totals):
            # if i == 2 or i == 4:
            #     x = F.upsample(x, scale_factor=2)
            # print(x.shape)
            
            x = conv(x)
            if i < len(self.totals) - 1:
                x = self.total_bns[i](x)
                x = self.gen_relu(x)
        x = torch.tanh(x)
        # print(x[0, :, :, 0].mean(), x[0, :, :, 1].mean(), x[0, :, :, 2].mean())
        return x, offsets
