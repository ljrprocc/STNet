import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pconv2d import *

def pconv3x3(in_channels, out_channels, stride=1, 
             padding=1, bias=True, groups=1):
    return PConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
        return_mask=True,
        multi_channel=True)


def pconv1x1(in_channels, out_channels, groups=1):
    return PConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        groups=groups,
        return_mask=True,
        multi_channel=True)

# def uppconv2d(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Upsample(mode='bilinear', scale_factor=2),
#         pconv1x1(in_channels, out_channels)
#     )


class FinalPConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, residual=True, norm=True):
        super(FinalPConv, self).__init__()
        self.residual = residual
        self.norm = norm
        self.conv1 = pconv3x3(in_channels, out_channels)
        self.conv2 = []
        self.conv3 = pconv1x1(out_channels, out_channels)
        for _ in range(blocks - 1):
            self.conv2.append(PConvLayer(out_channels, out_channels, stride=1, padding=1, bias=True, use_norm=norm, residual=residual))
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x, mask):
        return self.forward(x, mask)

    def forward(self, x, mask):
        x1, mask_in = self.conv1(x, mask)
        x1 = F.relu(x1)
        x2 = None
        for conv in self.conv2:
            x2, mask_in = conv(x1, mask_in)
            x1 = x2
        return self.conv3(x2, mask_in)


class DownPConvD(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, pooling=True, residual=True, norm=True):
        super(DownPConvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.norm = norm
        self.conv1 = pconv3x3(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(PConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=norm, residual=residual))
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x, mask):
        return self.forward(x, mask)
    
    def forward(self, x, mask):
        x1, mask_in = self.conv1(x, mask)
        # print('*')
        x1 = F.relu(x1)
        x2 = None
        for conv in self.conv2:
            x2, mask_in = conv(x1, mask_in)
            x1 = x2
        before_pool = x2
        before_pool_mask = mask_in
        if self.pooling:
            x2 = self.pool(x2)
            mask_in = self.pool(mask_in)
        # print(x2.shape)
        return x2, before_pool, mask_in, before_pool_mask


class UpPConvD(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, pooling=True, residual=True, norm=True, concat=True):
        super(UpPConvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.norm = norm
        self.concat = concat
        # self.upconv = uppconv2d(in_channels, out_channels)
        self.upconv_upsample = nn.Upsample(mode='bilinear', scale_factor=2)
        self.upconv_conv = pconv1x1(in_channels, out_channels)
        if self.concat:
            self.conv1 = pconv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = pconv3x3(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(PConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=norm, residual=residual))
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x, mask, from_down, from_down_mask):
        return self.forward(x, mask, from_down, from_down_mask)
    
    def forward(self, x, mask, from_down, from_down_mask):
        x, mask = self.upconv_upsample(x), self.upconv_upsample(mask)
        x, mask_in = self.upconv_conv(x, mask)
        # x, mask_in = self.upconv(x)
        if self.concat:
            # print(x.shape, from_down.shape)
            x1 = torch.cat((x, from_down), 1)
            mask_in = torch.cat((mask_in, from_down_mask), 1)
            # print(x1.shape)
        else:
            if from_down is not None:
                x1 = x + from_down
            else:
                x1 = x
        # print(mask_in.shape)
        x1, mask_in = self.conv1(x1, mask_in)
        x1 = F.relu(x1)
        x2 = None
        for conv in self.conv2:
            x2, mask_in = conv(x1, mask_in)
            x1 = x2
        # print(type(x2))
        return x2, mask_in