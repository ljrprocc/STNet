import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from networks.baselines import *

def extract_image_patches(img, kernel, stride=1, dilation=1):
    b,c,h,w = img.shape
    h2 = math.ceil(h /stride)
    w2 = math.ceil(w /stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    # print(h, h2, pad_row)
    # print(w, w2, pad_col)

    x = F.pad(img, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])
    
    
class ContextAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., trainging=True, fuse=True):
        super(ContextAttention, self).__init__()
        self.stride = stride
        self.softmax_scale = softmax_scale
        self.ksize = ksize
        self.rate = rate
        self.fuse_k = fuse_k
        self.fuse = fuse
    
    def forward(self, f, b, mask=None):
        kernel = self.rate * 2
        rate = self.rate
        
        # print(f.shape, b.shape)
        bs, cs, hs, ws = f.shape
        raw_w = extract_image_patches(b, kernel=kernel, stride=self.rate*self.stride, dilation=1)
        # print(raw_w.shape)
        raw_w = torch.reshape(raw_w, (bs, -1, kernel, kernel, cs))
        raw_w = raw_w.permute(0,1,4,2,3)

        f = F.interpolate(f, scale_factor=1./rate)
        fs = f.shape
        b = F.interpolate(b, size=(int(hs/rate), int(ws/rate)))
        # print(mask.shape)
        if mask is not None:
            mask = F.interpolate(mask, size=(int(hs/rate), int(ws/rate)))
        # print(mask.shape)
        int_fs = f.shape
        ibs, ics, ihs, iws = b.shape
        f_groups = torch.chunk(f, int_fs[0], dim=0)
        w = extract_image_patches(b, kernel=self.ksize, stride=self.stride, dilation=1)
        # print(f_groups[0].shape)
        # exit(-1)
        w = torch.reshape(w, (bs, -1, self.ksize, self.ksize, cs))
        w = w.permute(0,1,4,2,3)
        # print(w.shape)

        if mask is None:
            mask = torch.zeros((1, 1, ihs, iws))
        
        m = extract_image_patches(mask, kernel=self.ksize, stride=self.stride)
        # print(m.shape)
        m = torch.reshape(m, (bs, -1, self.ksize, self.ksize, 1))
        m = m.permute(0,2,3,4,1)
        # print(m.shape)
        # exit(-1)
        m = m[0]
        # print(mask.shape)
        mm = torch.mean(m, dim=(0,1,2)).eq(0.).float().to(f.device)
        w_groups = torch.chunk(w, bs, dim=0)
        raw_w_groups = torch.chunk(raw_w, bs, dim=0)
        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale
        fuse_weight = torch.reshape(torch.eye(k), [1,1,k,k]).to(f.device)
        # print(f_groups[0].shape)
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            wi = wi[0]
            # print(xi.shape)
            wi_normed = wi / torch.reshape(torch.clamp(torch.sqrt(torch.sum(wi * wi, (1,2,3))), max=1e-4), [-1, 1, 1, 1])
            yi = F.conv2d(xi, wi_normed, padding=1)
            # print(yi.shape)
            if self.fuse:
                yi = torch.reshape(yi, [1, 1, fs[2]*fs[3], ihs*iws])
                yi = F.conv2d(yi, fuse_weight, padding=1)
                yi = torch.reshape(yi, [1, fs[2], fs[3], ihs, iws])
                yi = yi.permute(0,2,1,4,3)
                yi = torch.reshape(yi, [1, 1, fs[2]*fs[3], ihs*iws])
                yi = F.conv2d(yi, fuse_weight, padding=1)
                yi = torch.reshape(yi, [1, fs[3], fs[2], iws, ihs])
                yi = yi.permute(0,2,1,4,3)
            yi = torch.reshape(yi, [1, fs[2], fs[3], iws*ihs])

            # softmax to match
            yi = yi.clone() * mm
            yi = F.softmax(yi.clone()*scale, 3)
            yi = yi.clone() * mm

            offset = torch.argmax(yi, 3).int()
            offset = torch.stack([offset // hs, offset % hs], axis=-1)
            offset = offset.permute(0,3,1,2)

            # paste center
            wi_center = raw_wi[0]
            yi = yi.permute(0,3,1,2)
            yi = F.conv_transpose2d(yi, wi_center, stride=rate, padding=1) / 4.
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, 0)
        offsets = torch.cat(offsets, 0)
        h_add = torch.reshape(torch.arange(ihs), [1, 1, ihs, 1]).repeat([bs, 1, 1, iws]).to(f.device)
        w_add = torch.reshape(torch.arange(iws), [1, 1, 1, iws]).repeat([bs, 1, ihs, 1]).to(f.device)
        # print(y.shape, offsets.shape)
        # print(h_add.shape, w_add.shape)
        offsets = offsets - torch.cat([h_add, w_add], 1)
        
        # exit(-1)
        return y, offsets


class Coarse2FineModel(nn.Module):
    def __init__(self, hidden_channels=48, dilation_depth=4):
        super(Coarse2FineModel, self).__init__()
        # Stage1 model
        self.hidden_channels = hidden_channels
        self.dilation_depth = dilation_depth
        self.gen_relu = nn.ELU()
        self.relu = nn.ReLU()
        self.last_act = nn.Tanh()
        self.build_inpaint_model()

    def build_inpaint_model(self):
        # Define Coarse-to-Fine Network
        # Stage 2, conv branch
        self.conv_1s = []
        self.conv_1s.append(nn.Conv2d(3, self.hidden_channels, 5, 1, padding=1))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 2, padding=1))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels*2, 3, 1, padding=1))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*2, 3, 2, padding=1))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*4, 3, 1, padding=1))
        self.conv_1s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        for i in range(self.dilation_depth):
            self.conv_1s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, dilation=2 ** (i + 1), padding=2 ** (i + 1)))
        self.conv_1s = nn.ModuleList(self.conv_1s)
        # Stage 2, attention branch
        self.conv_2s = []
        self.conv_2s.append(nn.Conv2d(3, self.hidden_channels, 5, 1, padding=1))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 2, padding=1))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels, 2*self.hidden_channels, 3, 1, padding=1))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*4, 3, 2, padding=1))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        # context attention
        self.conv_2s.append(ContextAttention(ksize=3, stride=1, rate=2))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.conv_2s.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.conv_2s = nn.ModuleList(self.conv_2s)
        # total merged branch
        self.totals = []
        self.totals.append(nn.Conv2d(self.hidden_channels*8, self.hidden_channels*4, 3, 1, padding=1))
        self.totals.append(nn.Conv2d(self.hidden_channels*4, self.hidden_channels*4, 3, 1, padding=1))
        self.totals.append(nn.ConvTranspose2d(self.hidden_channels*4, self.hidden_channels*2, 4, 2, padding=1))
        self.totals.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels*2, 3, 1, padding=1))
        self.totals.append(nn.ConvTranspose2d(self.hidden_channels*2, self.hidden_channels*2, 4, 2, padding=1))
        self.totals.append(nn.Conv2d(self.hidden_channels*2, self.hidden_channels, 3, 1, padding=1))
        self.totals.append(nn.Conv2d(self.hidden_channels, self.hidden_channels // 2, 3, 1, padding=1))
        self.totals.append(nn.Conv2d(self.hidden_channels // 2, 3, 3, 1, padding=1))
        self.totals = nn.ModuleList(self.totals)

    def forward(self, x, xori, mask=None):
        x1 = x * mask.repeat(1,3,1,1) + xori * (1. - mask.repeat(1,3,1,1))
        xnow = x1
        for conv in self.conv_1s:
            x1 = conv(x1)
            x1 = self.gen_relu(x1)
            # print(x1.shape)
        # print(mask.shape)
        # print(x1.shape)
        # x2 = x1 * mask + x * (1. - mask)
        x2 = xnow
        offsets = None
        for i, conv in enumerate(self.conv_2s):
            if i == 6:
                # print(x2.shape)
                x2, offsets = conv(x2, x2, mask=mask)
                # print(x2.shape)
            else:
                x2 = conv(x2)
                offsets = None
            x2 = self.gen_relu(x2)
        # print(x1.shape, x2.shape)
        x = torch.cat([x1, x2], 1)
        for i, conv in enumerate(self.totals):
            # if i == 2 or i == 4:
            #     x = F.upsample(x, scale_factor=2)
            # print(x.shape)
            x = conv(x)
            if i < len(self.totals) - 1:
                x = self.gen_relu(x)
            else:
                x = self.last_act(x)
        return xnow, offsets
