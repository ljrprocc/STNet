import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        
        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, x, mask_in=None):
        if mask_in is not None or self.last_size != (x.data.shape[2], x.data.shape[3]):
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                
                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0], x.data.shape[1], x.data.shape[2], x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2], x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PConv2d, self).forward(x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return torch.mul(output, self.update_mask), self.update_mask
        else:
            return output


class PConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act='ReLU', use_norm=True, residual=True):
        super(PConvLayer, self).__init__()
        self.conv = PConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, return_mask=True, multi_channel=True)
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.use_norm = use_norm
        self.residual = residual

        if act == 'ReLU':
            self.act = nn.ReLU(True)
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU(0.2, True)
        elif act == 'Tanh':
            self.act = nn.Tanh()

    def forward(self, x1, mask):
        x2, mask_update = self.conv(x1, mask)
        if self.use_norm:
            x2 = self.norm(x2)
        if self.residual:
            x2 = x2 + x1
        x2 = self.act(x2)
        # x1 = x2
        return x2, mask_update