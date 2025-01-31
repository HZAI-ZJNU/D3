from collections import OrderedDict
from torch import nn
import torch
import torch.nn.functional as F

COEFF = 12.0
BOTTLENECK_WIDTH = 128

__all__ = ['BottleneckLIP', 'BottleneckShared', 'SimplifiedLIP']


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x * weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)


def conv3x3(in_planes, out_planes, stride=1, g=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=g,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckLIP(nn.Module):
    def __init__(self, channels):
        super(BottleneckLIP, self).__init__()
        rp = BOTTLENECK_WIDTH

        self.postprocessing = nn.Sequential(
            OrderedDict((
                ('conv', conv1x1(rp, channels)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.postprocessing[0].weight.data.fill_(0.0)
        pass

    def forward_with_shared(self, x, shared):
        frac = lip2d(x, self.postprocessing(shared))
        return frac


class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


class BottleneckShared(nn.Module):
    def __init__(self, channels, g=1, r=BOTTLENECK_WIDTH):
        super(BottleneckShared, self).__init__()
        rp = r

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv1x1(channels, rp)),
                ('bn1', nn.InstanceNorm2d(rp, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(rp, rp, g=g)),
                ('bn2', nn.InstanceNorm2d(rp, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
            ))
        )

    def init_layer(self):
        pass

    def forward(self, x):
        return self.logit(x)


class SimplifiedLIP(nn.Module):
    def __init__(self, channels, g=1, k=3):
        super(SimplifiedLIP, self).__init__()

        rp = channels

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, k, padding=k//2, bias=False, groups=g)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac
