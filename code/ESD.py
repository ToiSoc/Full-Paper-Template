from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from models.resnet import *
from torch.hub import load_state_dict_from_url
import math
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'SEMICON_backbone']

model_urls = {
    'resnet18': 
        'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 
        'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 
        'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 
        'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 
        'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 
        'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 
        'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 
        'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# 定义卷积块
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                    stride=stride, padding=dilation, 
                    groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                    stride=stride, bias=False)

"""
    显式空间衰减注意机制（ESD）
"""
class ESD(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=num_heads)
        self.qkv2 = nn.Conv2d(dim, dim * 3, 1, groups=head_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, 
                        self.head_dim, H * W).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)
        x = ((attn @ v).reshape(B, C, H, W) + x).reshape(B, 
                self.num_heads, self.head_dim, H, W)
                .transpose(1, 2).reshape(B, C, H, W)
        y = self.norm(x)
        x = self.relu(y)
        qkv2 = self.qkv2(x).reshape(B, 3, self.head_dim, 
                        self.num_heads, H * W).transpose(0, 1)
        q, k, v = qkv2[0], qkv2[1], qkv2[2]
        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, self.head_dim, self.num_heads, H, W)
                        .transpose(1, 2).reshape(B, C, H, W) + y
        return x

# 定义基础块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, 
                                downsample=None, groups=1,
                                base_width=64, dilation=1, 
                                norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义精炼模块
class ResNet_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, 
                                zero_init_residual=False,
                                groups=1, width_per_group=64, 
                                norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, 
                                stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                            groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer))
            layers.append(ESD(planes * block.expansion, 
                            max(int(planes * block.expansion / 64), 16)))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)
