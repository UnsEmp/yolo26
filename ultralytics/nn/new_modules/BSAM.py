#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :BSAM.py
# @Time      :2024/7/1 下午4:26
# @Author    :Bangyan
import torch.nn as nn
import torch
class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16, light=False):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.light = light
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.light:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.shared_MLP = nn.Sequential(
                nn.Linear(in_features=c1, out_features=mid_channel),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(in_features=mid_channel, out_features=c1)
            )
        else:

            self.shared_MLP = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        if self.light:
            avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
            maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
            fc_out = (avgout + maxout)
        else:
            fc_out = (self.shared_MLP(self.avg_pool(x)))
        return x * self.act(fc_out)