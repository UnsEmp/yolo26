#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ASSPConv.py
# @Time      :2024/7/14 下午3:15
# @Author    :Bangyan


import torch
import torch.nn as nn
import torch.nn.functional as F




# Dilated Convolution
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv,self).__init__(*modules)


# Pool(1x1) -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
        nn.AdaptiveAvgPool2d(1),  # [256*1*1]
        # 自适应平均池化层，只需要给定输出的特征图的尺寸(括号内数字)就好了
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())

    def forward(self, x):
            size = x.shape[-2:]
            x = super(ASPPPooling, self).forward(x)
            return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    ASPP空洞卷积块
    """

    def __init__(self, in_channels, atrous_rates):  # atrous_rates=(6, 12, 18)
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # (64-1+2*0)/1+1=64[256*64*64]
            nn.BatchNorm2d(out_channels),  # [256*64*64]
            nn.ReLU()))  # 1x1卷积
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))  # 3*3卷积( padding=6, dilation=6 )
        modules.append(ASPPConv(in_channels, out_channels, rate2))  # 3*3 卷积( padding=12, dilation=12 )
        modules.append(ASPPConv(in_channels, out_channels, rate3))  # 3*3 卷积( padding=18, dilation=18 )  [256*64*64]
        modules.append(ASPPPooling(in_channels, out_channels))  # 全局平均池化操作，输出尺寸为（1,1） [256*1*1]
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(  # 特征融合？此时输入通道是原始输入通道的5倍。输出的结果又回到原始的通道数。
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),  # [1280*64*64]
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        net = self.project(res)  # 特征融合1280——>256
        # return x + net
        return net