'''
# ultralytics.nn.my_modules.MSAC2PSA 的 Docstring
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv.


# 深度可分离卷积 (Depthwise Separable Convolution)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))  # depthwise + pointwise


class MSAC2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """MSAC2PSA module with enhanced attention for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)

        # 使用深度可分离卷积，确保空间尺寸不改变
        self.cv1 = DepthwiseSeparableConv(c1, 2 * self.c, kernel_size=1, stride=1, padding=0)
        self.cv2 = DepthwiseSeparableConv(2 * self.c, c1, kernel_size=1, stride=1, padding=0)

        # 使用多个自适应PSABlock来进行更深层的特征提取
        self.m = nn.Sequential(*(AdaptivePSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分割x为两部分 a 和 b，确保它们的空间维度一致
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 调用PSABlock进行特征提取
        b = self.m(b)

        # 拼接a和b，确保空间维度一致
        output = torch.cat((a, b), dim=1)

        # 通过cv2卷积层输出最终结果
        return self.cv2(output)


class AdaptivePSABlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes=(3, 5, 7), attn_ratio=0.5, num_heads=8):
        """自适应PSABlock，使用深度可分离卷积和更少的通道数来提高轻量化性能。"""
        super(AdaptivePSABlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.attn_ratio = attn_ratio
        self.num_heads = num_heads

        # 使用深度可分离卷积减少参数
        self.conv_layers = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, in_channels, kernel_size=k, padding=k//2) for k in kernel_sizes
        ])

        # 学习空间注意力图
        self.attention_fc = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 将不同尺寸的卷积结果按通道拼接
        features = [conv(x) for conv in self.conv_layers]
        features = torch.cat(features, dim=1)  # 拼接成一个大的特征图
        attention = self.attention_fc(features)  # 学习空间注意力图
        return x * attention  # 用注意力图对输入特征图进行加权


class ECAAttention(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        """高效通道注意力（ECA）机制，进一步减少计算量。"""
        super(ECAAttention, self).__init__()
        t = int(abs((math.log(in_channels, 2) + b) / gamma))  # 计算卷积核大小
        self.kernel_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=in_channels, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))  # 使用sigmoid控制通道重要性


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, num_heads, num_levels):
        """多尺度注意力机制，使用多个头来捕捉不同尺度的特征。"""
        super(MultiScaleAttention, self).__init__()
        self.num_levels = num_levels
        self.attn_heads = nn.ModuleList([ECAAttention(in_channels) for _ in range(num_heads)])
        self.sampling_offsets = nn.Linear(in_channels, num_levels * 2)  # 学习采样偏移量

    def forward(self, x):
        # 为每个尺度应用注意力机制
        sampled_values = [head(x) for head in self.attn_heads]
        return torch.stack(sampled_values, dim=-2)  # 将结果堆叠在多尺度维度上


# 为更好的收敛，增加了权重初始化
def init_weights(model):
    """初始化模型权重，加速收敛过程。"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


'''

########################################################################################################

'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv  # 深度可分离卷积 (Depthwise Separable Convolution)

# 深度可分离卷积（Depthwise Separable Convolution）优化版本
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)  # 添加BatchNorm层，帮助加速训练并稳定模型

    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))


class MSAC2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """MSAC2PSA模块，增强了特征提取能力"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)

        # 使用深度可分离卷积，减少参数量
        self.cv1 = DepthwiseSeparableConv(c1, 2 * self.c, kernel_size=1, stride=1, padding=0)
        self.cv2 = DepthwiseSeparableConv(5 * self.c, c1, kernel_size=1, stride=1, padding=0)
        self.cv3 = DepthwiseSeparableConv(c1, c1, kernel_size=1, stride=1, padding=0)

        # 增强特征提取的自适应PSABlock
        self.m = nn.Sequential(*(AdaptivePSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

        # 使用SE模块来进一步增强特征
        self.se_block = SELayer(self.c)

        # 使用高效通道注意力机制来增强通道间的信息流
        self.cross_attention = CrossAttention(self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)  # 分割通道，确保空间维度一致
        b = self.m(b)  # 使用PSABlock进行特征提取
        b = self.se_block(b)  # 使用SE模块进一步增强特征
        b = self.cross_attention(b)  # 引入跨通道注意力机制
        output = torch.cat((a, b), dim=1)  # 拼接a和b，确保空间维度一致
        t = self.cv2(output)  # 输出最终结果
        return t


class AdaptivePSABlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes=(3, 5, 7), attn_ratio=0.5, num_heads=8):
        """自适应PSABlock，增强了轻量化性能和空间注意力图学习"""
        super(AdaptivePSABlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.attn_ratio = attn_ratio
        self.num_heads = num_heads

        # 使用深度可分离卷积来减少计算量
        self.conv_layers = nn.ModuleList([DepthwiseSeparableConv(in_channels, in_channels, kernel_size=k, padding=k//2) for k in kernel_sizes])

        # 学习空间注意力图
        self.attention_fc = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 将不同尺寸的卷积结果按通道拼接
        features = [conv(x) for conv in self.conv_layers]
        features = torch.cat(features, dim=1)  # 拼接成一个大的特征图
        attention = self.attention_fc(features)  # 学习空间注意力图
        return x * attention  # 用注意力图对输入特征图进行加权


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """Squeeze-and-Excitation模块，增强通道间的关系"""
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)  # 对特征图应用SE模块


class CrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        """跨通道注意力机制，增强通道间信息流"""
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.ModuleList([ECAAttention(in_channels) for _ in range(num_heads)])
        self.sampling_offsets = nn.Linear(in_channels, 2 * num_heads)  # 学习采样偏移量

    def forward(self, x):
        sampled_values = [head(x) for head in self.attn_heads]
        return torch.cat(sampled_values, dim=1)  # 将多个通道头的结果拼接


class ECAAttention(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        """高效通道注意力机制（ECA），减少计算量"""
        super(ECAAttention, self).__init__()
        t = int(abs((math.log(in_channels, 2) + b) / gamma))  # 计算卷积核大小
        self.kernel_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=in_channels, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))  # 使用sigmoid控制通道重要性


# 为更好的收敛，增加了权重初始化
def init_weights(model):
    """初始化模型权重，加速收敛过程。"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

'''

############################################################################


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 动态卷积核生成（Dynamic Kernel Generation）和深度可分离卷积优化
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dynamic=False):
        super().__init__()
        self.dynamic = dynamic
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Mish()  # 使用Mish激活函数，提升训练表现

        # 动态卷积（通过学习生成卷积核）
        if dynamic:
            self.kernel_size = nn.Parameter(torch.randn(in_channels, out_channels))  # 动态生成卷积核

    def forward(self, x):
        if self.dynamic:
            kernel = F.softmax(self.kernel_size, dim=0)  # 使用softmax生成概率卷积核
            self.depthwise.weight = nn.Parameter(kernel)  # 动态调整卷积核
        return self.act(self.bn(self.pointwise(self.depthwise(x))))  # 使用Mish激活函数


class MSAC2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """MSAC2PSA模块，增强了特征提取能力."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)

        # 使用深度可分离卷积，减少参数量
        self.cv1 = DepthwiseSeparableConv(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, dynamic=True)
        self.cv2 = DepthwiseSeparableConv(5 * self.c, c1, kernel_size=1, stride=1, padding=0, dynamic=True)
        self.cv3 = DepthwiseSeparableConv(c1, c1, kernel_size=1, stride=1, padding=0)

        # 增强特征提取的自适应PSABlock
        self.m = nn.Sequential(*(AdaptivePSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

        # 使用SE模块来进一步增强特征
        self.se_block = SELayer(self.c)

        # 使用高效通道注意力机制来增强通道间的信息流
        self.cross_attention = CrossAttention(self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)  # 分割通道，确保空间维度一致
        b = self.m(b)  # 使用PSABlock进行特征提取
        b = self.se_block(b)  # 使用SE模块进一步增强特征
        b = self.cross_attention(b)  # 引入跨通道注意力机制
        output = torch.cat((a, b), dim=1)  # 拼接a和b，确保空间维度一致
        t = self.cv2(output)  # 输出最终结果
        return t


class AdaptivePSABlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes=(3, 5, 7), attn_ratio=0.5, num_heads=8):
        """自适应PSABlock，增强了轻量化性能和空间注意力图学习."""
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.attn_ratio = attn_ratio
        self.num_heads = num_heads

        # 使用深度可分离卷积来减少计算量
        self.conv_layers = nn.ModuleList(
            [DepthwiseSeparableConv(in_channels, in_channels, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )

        # 学习空间注意力图
        self.attention_fc = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * in_channels, in_channels, kernel_size=1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        # 将不同尺寸的卷积结果按通道拼接
        features = [conv(x) for conv in self.conv_layers]
        features = torch.cat(features, dim=1)  # 拼接成一个大的特征图
        attention = self.attention_fc(features)  # 学习空间注意力图
        return x * attention  # 用注意力图对输入特征图进行加权


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """Squeeze-and-Excitation模块，增强通道间的关系."""
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)  # 对特征图应用SE模块


class CrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        """跨通道注意力机制，增强通道间信息流."""
        super().__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.ModuleList([ECAAttention(in_channels) for _ in range(num_heads)])
        self.sampling_offsets = nn.Linear(in_channels, 2 * num_heads)  # 学习采样偏移量

    def forward(self, x):
        sampled_values = [head(x) for head in self.attn_heads]
        return torch.cat(sampled_values, dim=1)  # 将多个通道头的结果拼接


class ECAAttention(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        """高效通道注意力机制（ECA），减少计算量."""
        super().__init__()
        t = int(abs((math.log(in_channels, 2) + b) / gamma))  # 计算卷积核大小
        self.kernel_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))  # 使用sigmoid控制通道重要性


# 为更好的收敛，增加了权重初始化
def init_weights(model):
    """初始化模型权重，加速收敛过程。."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
