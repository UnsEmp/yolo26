import torch
import torch.nn as nn
from ultralytics.nn.modules import GhostConv

class Gcbv3(nn.Module):
    """
    Gcbv3 (Ghost Conv Branch v3) Module
    结构描述:
    1. Input -> 分为三个支路
    2. Branch 1: GhostConv (保持通道不变) -> MaxPool (stride=1, 保持尺寸)
    3. Branch 2: AvgPool (stride=1, 保持尺寸)
    4. Branch 3: Identity (原样保持)
    5. Fusion: Element-wise Add (Branch1 + Branch2 + Branch3)
    6. Output: GhostConv (调整通道到 c2)
    """
    def __init__(self, c1, c2, k=3):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 池化核大小 (默认为 3)
        """
        super().__init__()
        
        # 支路 1：先 GhostConv 提取特征，再 MaxPool 提取显著特征
        # 注意：这里的 GhostConv 输出通道必须是 c1，以便后续与原始输入相加
        self.branch1_conv = GhostConv(c1, c1, k=1, s=1) 
        # padding=k//2 确保池化后尺寸不变
        self.branch1_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

        # 支路 2：AvgPool 提取背景/平滑特征
        self.branch2_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k//2)

        # 支路 3：Identity (不需要定义层，直接用 x)

        # 最后的融合层：将相加后的结果映射到目标通道 c2
        self.final_conv = GhostConv(c1, c2, k=1, s=1)

    def forward(self, x):
        # 支路 1 处理
        b1 = self.branch1_conv(x)
        b1 = self.branch1_pool(b1)

        # 支路 2 处理
        b2 = self.branch2_pool(x)

        # 支路 3 处理 (原始信息)
        b3 = x

        # 融合：三个支路特征相加 (Element-wise Add)
        # 注意：相加要求 b1, b2, b3 的形状完全一致
        out = b1 + b2 + b3

        # 输出：通过最后的 GhostConv 调整通道数
        return self.final_conv(out)