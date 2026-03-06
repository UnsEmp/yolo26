# import torch
# import torch.nn as nn
# from ultralytics.nn.modules.conv import Conv


# # ==========================================
# # 1. 引用或确认 C3Ghost (官方已有，无需重写，只需确认导入)
# # from .block import C3Ghost 
# # ==========================================

# # ==========================================
# # 2. 创新的 A2C2f_Star 模块 (用于 Neck)
# # 结合了 C2f 的梯度流 + StarNet 的高维映射 + 注意力
# # ==========================================
# class StarInteraction(nn.Module):
#     """StarNet 核心交互单元: 深度卷积 + 元素乘法"""
#     def __init__(self, c1, c2, k=3, s=1):
#         super().__init__()
#         self.dwconv = Conv(c1, c1, k, s, g=c1, act=False) 
#         self.f1 = Conv(c1, c2, 1, 1, act=False)
#         self.f2 = Conv(c1, c2, 1, 1, act=False)
#         self.act = nn.ReLU6()

#     def forward(self, x):
#         x_spatial = self.dwconv(x)
#         return self.f1(x_spatial) * self.act(self.f2(x_spatial))

# class A2c2fStarFast(nn.Module):
#     """
#     Safe Fast Version of A2C2f-Star
#     - No inner blocks
#     - Channel-safe
#     - Fully compatible with Ultralytics parse_model
#     """

#     def __init__(self, c1, c2, n=1, e=0.5):
#         super().__init__()

#         # hidden channels
#         c_ = int(c2 * e)

#         # IMPORTANT: in_channels MUST be c1
#         self.cv1 = Conv(c1, 2 * c_, 1, 1)

#         # concat channels = 2 * c_
#         self.cv2 = Conv(2 * c_, c2, 1, 1)

#     def forward(self, x):
#         # x: [B, c1, H, W]
#         y1, y2 = self.cv1(x).chunk(2, 1)
#         out = torch.cat((y1, y2), dim=1)
#         return self.cv2(out)

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class StarInteraction(nn.Module):
    """StarNet-style interaction unit (depthwise conv + gated projection)"""
    def __init__(self, c):
        super().__init__()
        self.dwconv = Conv(c, c, 3, 1, g=c, act=False)
        self.f1 = Conv(c, c, 1, 1, act=False)
        self.f2 = Conv(c, c, 1, 1, act=False)
        self.act = nn.ReLU6(inplace=False)  # ⚠️ 禁止 inplace

    def forward(self, x):
        x = self.dwconv(x)
        return self.f1(x) * self.act(self.f2(x))


class A2C2fStarFast(nn.Module):
    """
    Repeatable A2C2f-Star module
    - Fully compatible with Ultralytics parse_model
    - Safe for repeats > 1
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 1. channel split
        self.cv1 = Conv(c1, 2 * c_, 1, 1)

        # 2. repeatable Star blocks
        self.m = nn.ModuleList(StarInteraction(c_) for _ in range(n))

        # 3. channel fuse
        self.cv2 = Conv((2 + n) * c_, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # [y1, y2]

        for block in self.m:
            y.append(block(y[-1]))

        return self.cv2(torch.cat(y, dim=1))
