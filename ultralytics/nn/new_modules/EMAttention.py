import torch
from torch import nn
import torch.nn.functional as F

class EMA(nn.Module):
    """
    EMA++: Export-friendly + stronger for small objects & occlusion
    Key upgrades:
      1) Replace AdaptivePool(None,1) with reduce ops (mean/max) for export stability
      2) Add channel gate (ECA/SE-like) to complement coordinate/spatial weighting
      3) Residual scaling gamma (init 0) to prevent over-suppression, improve recall
      4) Use DWConv large-kernel for local context at low cost
    """
    def __init__(self, channels, factor=32, reduction=16, k_large=5, tau=1.0):
        super().__init__()
        # pick a valid group number (must divide channels)
        g = min(factor, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.tau = tau  # softmax temperature (stability)

        c_g = channels // self.groups

        # Coordinate mixing (like CA, but group-wise)
        self.gn = nn.GroupNorm(num_groups=c_g, num_channels=c_g)  # instance-like per group

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c_g, c_g, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_g),
            nn.SiLU()
        )

        # Large kernel local context (DWConv to keep light)
        pad = k_large // 2
        self.dw_large = nn.Sequential(
            nn.Conv2d(c_g, c_g, kernel_size=k_large, padding=pad, groups=c_g, bias=False),
            nn.BatchNorm2d(c_g),
            nn.SiLU()
        )
        self.pw = nn.Sequential(
            nn.Conv2d(c_g, c_g, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_g),
            nn.SiLU()
        )

        # Channel gate (SE-lite)
        mid = max(8, c_g // reduction)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_g, mid, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid, c_g, 1, bias=True),
            nn.Sigmoid()
        )

        # learnable residual scaling to avoid over-attention
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        c_g = c // g

        # group reshape
        gx = x.view(b * g, c_g, h, w)

        # ---- Export-friendly mixed pooling for coordinate descriptors ----
        # Hx1 descriptor (reduce over W)
        x_h = gx.mean(dim=3, keepdim=True) + gx.amax(dim=3, keepdim=True)  # (bg, cg, h, 1)
        # 1xW descriptor (reduce over H)
        x_w = gx.mean(dim=2, keepdim=True) + gx.amax(dim=2, keepdim=True)  # (bg, cg, 1, w)
        x_w = x_w.permute(0, 1, 3, 2)                                       # (bg, cg, w, 1)

        # coordinate interaction
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))                     # (bg, cg, h+w, 1)
        a_h, a_w = torch.split(hw, [h, w], dim=2)
        a_w = a_w.permute(0, 1, 3, 2)                                       # (bg, cg, 1, w)

        x1 = self.gn(gx * torch.sigmoid(a_h) * torch.sigmoid(a_w))

        # local context branch (DW large kernel + PW)
        x2 = self.pw(self.dw_large(gx))

        # ---- Cross-aggregation (original EMA spirit) ----
        # global descriptors
        v1 = F.adaptive_avg_pool2d(x1, 1).view(b * g, c_g)                  # (bg, cg)
        v2 = F.adaptive_avg_pool2d(x2, 1).view(b * g, c_g)

        # temperature softmax improves stability on dense/occluded scenes
        q1 = self.softmax((v1 / self.tau).unsqueeze(1))                     # (bg, 1, cg)
        k1 = x2.view(b * g, c_g, -1)                                        # (bg, cg, hw)

        q2 = self.softmax((v2 / self.tau).unsqueeze(1))
        k2 = x1.view(b * g, c_g, -1)

        # spatial weights
        w_sp = (torch.matmul(q1, k1) + torch.matmul(q2, k2)).view(b * g, 1, h, w)
        w_sp = torch.sigmoid(w_sp)

        # channel weights
        w_ch = self.ch_gate(gx)

        # combine + residual scaling
        attn = w_sp * w_ch
        out = gx * (1.0 + self.gamma * attn)

        return out.view(b, c, h, w)


# import torch
# from torch import nn

# __all__ = ['EMA', 'C2f_EMA']

# class EMA(nn.Module):
#     def __init__(self, channels, factor=32):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
        
#         # 计算分组后的通道数
#         mid_channels = channels // self.groups

#         # ==========================================
#         # [修改点 2] 混合池化策略 (Mixed Pooling)
#         # ==========================================
#         # 原代码仅使用了 AvgPool。这里同时定义 MaxPool，
#         # 目的是同时保留背景信息(Avg)和纹理/显著特征(Max)。
#         self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))
#         self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))
        
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))

#         self.gn = nn.GroupNorm(mid_channels, mid_channels)

#         # ==========================================
#         # [修改点 3] 优化激活函数 (Optimized Activation)
#         # ==========================================
#         # 原代码直接使用 Conv2d，没有 BN 和激活函数。
#         # 修改为 Conv + BN + SiLU (Swish) 结构，提供非线性表达并稳定梯度。
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU()  # 使用 SiLU 代替无激活或普通 ReLU
#         )

#         # ==========================================
#         # [修改点 1] 多尺度大核卷积 (Multi-scale Large Kernel)
#         # ==========================================
#         # 原代码使用 3x3 卷积。
#         # 修改为 5x5 (或更大) 卷积，以扩大局部感受野。
#         # 配合上面的坐标注意力(全局信息)，形成了"全局+大局部"的多尺度结构。
#         self.conv_large = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False), # padding=2 保持尺寸
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU()
#         )

#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 分组处理
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        
#         # ==========================================
#         # [应用修改点 2] 混合池化前向传播
#         # ==========================================
#         # 将 AvgPool 和 MaxPool 的结果相加 (或者拼接，这里为了保持维度选择相加)
#         x_h = self.avg_pool_h(group_x) + self.max_pool_h(group_x)
#         x_w = self.avg_pool_w(group_x) + self.max_pool_w(group_x)
#         # 此时 x_w 需要调整维度
#         x_w = x_w.permute(0, 1, 3, 2)

#         # 1x1 分支交互
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)

#         # 生成带有坐标信息的特征 x1
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        
#         # ==========================================
#         # [应用修改点 1] 大核卷积前向传播
#         # ==========================================
#         # 使用 5x5 卷积替代原来的 3x3
#         x2 = self.conv_large(group_x)

#         # 下面是 EMA 原有的特征聚合逻辑 (Cross-Scale Aggregation)
#         # 计算全局描述符
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, hw
        
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, hw
        
#         # 矩阵乘法聚合
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        
#         # 最终输出
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))


# class Bottleneck(nn.Module):
#     """Standard bottleneck."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#         self.Attention = EMA(c2)

#     def forward(self, x):
#         """'forward()' applies the YOLO FPN to input data."""
#         return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



# class C2f_EMA(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))



