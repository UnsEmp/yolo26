import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class StarBlock(nn.Module):
    """Gated Star Operation (StarNet-style)."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class SDA2C2f(nn.Module):
    """Star-Dynamic-A2C2f (YOLOv12 compatible)."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        a2=True,  # 占位，对齐原 A2C2f
        area=1,
        residual=True,
        mlp_ratio=2.0,  # 占位
        e=0.5,
        g=1,  # 占位
        shortcut=True,  # 占位
    ):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((n + 2) * c_, c2, 1)

        self.use_residual = residual and c1 == c2

        # self.gamma = nn.Parameter(0.01 * torch.ones(c2)) if residual else None
        self.gamma = nn.Parameter(0.01 * torch.ones(c2)) if self.use_residual else None

        # Texture / Pose branch
        self.blocks = nn.ModuleList(nn.Sequential(DynamicA2Block(c_, area), StarBlock(c_)) for _ in range(n))

        # Structure prior branch
        self.structure_branch = WTConv(c_)

    def forward(self, x):
        y = [self.cv1(x)]

        for block in self.blocks:
            y.append(block(y[-1]))

        y.append(self.structure_branch(y[0]))

        out = self.cv2(torch.cat(y, dim=1))

        if self.use_residual:
            return x + self.gamma.view(1, -1, 1, 1) * out

        return out


class WTConv(nn.Module):
    """Wavelet-like Low-Frequency Structure Conv (Safe Approximation)."""

    def __init__(self, c):
        super().__init__()
        self.low_pass = nn.AvgPool2d(2, stride=2)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        x_low = self.low_pass(x)
        x_low = nn.functional.interpolate(x_low, size=x.shape[-2:], mode="nearest")
        return self.conv(x_low)


class DynamicA2Block(nn.Module):
    """Dynamic Area Attention Block."""

    def __init__(self, c, area=1):
        super().__init__()
        self.area = area

        self.q = nn.Conv2d(c, c, 1)
        self.k = DynamicConv(c)
        self.v = DynamicConv(c)

        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # reshape to (B, C, HW)
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H, W)

        return self.proj(out)


class DynamicConv(nn.Module):
    """Lightweight Dynamic Convolution (Safe Version)."""

    def __init__(self, c):
        super().__init__()
        self.kernel_gen = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.conv = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)

    def forward(self, x):
        weight = self.kernel_gen(x)
        return self.conv(x * weight)
