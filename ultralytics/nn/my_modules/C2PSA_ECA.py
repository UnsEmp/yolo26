import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SRMHSA2D(nn.Module):
    """Spatial-Reduction Multi-Head Self-Attention for 2D feature maps. Q: full resolution, K/V: pooled by sr_ratio.
    """

    def __init__(self, dim, num_heads=4, sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.sr = nn.Identity()

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        q = self.q(x)  # (B,C,H,W)
        q = q.flatten(2).transpose(1, 2)  # (B, HW, C)

        x_sr = self.sr(x)
        kv = self.kv(x_sr)  # (B,2C,H',W')
        kv = kv.flatten(2).transpose(1, 2)  # (B, H'W', 2C)
        k, v = kv.chunk(2, dim=-1)  # (B, H'W', C) each

        # reshape to heads
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, HW, d)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, H'W', d)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,heads,HW,H'W')
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B,heads,HW,d)

        out = out.transpose(1, 2).contiguous().view(B, H * W, C)  # (B,HW,C)
        out = out.transpose(1, 2).view(B, C, H, W)  # (B,C,H,W)
        return self.proj(out)


class LiteFFN(nn.Module):
    """Mobile-friendly FFN: PW -> DW -> PW."""

    def __init__(self, c, expansion=2.0, drop=0.0):
        super().__init__()
        hidden = int(c * expansion)
        self.pw1 = Conv(c, hidden, k=1, s=1)
        self.dw = Conv(hidden, hidden, k=3, s=1, g=hidden)
        self.pw2 = Conv(hidden, c, k=1, s=1, act=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pw2(self.dw(self.pw1(x))))


class PSABlock_SRA(nn.Module):
    """PSA-like block using SRMHSA + lightweight FFN."""

    def __init__(self, c, num_heads=4, sr_ratio=2, ffn_expansion=2.0):
        super().__init__()
        self.attn = SRMHSA2D(c, num_heads=num_heads, sr_ratio=sr_ratio)
        self.ffn = LiteFFN(c, expansion=ffn_expansion)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2PSA_ECA(nn.Module):
    """C2PSA with spatial-reduction attention (lighter than full MHSA)."""

    def __init__(self, c1, c2, n=1, e=0.5, sr_ratio=2, ffn_expansion=2.0):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)

        # 关键：避免 num_heads=0
        num_heads = max(1, self.c // 64)
        # 也可更激进：限制最大头数，提升端侧速度
        num_heads = min(num_heads, 8)

        self.m = nn.Sequential(
            *(
                PSABlock_SRA(self.c, num_heads=num_heads, sr_ratio=sr_ratio, ffn_expansion=ffn_expansion)
                for _ in range(n)
            )
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))
