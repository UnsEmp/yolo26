import torch
from torch import nn


def autopad(k, p=None, d=1):
    """自动填充（同YOLOv8）."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积（同YOLOv8）."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RCSA(nn.Module):
    """肉牛进食识别专用注意力机制：Ruminant Channel-Spatial Attention (RCSA) 输入: 特征图 (b, c, h, w) 输出: 加权特征图 (b, c, h, w).
    """

    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0, "channels must be divisible by factor"

        # 通道注意力分支：平均+最大池化 + 1x1卷积
        self.channel_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.channel_conv = nn.Conv2d(
            2 * (channels // self.groups), channels // self.groups, kernel_size=1, stride=1, padding=0
        )
        self.channel_gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.channel_sigmoid = nn.Sigmoid()

        # 空间-位置注意力分支：h/w分离池化 + 位置编码 + 3x3卷积
        self.spatial_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.spatial_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.spatial_conv1x1 = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0
        )
        self.spatial_conv3x3 = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1
        )
        self.spatial_gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.spatial_sigmoid = nn.Sigmoid()

        # 位置编码（Coordinate Attention简化版）
        self.pos_conv_h = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0
        )
        self.pos_conv_w = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # 1. 分组预处理
        group_x = x.reshape(b * self.groups, -1, h, w)  # (b*g, c//g, h, w)

        # ------------------- 通道注意力分支 -------------------
        gap = self.channel_gap(group_x)  # (b*g, c//g, 1, 1)
        gmp = self.channel_gmp(group_x)  # (b*g, c//g, 1, 1)
        channel_cat = torch.cat([gap, gmp], dim=1)  # (b*g, 2*c//g, 1, 1)
        channel_weight = self.channel_sigmoid(self.channel_gn(self.channel_conv(channel_cat)))  # (b*g, c//g, 1, 1)

        # ------------------- 空间-位置注意力分支 -------------------
        # h/w分离池化（同EMA）
        x_h = self.spatial_pool_h(group_x)  # (b*g, c//g, h, 1)
        x_w = self.spatial_pool_w(group_x).permute(0, 1, 3, 2)  # (b*g, c//g, w, 1)
        hw_cat = torch.cat([x_h, x_w], dim=2)  # (b*g, c//g, h+w, 1)
        hw_fusion = self.spatial_conv1x1(hw_cat)  # (b*g, c//g, h+w, 1)
        x_h, x_w = torch.split(hw_fusion, [h, w], dim=2)  # 拆分回h/w方向

        # 位置编码（简化版CA）
        pos_h = self.pos_conv_h(x_h).sigmoid()  # (b*g, c//g, h, 1)
        pos_w = self.pos_conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()  # (b*g, c//g, 1, w)

        # 空间特征融合
        spatial_x = group_x * pos_h * pos_w  # 位置编码加权
        spatial_x = self.spatial_conv3x3(spatial_x)  # 3x3卷积提取局部特征
        spatial_weight = self.spatial_sigmoid(self.spatial_gn(spatial_x))  # (b*g, c//g, h, w)

        # ------------------- 权重融合与输出 -------------------
        fusion_weight = channel_weight * spatial_weight  # 并行融合（元素级相乘）
        out = group_x * fusion_weight  # 加权分组特征
        return out.reshape(b, c, h, w)  # 还原原始维度


# ------------------- 集成到YOLOv8的C2f模块 -------------------
class Bottleneck_RCSA(nn.Module):
    """带RCSA的Bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.attention = RCSA(c2)  # 替换原有的EMA/CBAM

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C2f_RCSA(nn.Module):
    """带RCSA的C2f模块（YOLOv8兼容）."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_RCSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    from thop import profile

    # 1. 测试RCSA模块
    rcsa = RCSA(channels=256, factor=32)
    x = torch.randn(1, 256, 32, 32)  # 模拟YOLOv8 P4特征图
    with torch.no_grad():
        out = rcsa(x)
        print(f"RCSA输入维度: {x.shape}, 输出维度: {out.shape}")  # 维度需一致

    # 2. 计算量/参数量分析
    flops, params = profile(rcsa, inputs=(x,))
    print(f"RCSA FLOPs: {flops / 1e6:.2f}M, 参数数量: {params / 1e3:.2f}K")

    # 3. 测试C2f_RCSA模块
    c2f_rcsa = C2f_RCSA(c1=256, c2=256, n=1)
    out_c2f = c2f_rcsa(x)
    print(f"C2f_RCSA输入维度: {x.shape}, 输出维度: {out_c2f.shape}")
