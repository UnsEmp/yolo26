import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class GhostStarBlock(nn.Module):
    """GhostStarBlock: 结合 Ghost 的轻量化 (DWConv) 和 StarNet 的高维映射 (Element-wise Mul) 结构: Conv(1x1) -> DWConv(3x3) -> Split
    -> Mul -> Conv(1x1).
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        # 1. 升维/特征变换 (1x1 Conv)
        # 这里我们将通道数扩大，以便后续做 Split 操作
        # mid_c 决定了“星空”的宽度
        mid_c = c2 * 2
        self.conv1 = Conv(c1, mid_c, 1, 1)

        # 2. 空间特征提取 (Depthwise Conv)
        # 极其廉价的操作，捕捉空间上下文
        self.dwconv = Conv(mid_c, mid_c, k, s, g=mid_c, act=False)

        # 3. 降维/输出 (1x1 Conv)
        # 不使用激活函数，因为前面的 Mul 已经提供了足够强的非线性
        self.conv2 = Conv(c2, c2, 1, 1, act=False)

    def forward(self, x):
        res = x

        # 1. 升维 + 空间提取
        x = self.conv1(x)
        x = self.dwconv(x)

        # 2. Star Operation (核心创新)
        # 将特征图沿通道一分为二
        x1, x2 = x.chunk(2, 1)
        # 元素乘法：x1 * ReLU(x2) 或 x1 * x2
        # 这步操作模拟了高维特征映射，显著提升特征丰富度
        x = x1 * x2

        # 3. 降维输出
        x = self.conv2(x)

        # 残差连接 (如果通道数匹配)
        return x + res if x.shape == res.shape else x


class C3GhostStar(nn.Module):
    """CSP结构封装 GhostStarBlock 比 C2f 轻量，比 C3Ghost 精度高.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 堆叠 GhostStarBlock
        self.m = nn.ModuleList(GhostStarBlock(self.c, self.c) for _ in range(n))

    def forward(self, x):
        # CSP 分支逻辑：一半走ResNet，一半直接连
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
