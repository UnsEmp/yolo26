import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class StarInteraction(nn.Module):
    """核心创新单元：Star Interaction Block 结合了 深度卷积(Spatial) + 元素乘法(High-dim feature) + 通道缩放.
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        # 1. 深度卷积提取空间特征
        self.dwconv = Conv(c1, c1, k, s, g=c1, act=False)

        # 2. 元素乘法所需的双路投影
        # 我们需要两个分支来进行 point-wise multiplication
        self.f1 = Conv(c1, c2, 1, 1, act=False)
        self.f2 = Conv(c1, c2, 1, 1, act=False)

        # 3. 激活函数
        self.act = nn.ReLU6()  # 或者 nn.SiLU()

    def forward(self, x):
        # 空间建模
        x_spatial = self.dwconv(x)

        # Star Operation:
        # 分支1 (变换) * 分支2 (门控/激活)
        # 这种 x1 * x2 的操作能产生隐式的高维特征
        out = self.f1(x_spatial) * self.act(self.f2(x_spatial))

        return out


class A2C2fStar(nn.Module):
    """Ultralytics-safe A2C2fStar - repeat 交给 YAML - 内部不做 dense concat.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * c, 1, 1)
        self.cv2 = Conv(2 * c, c2, 1, 1)

        self.m = StarInteraction(c, c)

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, 1)
        y2 = self.m(y2)
        return self.cv2(torch.cat((y1, y2), 1))
