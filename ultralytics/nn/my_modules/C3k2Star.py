import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class StarBlock(nn.Module):
    """AMP-safe StarBlock (NO inplace, NO tensor reuse bug)."""

    def __init__(self, c1, c2, k=3, s=1, shortcut=True):
        super().__init__()

        self.conv1 = Conv(c1, c2, 1, 1)
        # ⚠️ 关键 1：关闭 Conv 内部的 inplace SiLU
        self.conv1.act.inplace = False

        self.dwconv = Conv(c2, c2, k, s, g=c2, act=False)

        # ⚠️ 关键 2：使用非 inplace 的 SiLU
        self.act = nn.SiLU(inplace=False)

        self.conv2 = Conv(c2, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        res = x

        x = self.conv1(x)

        # ⚠️ 关键 3：分支前 clone，切断 Autograd version 依赖
        x_dw = self.dwconv(x.clone())
        x_act = self.act(x)

        x = x_dw * x_act
        x = self.conv2(x)

        return x + res if self.add else x


class C3kStar(nn.Module):
    """C3k 的 StarBlock 版本封装。 当 c3k=True 时调用此模块，允许更灵活的核大小配置（虽然默认通常还是3）。.
    """

    def __init__(self, c1, c2, k=3, step=1):
        super().__init__()
        # 这里可以直接调用 StarBlock，保持接口灵活性
        self.m = StarBlock(c1, c2, k=k, shortcut=True)

    def forward(self, x):
        return self.m(x)


class C3k2Star(nn.Module):
    """基于 YOLOv11 C3k2 改良，核心模块替换为 StarBlock."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # C2f/C3k2 的核心机制：输入 n+2 个 tensor 进行 concat，所以输入通道是 (2+n)*c
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 核心修复逻辑：
        # 如果 c3k=True，使用 C3kStar 包装器（通常用于支持可变卷积核）
        # 如果 c3k=False，直接使用 StarBlock
        # 注意：这里我们默认 k=3，如果需要更小的核，可以在 C3kStar 初始化时调整
        self.m = nn.ModuleList(
            C3kStar(self.c, self.c, k=3) if c3k else StarBlock(self.c, self.c, k=3, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x):
        # 1. Split: 将 cv1 的输出分为两份
        y = list(self.cv1(x).chunk(2, 1))
        # 2. Extend: 将后半部分依次通过模块 m，并将所有中间结果存入 list
        y.extend(m(y[-1]) for m in self.m)
        # 3. Concat & Output: 将所有特征图拼接并通过 cv2
        return self.cv2(torch.cat(y, 1))
