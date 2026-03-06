import torch
import torch.nn as nn


class RepGhostConv(nn.Module):
    """训练时：多分支提取特征 推理时：融合为一个 Conv 注意：这是一个简化版逻辑，需要配合 deploy() 方法使用.
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.act = act

        # Primary: 主分支 + 1x1 旁路
        self.cv1_main = nn.Conv2d(c1, c_, k, s, padding=k // 2, bias=False)
        self.cv1_1x1 = nn.Conv2d(c1, c_, 1, s, bias=False) if k != 1 else None
        self.bn1 = nn.BatchNorm2d(c_)

        # Cheap: 3x3 DW + 1x1 DW 旁路
        self.cv2_main = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False)
        self.cv2_1x1 = nn.Conv2d(c_, c_, 1, 1, 0, groups=c_, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)

        self.act_func = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 1. Primary Part (Rep style)
        y1 = self.cv1_main(x)
        if self.cv1_1x1:
            y1 += self.cv1_1x1(x)
        y1 = self.bn1(y1)
        y1 = self.act_func(y1)

        # 2. Cheap Part (Rep style)
        y2 = self.cv2_main(y1) + self.cv2_1x1(y1)
        y2 = self.bn2(y2)
        y2 = self.act_func(y2)

        return torch.cat((y1, y2), 1)
