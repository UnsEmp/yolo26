import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class SSGhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        assert c2 % 4 == 0, "c2 must be divisible by 4"

        c_ = c2 // 4

        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)

        self.cv2 = nn.ModuleList([
            # 3x3 DWConv
            Conv(c_, c_, 3, 1, None, c_, act=act),

            # 3x3 DWConv + dilation=2 (等效 5x5)
            nn.Conv2d(
                c_, c_,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                groups=c_,
                bias=False
            ),

            # Identity
            nn.Identity()
        ])

    def forward(self, x):
        y = self.cv1(x)

        out = [y]
        for op in self.cv2:
            out.append(op(y))

        res = torch.cat(out, dim=1)
        return self.channel_shuffle(res, 4)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)
