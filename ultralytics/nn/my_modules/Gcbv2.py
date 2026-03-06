import torch.nn as nn

from ultralytics.nn.modules import GhostConv


class Gcbv1(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.cv1 = GhostConv(c1, c2, 3, 1)
        self.cv2 = GhostConv(c2, c2, 3, 1)
        self.cv3 = GhostConv(c1, c2, 3, 1)

        self.pool_max = nn.MaxPool2d(3, 1, 1)
        self.pool_avg = nn.AvgPool2d(3, 1, 1)

        self.cv_align = GhostConv(c1, c2, 1, 1)

    def forward(self, x):
        x1 = self.pool_max(self.cv1(x))  # C = c2
        x2 = self.pool_avg(x)  # C = c1
        x2 = self.cv_align(x2)  # C = c2
        x = self.cv3(x)

        out = x + x1 + x2
        out = self.cv2(out)
        return out
