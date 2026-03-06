import torch
import torch.nn as nn

from ultralytics.nn.modules import GhostConv


class Gcbv4(nn.Module):
    """GCBv4 (Ghost Conv Branch v4) 结构逻辑: 1. Input (c1) -> 分为两个主分支 2. Branch 1: GhostConv (c1 -> c1) 3. Branch 2: -
    Split: 切分为两半 (c1/2, c1/2) - Sub-branch 2a: MaxPool (保留纹理/边界) - Sub-branch 2b: AvgPool (保留背景) * Learnable
    Parameter - Combine: Element-wise Add (2a + 2b) - BatchNorm 4. Concat: Branch 1 (c1) + Branch 2 (c1/2) = 1.5 *
    c1 通道 5. Final: GhostConv (1.5*c1 -> c2) 调整回目标大小.
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        assert c1 % 2 == 0, f"GCBv4 输入通道数必须是偶数，但收到了 {c1}"

        c_half = c1 // 2

        # --- Branch 1 ---
        # 完整的 GhostConv 操作，保持原通道数 c1
        self.branch1_conv = GhostConv(c1, c1, k=3, s=s)

        # --- Branch 2 ---
        # 也就是 split 后的操作
        # MaxPool 保持尺寸不变 (padding=k//2)
        self.b2_maxpool = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)
        # AvgPool 保持尺寸不变
        self.b2_avgpool = nn.AvgPool2d(kernel_size=k, stride=s, padding=k // 2)

        # 可学习参数 (Learnable Parameter)
        # 初始化为 0.5 或者 1.0，允许网络学习 AvgPool 的权重
        # shape 为 [1] 表示对所有通道统一加权，也可设为 [1, c_half, 1, 1] 进行通道级加权
        self.alpha = nn.Parameter(torch.ones(1, c_half, 1, 1) * 0.5)

        # Batch Normalization 模块
        self.b2_bn = nn.BatchNorm2d(c_half)

        # --- Final ---
        # 此时通道变成了 c1 (Branch1) + c1/2 (Branch2) = 1.5 * c1
        c_concat = c1 + c_half
        self.final_conv = GhostConv(c_concat, c2, k=1, s=1)

    def forward(self, x):
        # --- Branch 1 ---
        y1 = self.branch1_conv(x)

        # --- Branch 2 ---
        # 1. Split 操作：在通道维度(dim=1)切分成两份
        x_split_a, x_split_b = x.chunk(2, 1)

        # 2. 分别池化
        out_max = self.b2_maxpool(x_split_a)
        out_avg = self.b2_avgpool(x_split_b)

        # 3. 结合：Max + (Avg * alpha) -> BN
        # 这里实现了你要求的"加入一个可学习参数与maxpool结果结合"
        y2 = out_max + (out_avg * self.alpha)
        y2 = self.b2_bn(y2)

        # --- Concat ---
        # 结合 Branch 1 和 Branch 2
        # y1 是 c1, y2 是 c1/2 -> 结果是 1.5 * c1
        y_cat = torch.cat((y1, y2), 1)

        # --- Final Output ---
        return self.final_conv(y_cat)
