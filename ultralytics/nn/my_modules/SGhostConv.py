import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
class SGhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=3, s=2, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()

        # print("***", c1, c2, k, s, g, act)
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)

        gate = torch.sigmoid(self.cv2(y))
        out_star = y * gate 
        
        # 4. 为了凑够 c2 通道 (保持 GhostConv 的行为)，我们将 y 和 out_star 拼接
        # y: 原始信息, out_star: 注意力激活后的信息
        return torch.cat((y, out_star), 1)
        # return torch.cat((y, self.cv2(y)), 1)