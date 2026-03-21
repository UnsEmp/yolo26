import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SPDSGhost(nn.Module):
    """SPD_SGhost: Space-to-Depth Spectral Ghost Module. Combines lossless downsampling (SPD) with efficient feature
    expansion (SGhost).

    Structure: Input -> SPD(Slice) -> [H/2, W/2, 4*C1]
          -> 1x1 Fusion (Primary) -> y
          -> 5x5 DWConv + Sigmoid (Gate) -> mask
          -> y * mask (Ghost Features)
          -> Concat(y, Ghost) -> Output
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Args:
            c1 (int): Input channels
            c2 (int): Output channels.
        """
        super().__init__()
        # SPD transform increases channels by 4
        c1_spd = c1 * 4

        # Output channels for the primary branch (y)
        # We need final output to be c2. Since we concat (y, y*gate),
        # the intermediate channel c_ must be c2 // 2.
        c_ = c2 // 2

        # 1. Primary Conv (Fusion):
        # Uses 1x1 kernel to efficiently fuse the 4 slices from SPD.
        # This is the "Primary" convolution in GhostNet terms.
        self.cv1 = Conv(c1_spd, c_, k=1, s=1, p=None, g=g, act=act)

        # 2. Cheap/Gated Operation (The "Ghost" Generator):
        # Uses 5x5 Depthwise Conv to capture context with minimal params.
        # Note: g=c_ ensures it is Depthwise.
        self.cv2 = Conv(c_, c_, k=5, s=1, p=None, g=c_, act=False)

        # Activation for the gate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------------
        # Step 1: Space-to-Depth (SPD)
        # ------------------------------------------------
        _B, _C, H, W = x.shape
        # Pad if dimensions are odd
        if H % 2 != 0 or W % 2 != 0:
            x = torch.nn.functional.pad(x, (0, W % 2, 0, H % 2))

        # Slicing (0::2 means start at 0, step 2)
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        # [B, 4*C, H/2, W/2]
        x_spd = torch.cat([x0, x1, x2, x3], dim=1)

        # ------------------------------------------------
        # Step 2: SGhost Processing
        # ------------------------------------------------
        # Primary features (intrinsic)
        y = self.cv1(x_spd)

        # Gated/Ghost features (cheap operation)
        # Using 5x5 DWConv to perceive broader context for the gate
        gate = self.sigmoid(self.cv2(y))
        out_star = y * gate

        # ------------------------------------------------
        # Step 3: Feature Concatenation
        # ------------------------------------------------
        return torch.cat((y, out_star), 1)
