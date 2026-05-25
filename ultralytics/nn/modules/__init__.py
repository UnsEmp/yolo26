# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""

from ..my_modules.A2C2fStar import A2C2fStar
from ..my_modules.A2C2fStarFast import A2C2fStarFast
from ..my_modules.C2PSA_ECA import C2PSA_ECA
from ..my_modules.C3GhostStart import C3GhostStar
from ..my_modules.C3k2Star import C3k2Star
from ..my_modules.MSAC2PSA import MSAC2PSA
from ..my_modules.SDA2C2f import SDA2C2f
from ..my_modules.SGhostConv import SGhostConv
from ..my_modules.SimSPPF import SimSPPF
from ..my_modules.SPDSGhost import SPDSGhost
from ..my_modules.SSGhostConv import SSGhostConv
from ..new_modules.Attention import CBAM, ECA, GAM
from ..new_modules.CGAttention import C2f_CGA
from ..new_modules.DAttention import DAttentionBaseline
from ..new_modules.DLKAttention import C2f_DLKA, deformable_LKA
from ..new_modules.EMAttention import EMA
from ..new_modules.FocusedLinearAttention import C2f_FLA
from ..new_modules.HAttention import HAT
from ..new_modules.LSKAttention import LSKA
from ..new_modules.TripletAttention import C2f_TripletAt
from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    OBB26,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    Pose26,
    RTDETRDecoder,
    Segment,
    Segment26,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C2PSA_ECA",
    "C3",
    "C3TR",
    "CBAM",
    "CBAM",
    "CIB",
    "DFL",
    "ECA",
    "ELAN1",
    "EMA",
    "GAM",
    "HAT",
    "LSKA",
    "MLP",
    "MSAC2PSA",
    "OBB",
    "OBB26",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "A2C2f",
    "A2C2fStar",
    "A2C2fStarFast",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C2f_CGA",
    "C2f_DLKA",
    "C2f_FLA",
    "C2f_TripletAt",
    "C3Ghost",
    "C3GhostStar",
    "C3k2",
    "C3k2Star",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DAttentionBaseline",
    "DLKAttention",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    "Focus",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Pose26",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "SDA2C2f",
    "SGhostConv",
    "SPDSGhost",
    "SSGhostConv",
    "Segment",
    "Segment26",
    "SimSPPF",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "YOLOESegment26",
    "deformable_LKA",
    "v10Detect",
)
