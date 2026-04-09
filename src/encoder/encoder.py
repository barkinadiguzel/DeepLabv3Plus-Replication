import torch
import torch.nn as nn
from backbone.xception import AlignedXception
from blocks.aspp import ASPP

class Encoder(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, aspp_channels=256):
        super().__init__()
        self.backbone = AlignedXception(in_channels, output_stride)
        self.aspp = ASPP(1024, aspp_channels, output_stride=output_stride)

    def forward(self, x):
        features = self.backbone(x)
        x = self.aspp(features)
        return x, features 
