import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, scale_factor=None, target_size=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size

    def forward(self, x):
        if self.target_size is not None:
            return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        else:
            return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
