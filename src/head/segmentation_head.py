import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        return self.conv(x)
