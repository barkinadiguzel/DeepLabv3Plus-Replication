import torch
import torch.nn as nn
import torch.nn.functional as F
from .depthwise_sep import DepthwiseSeparableConv

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6,12,18]):
        super().__init__()
        self.rates = rates
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(out_channels)
        
        self.atrous_blocks = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in rates
        ])
        
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*(len(rates)+2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]  
        outputs = [self.bn_1x1(self.conv_1x1(x))]
        for atrous_block in self.atrous_blocks:
            outputs.append(atrous_block(x))
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=size, mode='bilinear', align_corners=False)
        outputs.append(img_pool)
        x = torch.cat(outputs, dim=1)
        x = self.project(x)
        return x
