import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from head.segmentation_head import SegmentationHead
from config import NUM_CLASSES, LOW_LEVEL_CHANNELS, DECODER_CHANNELS

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(low_level_channels=LOW_LEVEL_CHANNELS,
                               decoder_channels=DECODER_CHANNELS)
        self.head = SegmentationHead(DECODER_CHANNELS, num_classes)

    def forward(self, x):
        x, low_level = self.encoder(x)
        x = self.decoder(x, low_level)
        x = self.head(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)  
        return x
