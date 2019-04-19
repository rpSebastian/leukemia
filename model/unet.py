import torch
import torch.nn as nn
from .layers import DoubleConv, Up, Conv11

class UNet(torch.nn.Module):
    def __init__(self, in_channels = 3, num_classes = 3):
        super(UNet, self).__init__()
        features = [32, 64, 128, 256, 512]
        self.conv1 = DoubleConv(in_channels, features[0])
        self.conv2 = DoubleConv(features[0], features[1])
        self.conv3 = DoubleConv(features[1], features[2])
        self.conv4 = DoubleConv(features[2], features[3])
        self.conv5 = DoubleConv(features[3], features[4])
        self.up4 = Up(features[4], features[3])
        self.up3 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up1 = Up(features[1], features[0])
        self.outconv = Conv11(features[0], 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.reset_params()
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))
        x = self.up4(x5, [x4])  
        x = self.up3(x, [x3]) 
        x = self.up2(x, [x2]) 
        x = self.up1(x, [x1])
        x = self.outconv(x)
        return x

