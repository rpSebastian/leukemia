import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DoubleConv, Up, Conv11
class UNetPP(torch.nn.Module):
    def __init__(self, in_channels = 3, num_classes = 2, num_layers = 4):
        super(UNetPP, self).__init__()
        self.num_layers = num_layers
        features = [64, 128, 256, 512, 1024]
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.conv00 = DoubleConv(3, features[0])
        self.conv10 = DoubleConv(features[0], features[1])
        self.up01 = Up(features[1], features[0], 1)
        self.outconv1 = Conv11(features[0], num_classes)

        self.conv20 = DoubleConv(features[1], features[2])
        self.up11 = Up(features[2], features[1], 1)
        self.up02 = Up(features[1], features[0], 2)
        self.outconv2 = Conv11(features[0], num_classes)

        self.conv30 = DoubleConv(features[2], features[3])
        self.up21 = Up(features[3], features[2], 1)
        self.up12 = Up(features[2], features[1], 2)
        self.up03 = Up(features[1], features[0], 3)
        self.outconv3 = Conv11(features[0], num_classes)

        self.conv40 = DoubleConv(features[3], features[4])
        self.up31 = Up(features[4], features[3], 1)
        self.up22 = Up(features[3], features[2], 2)
        self.up13 = Up(features[2], features[1], 3)
        self.up04 = Up(features[1], features[0], 4)
        self.outconv4 = Conv11(features[0], num_classes)    
        self.reset_params()
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        out = 0
        if self.num_layers >= 1:
            x00 = self.conv00(x)
            x10 = self.conv10(self.maxpool(x00))
            x01 = self.up01(x10, [x00])
            out1 = self.outconv1(x01)
            out += out1

        if self.num_layers >= 2:
            x20 = self.conv20(self.maxpool(x10))
            x11 = self.up11(x20, [x10])
            x02 = self.up02(x11, [x00, x01])
            out2 = self.outconv2(x02)
            out += out2
        
        if self.num_layers >= 3:
            x30 = self.conv30(self.maxpool(x20))
            x21 = self.up21(x30, [x20])
            x12 = self.up12(x21, [x10, x11])
            x03 = self.up03(x12, [x00, x01, x02])
            out3 = self.outconv3(x03)
            out += out3

        if self.num_layers >= 4:
            x40 = self.conv40(self.maxpool(x30))
            x31 = self.up31(x40, [x30])
            x22 = self.up22(x31, [x20, x21])
            x13 = self.up13(x22, [x10, x11, x12])
            x04 = self.up04(x13, [x00, x01, x02, x03])
            out4 = self.outconv4(x04)
            out += out4

        out /= self.num_layers
        return out 
