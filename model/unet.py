import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv11(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv11, self).__init__()
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 先进行maxpool， 再进行两次卷积
# in_channels --> out_channels
# 图像长宽各变为一半
class DownLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

# 首先进行上采样，再与同层的结果连接在一起
# in_channels --> inchannels * 2 -->  out_channels
# 图像长宽变为原来的两倍
class UpLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels * 2, out_channels)
    
    def forward(self, x, bx):
        # x = nn.functional.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upsample(x)
        x = torch.cat((x, bx), 1)
        x = self.conv(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = DoubleConv(3, 32)
        self.down1 = DownLayer(32, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256, 256)
        self.up4 = UpLayer(256, 128)
        self.up3 = UpLayer(128, 64)
        self.up2 = UpLayer(64, 32)
        self.up1 = UpLayer(32, 32)
        self.conv2 = Conv11(32, 2)
        self.reset_params()
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        # x --> batch * 3 * 512 * 512  # 2M
        x1 = self.conv(x)    # x1 --> batch * 64 * 512 * 512 # 128M
        x2 = self.down1(x1)  # x2 --> batch * 128 * 256 * 256 # 64M
        x3 = self.down2(x2)  # x3 --> batch * 256 * 128 * 128 # 32M
        x4 = self.down3(x3)  # x4 --> batch * 512 * 64 * 64 # 16M
        x = self.down4(x4)   # x --> batch * 1024 * 32 * 32 # 4M
        x = self.up4(x, x4)  # x --> batch * 256 * 64 * 64 # 8M
        x = self.up3(x, x3)  # x --> batch * 128 * 128 * 128 # 16M
        x = self.up2(x, x2)  # x --> batch * 64 * 256 * 256 # 32M
        x = self.up1(x, x1)  # x --> batch * 64 * 512 * 512 # 128M
        x = self.conv2(x)    # x --> batch * num_classes * 512 * 512
        return x

