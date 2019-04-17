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
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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

class UpLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        self.upsample = nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=2, stride=2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, bx):
        x = self.upsample(x)
        x = torch.cat((x, bx), 1)
        x = self.conv(x)
        return x

class UNetPlusPlus(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetPlusPlus, self).__init__()
        self.conv = DoubleConv(in_channels, 64)
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(4):
            down_conv = DownLayer(64 * 2 ** i, 64 * 2 ** (i + 1))
            self.down_convs.append(down_conv)
        for tot in range(1, 5):
            for j in range(1, tot + 1):
                i = tot - j
                up_conv = UpLayer(64 * (j+1) * 2 ** i, 64 * 2 ** i)
                self.up_convs.append(up_conv)
        self.out_convs = nn.ModuleList()
        for i in range(4):
            self.out_convs.append(Conv11(64, num_classes))

    def forward(self, x):
        x_in = self.conv(x)
        x = [[0 for i in range(5)] for j in range(5)]
        x[0][0] = x_in
        for i, module in enumerate(self.down_convs):
            x[i + 1][0] = module(x[i][0])

        name = []
        for tot in range(1, 5):
            for j in range(1, tot+1):
                name.append((tot-j, j))

        for id, module in enumerate(self.up_convs):
            i, j = name[id]
            xs = x[i][0]
            for k in range(1, j):
                xs = torch.cat((xs, x[i][k]), 1)
            x[i][j] = module(x[i + 1][j - 1], xs)
        
        for i, module in enumerate(self.out_convs):
            x[0][i + 1] = module(x[0][i + 1])
        
        for i in range(2, 5):
            x[0][1] += x[0][i] 
        
        x[0][1] = x[0][1] / 4
        return x[0][1]
        
        
        
