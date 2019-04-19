import torch
import torch.nn as nn

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

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_low=1):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv((n_low + 1) * out_channels, out_channels)
    
    def forward(self, high, low):
        high = self.upsample(high)
        low.append(high)
        x = torch.cat(low, 1)
        x = self.conv(x)
        return x