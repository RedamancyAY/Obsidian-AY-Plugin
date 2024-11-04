import torch.nn as nn
import torch
import math


# +
class Attention1D(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super().__init__()
        self.b = b 
        self.gama = gama
        self.in_channel = in_channel

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2 == 0:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channel, in_channel, 1, bias=False)

    def forward(self, x):

        short_cut = x
        x = self.pool(x)  # (B, C, 1)
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = short_cut * x
        return x
    
# x = torch.randn(2, 64, 750)
# m = Attention1D(64)
# m(x).shape

# +
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_avg = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_max = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1, 1)

        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc_max(y_max).view(b, c, 1, 1)

        y = self.sigmoid(y_avg + y_avg)
        return x * y.expand_as(x)


class ChannelPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max, _ = torch.max(x, 1, keepdim=True)
        x_avg = torch.mean(x, 1).unsqueeze(1)
        x = torch.concat([x_max, x_avg], dim=1)
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x):
        x_out = self.ChannelAttention(x)
        x_out = self.SpatialAttention(x_out)
        return x_out

# +
# m = CBAM(64)
# x = torch.randn(2, 64, 33, 33)
# m(x).shape
