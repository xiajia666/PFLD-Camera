import torch
import torch.nn.functional as F
import torch.nn as nn

def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))

#
# class dilated(nn.Module):
#     def __init__(self, in_channels):
#         super(dilated, self).__init__()
#
#         # assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
#         # if output_stride == 16:
#         #     dilations = [1, 6, 12, 18]
#         # elif output_stride == 8:
#         dilations = [1, 2, 4, 8]
#
#         self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
#         self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
#         self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
#         self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])
#
#         self.avg_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(in_channels, 256, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True))
#
#         self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(0.5)
#
#         # initialize_weights(self)
#
#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
#
#         x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.dropout(self.relu(x))
#
#         return x

def conv_bn(inp, oup, kernel_size, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True))

def conv_bn1(inp, oup, kernel_size, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True))

class dilated(nn.Module):
    def __init__(self, in_channels):
        super(dilated, self).__init__()

        # assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        # if output_stride == 16:
        #     dilations = [1, 6, 12, 18]
        # elif output_stride == 8:
        dilations = [2, 4, 6, 8]
        leaky = 0
        self.aspp1 = assp_branch(in_channels, 256, 3, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.conv1 = conv_bn(in_channels, 256, 1, stride=1, leaky=leaky)
        self.conv2 = conv_bn1(in_channels, 256, 3, stride=1, leaky=leaky)

        # initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)

        x = self.conv1(x1)
        x = self.aspp1(x)
        x = self.conv1(x)

        x2 = x + x1

        x = self.conv1(x2)
        x = self.aspp2(x)
        x = self.conv1(x)
        x3 = x + x2

        x = self.conv1(x3)
        x = self.aspp3(x)
        x = self.conv1(x)
        x4 = x + x3

        x = self.conv1(x4)
        x = self.aspp4(x)
        x = self.conv1(x)
        x5 = x + x4

        # x = self.dropout(self.relu(x))

        return x5