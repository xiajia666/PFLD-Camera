#!/usr/bin/env python
# coding:utf-8
"""
Name : mobilenetv3.py
Author  : @ Chenxr
Create on   : 2021/2/8 22:45
Desc: None
"""
'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        # print(self.se(x))
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class Block_DCN(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_DCN, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = DeformConv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()


        # self.bneck1 = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
        #     Block(3, 16, 64, 32, nn.ReLU(inplace=True), None, 2),
        #     Block(3, 32, 72, 32, nn.ReLU(inplace=True), None, 1),
        # )
        # self.bneck2 = nn.Sequential(
        #     Block(5, 32, 72, 64, nn.ReLU(inplace=True), SeModule(64), 2),
        #     Block(5, 64, 120, 64, nn.ReLU(inplace=True), SeModule(64), 1),
        #     Block(5, 64, 120, 64, nn.ReLU(inplace=True), SeModule(64), 1),
        # )
        #
        # self.bneck3 = nn.Sequential(
        #     Block(3, 64, 240, 80, hswish(), None, 2),
        #     Block(3, 80, 200, 80, hswish(), None, 1),
        #     Block(3, 80, 184, 80, hswish(), None, 1),
        #     Block(3, 80, 184, 128, hswish(), None, 1),
        # )
        # self.bneck4= nn.Sequential(
        #     Block(3, 128, 480, 128, hswish(), SeModule(128), 1),
        #     Block(3, 128, 672, 128, hswish(), SeModule(128), 1),
        #     Block(5, 128, 672, 256, hswish(), SeModule(256), 1),
        #     Block(5, 256, 672, 256, hswish(), SeModule(256), 2),
        #     Block(5, 256, 960, 256, hswish(), SeModule(256), 1),
        # )

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 32, nn.ReLU(inplace=True), None, 2),
            Block(3, 32, 72, 32, nn.ReLU(inplace=True), None, 1),
            Block(5, 32, 72, 64, nn.ReLU(inplace=True), SeModule(64), 2),
            Block(5, 64, 120, 64, nn.ReLU(inplace=True), SeModule(64), 1),
            Block(5, 64, 120, 64, nn.ReLU(inplace=True), SeModule(64), 1),
        )


        self.bneck2 = nn.Sequential(
            Block(3, 64, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 128, hswish(), None, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(3, 128, 480, 128, hswish(), SeModule(128), 1),
            Block(3, 128, 672, 128, hswish(), SeModule(128), 1),
            Block(5, 128, 672, 256, hswish(), SeModule(256), 1),
            Block(5, 256, 672, 256, hswish(), SeModule(256), 2),
            Block(5, 256, 960, 256, hswish(), SeModule(256), 1),
        )

        self.conv2 = nn.Conv2d(256, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # print('x1',x.shape)
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck1(out)
        # print('x2', out.shape)
        out = self.bneck2(out)
        # print('x3', out.shape)
        out = self.bneck3(out)
        # print('x4', out.shape)
        # out = self.bneck4(out)
        # print('x5', out.shape)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

if __name__ == '__main__':
    net = MobileNetV3()
    x = torch.randn(2,3,224,224)
    y = net(x)
    #print(y)

# test()




