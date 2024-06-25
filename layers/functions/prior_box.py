import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object): # 一些预先设置好的在图片上的方框，网络的预测结果只是对这些先验框进行判断进而调整。每一个网格点有两个框，20*20的有效特征层一共具有400个特征点，800个先验框
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes'] # 先验框基础的边长
        self.steps = cfg['steps']  # 三个有效特征层对输入图片长和宽压缩的倍数
        self.clip = cfg['clip'] # 是否在生成先验框之后clip在【0,1】之间
        self.image_size = image_size # 根据图片的大小生成先验框
        # 三个有效特征层的高和宽
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps): # 对所有特征层进行循环
            min_sizes = self.min_sizes[k] # 取出每一个特征层对应的先验框
            # 每个网格点2个先验框，都是正方形
            for i, j in product(range(f[0]), range(f[1])): # 对有效特征层对应的网格点进行遍历，
                for min_size in min_sizes:
                    # 将先验框映射到网格点上
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        # 中心宽高的形式
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
