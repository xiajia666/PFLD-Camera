import torch.nn.functional as F
import torch.nn as nn
import torch
from models.ASPP import dilated as dilated
import torch
import torch.nn as nn


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

"""
def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # with open('./input.txt', 'r+') as fn:
        #     fn.write(str(input))
        # print(input)
        # names = list(input.keys())
        input = list(input.values())
        # with open('./g.txt', 'r+') as fn:
        #     fn.write(str(input))
        # input = list(input)

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNetV1()
    model.eval()
    # print(model)
    input = torch.randn(32, 3, 224, 224)
    #y = model(input)
    # print(y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1): # 深度可分离卷积与普通卷积构成
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )




class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        # 普通的3*3卷积，通道数为32
        conv3X3 = self.conv3X3(input)
        # 两次3*3卷积的堆叠代替5*5卷积，默认卷积核大小为3*3，通道数为16
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        # 三次3*3卷积的堆叠代替7*7卷积，默认卷积核大小为3*3，通道数为16
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        # 将三个并行结构堆叠，激活函数激活
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

# class FPN(nn.Module):
#     def __init__(self,in_channels_list,out_channels):
#         super(FPN,self).__init__()
#         leaky = 0
#         if (out_channels <= 64):
#             leaky = 0.1
#
#         self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
#         self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
#         self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)
#         self.output4 = conv_bn1X1(in_channels_list[3], out_channels, stride= 1, leaky=leaky)
#         self.deliated = dilated(in_channels_list[3])
#
#         self.cbam1 = CBAMLayer(in_channels_list[0])
#         self.cbam2 = CBAMLayer(in_channels_list[1])
#         self.cbam3 = CBAMLayer(in_channels_list[2])
#
#         self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
#         self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)
#         self.merge3 = conv_bn(out_channels, out_channels, leaky=leaky)
#
#     def forward(self, input):
#         # names = list(input.keys())
#         input = list(input.values())
#         # 对获取到的三个有效特征层利用三个1*1的卷积进行通道数的调整，64
#         # print('0:',input[0].shape)
#         # print('1:', input[1].shape)
#         # print('2:', input[2].shape)
#         # print('3:',input[3].shape)
#         output11 = self.cbam1(input[0])
#         output1 = self.output1(output11)
#
#         output12 = self.cbam2(input[1])
#         output2 = self.output2(output12)
#
#         output13 = self.cbam3(input[2])
#         output3 = self.output3(output13)
#
#         output = self.deliated(input[3])
#         output4 = self.output4(output)
#
#         up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="bilinear")
#         output3 = output3 + up4
#         # 利用64通道的卷积进行特征整合
#         output3 = self.merge3(output3)
#
#         # 对最小的有效特征层进行上采样
#         up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="bilinear")
#         output2 = output2 + up3
#         # 利用64通道的卷积进行特征整合
#         output2 = self.merge2(output2)
#
#         up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="bilinear")
#         output1 = output1 + up2
#         output1 = self.merge1(output1)
#
#         # out = [output1, output2, output3]
#         out = [output1, output2, output3, output4]
#         return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)


    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])


        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="bilinear")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="bilinear")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            # 320,320,8 -> 320,320,16
            conv_dw(8, 16, 1),   # 7
            # 320,320,16 -> 160,160,32
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            # 160,160,32 -> 80,80,64
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43  C3

        )
        self.stage2 = nn.Sequential(
            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219  C4
        )
        self.stage3 = nn.Sequential(
            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301  C5
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x




class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # 普通的卷积层，即获得了偏移量之后的特征图再接一个普通卷积
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # 获得偏移量，卷积核的通道数应该为2xkernel_sizexkernel_size
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        # 偏移量初始化为0
        nn.init.constant_(self.p_conv.weight, 0)
        # 注册module反向传播的hook函数, 可以查看当前层参数的梯度
        self.p_conv.register_backward_hook(self._set_lr)
        # 将modulation赋值给当前类
        self.modulation = modulation
        if modulation:
            # 如果是DCN V2，还多了一个权重参数，用m_conv来表示
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # 注册module反向传播的hook函数, 可以查看当前层参数的梯度
            self.m_conv.register_backward_hook(self._set_lr)

    # 静态方法 类或实例均可调用，这函数的结合hook可以输出你想要的Variable的梯度
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # 获得输入特征图x的偏移量
        # 假设输入特征图shape是[1,3,32,32]，然后卷积核是3x3，
        # 输出通道数为32，那么offset的shape是[1,2*3*3,32]
        offset = self.p_conv(x)
        # 如果是DCN V2那么还需要获得输入特征图x偏移量的权重项
        # 假设输入特征图shape是[1,3,32,32]，然后卷积核是3x3，
        # 输出通道数为32，那么offset的权重shape是[1,3*3,32]
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        # 卷积核尺寸大小
        ks = self.kernel_size
        # N=2*3*3/2=3*3=9
        N = offset.size(1) // 2
        # 对输入x进行padding
        if self.padding:
            x = self.zero_padding(x)
        # 将offset放到网格上，也就是标定出每一个坐标位置
        # (b, 2N, h, w)
        # 这个函数用来获取所有的卷积核偏移之后相对于原始特征图x的坐标（现在是浮点数）
        p = self._get_p(offset, dtype)

        # 我们学习出的量是float类型的，而像素坐标都是整数类型的，
        # 所以我们还要用双线性插值的方法去推算相应的值
        # 维度转换，现在p的维度为(b, h, w, 2N)
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整
        q_lt = p.detach().floor()
        # +1相当于向上取整，这里为什么不用向上取整函数呢？是因为如果正好是整数的话，向上取整跟向下取整就重合了，这是我们不想看到的。
        q_rb = q_lt + 1
        # 将lt限制在图像范围内，其中[..., :N]代表x坐标，[..., N:]代表y坐标,# 将q_lt即左上角坐标的值限制在图像范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 将rb限制在图像范围内,# 将q_rb即右下角坐标的值限制在图像范围内
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 获得lb,# 用q_lt的前半部分坐标q_lt_x和q_rb的后半部分q_rb_y组合成q_lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        # 插值的时候需要考虑一下padding对原始索引的影响
        # (b, h, w, N)
        # torch.lt() 逐元素比较input和other，即是否input < other
        # torch.rt() 逐元素比较input和other，即是否input > other
        #  # 对p的坐标也要限制在图像范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 插值的4个系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        # 现在只获取了坐标值，我们最终木的是获取相应坐标上的值，
        # 这里我们通过self._get_x_q()获取相应值。
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 插值的最终操作在这里
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # 偏置点含有九个方向的偏置，_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式，
        # 于是就可以用 3×3 stride=3 的卷积核进行 Deformable Convolution，
        # 它等价于使用 1×1 的正常卷积核（包含了这个点9个方向的 context）对原特征直接进行卷积。
        # 在获取所有值后我们计算出x_offset，但是x_offset的size
        # 是(b,c,h,w,N)，我们的目的是将最终的输出结果的size变
        # 成和x一致即(b,c,h,w)，所以在最后用了一个reshape的操作。
        # 这里ks=3
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype): # 用于生成卷积核位置的偏移量,用来生成卷积的相对坐标，其中卷积的中心点被看成原点，然后其它点的坐标都是相对于原点来说的，
        # 例如self.kernel_size=3，通过torch.meshgrid生成从（-1，-1）到（1，1）9个坐标。将坐标的x和y分别存储，然后再将x，y以(1,2N,1,1)的形式返回，这样我们就获取了一个卷积核的所有相对坐标。
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype): # 用于生成卷积核位置的基准值
        # 是获取卷积核在特征图上对应的中心坐标，也即论文公式中的p_0，通过torch.mershgrid生成所有的中心坐标，然后通过kernel_size推断初始坐标，然后通过stride推断所有的中心坐标，
        # 设w = 7, h = 5, stride = 1
        # 有p_0_x = tensor([[1, 1, 1, 1, 1, 1, 1],
        # [2, 2, 2, 2, 2, 2, 2],
        # [3, 3, 3, 3, 3, 3, 3],
        # [4, 4, 4, 4, 4, 4, 4],
        # [5, 5, 5, 5, 5, 5, 5]])
        # p_0_x.shape = [5, 7]
        # p_0_y = tensor([[1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7],
        # [1, 2, 3, 4, 5, 6, 7]])
        # p_0_y.shape = [5, 7]
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        # p_0_x的shape为torch.Size([1, 9, 5, 7])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_0_y的shape为torch.Size([1, 9, 5, 7])
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_0的shape为torch.Size([1, 18, 5, 7])
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype): # 根据偏移量计算出采样点的位置,函数用来获取所有的卷积核偏移之后相对于原始特征图x的坐标
        # N = 18 / 2 = 9
        # h = 32
        # w = 32
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # 卷积坐标加上之前学习出的offset后就是论文提出的公式(2)也就是加上了偏置后的卷积操作。
        # 比如p(在N=0时)p_0就是中心坐标，而p_n=(-1,-1)，所以此时的p就是卷积核中心坐标加上
        # (-1,-1)(即红色块左上方的块)再加上offset。同理可得N=1,N=2...分别代表了一个卷积核
        # 上各个元素。
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N): # 根据采样点的位置从输入特征中提取对应的特征值。
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod # 用于将提取的特征值重新组织为适合进行卷积操作的形状。
    def _reshape_x_offset(x_offset, ks):
        # 函数首先获取了x_offset的所有size信息，然后以kernel_size为
        # 单位进行reshape，因为N=kernel_size*kernel_size，所以我们
        # 分两次进行reshape，第一次先把输入view成(b,c,h,ks*w,ks)，
        # 第二次再view将size变成(b,c,h*ks,w*ks)
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
