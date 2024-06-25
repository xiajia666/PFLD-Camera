"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""

from typing import Optional, Tuple, Union, Dict
import math
import torch
from torch.nn import functional
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, AvgPool2d, Linear
from .m_transformer import TransformerEncoder
from .m_config import get_config
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU
def Conv_Block(in_channel, out_channel, kernel_size, stride, padding, group=1, has_bn=True, is_linear=False):
    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, groups=group, bias=False),
        BatchNorm2d(out_channel) if has_bn else Sequential(),
        ReLU(inplace=True) if not is_linear else Sequential()
    )

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,   # 每一个token的序列长度
        ffn_dim: int,   # MLP中第一个全连接层节点的个数
        n_transformer_blocks: int = 2,  #
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,  # patch的宽高
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        #local representation
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape
        # 可以被patch完整划分
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False) # 插值的方式
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict,input_size, landmark_number):
        super().__init__()

        image_channels = 3
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        # self.conv8 = Conv_Block(64, 128, input_size // 16, 1, 0, has_bn=False)
        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        # self.classifier = nn.Sequential()
        # self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        # self.classifier.add_module(name="flatten", module=nn.Flatten())
        # if 0.0 < model_cfg["cls_dropout"] < 1.0:
        #     self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        # self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # weight init
        self.apply(self.init_parameters)

        self.avg_pool1 = nn.AvgPool2d(56)
        self.avg_pool2 = nn.AvgPool2d(28)
        self.avg_pool3 = nn.AvgPool2d(14)
        self.avg_pool4 = nn.AvgPool2d(7)
        self.avg_pool5=  nn.AvgPool2d(4)
        self.fc = Linear(960, 196)
        self.conv7 = Conv_Block(80 ,96 , 3, 1, 1)
        self.conv8 = Conv_Block(96 , 128 , input_size // 16, 1, 0, has_bn=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_all = Conv_Block(480,192,1,1,0)
        self.dropout = nn.Dropout(0.2)
        self.landmarks = Linear(240, 196)
        # self.landmarks1 = Linear(512, 196)
        # self.pose = Linear(240, 3)
        # # self.pose1 = Linear(512, 3)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc_yaw = nn.Linear(864 , num_bins)
        # self.fc_pitch = nn.Linear(864 , num_bins)
        # self.fc_roll = nn.Linear(864 , num_bins)
        # # self.fc_yaw = nn.Linear(160 , num_bins)
        # self.fc_pitch = nn.Linear(160 , num_bins)
        # self.fc_roll = nn.Linear(160 , num_bins)
    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass



    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 3, 1, 2)
        # print('x',x.shape)

        x = self.conv_1(x)
        ou1 = self.layer_1(x) # ou1 : torch.Size([32, 32, 56, 56])
        # print('ou1 :', ou1.shape)
        x1 = self.avg_pool1(ou1)
        # print('x1:', x1.shape)
        x1 = x1.contiguous().view(x1.size(0), -1)
        # print('x1:', x1.shape)
        ou2 = self.layer_2(ou1) # ou2 : torch.Size([32, 64, 28, 28])
        # print('ou2 :', ou2.shape)
        x2 = self.avg_pool2(ou2)
        x2 = x2.contiguous().view(x2.size(0), -1)
        # print('x2:', x2.shape)
        ou3 = self.layer_3(ou2) # ou3 : torch.Size([32, 96, 14, 14])
        # print('ou3 :', ou3.shape)
        # print('x3:', x.shape)
        x3 = self.avg_pool3(ou3)
        x3 = x3.contiguous().view(x3.size(0), -1)
        # print('x3:', x3.shape)
        ou4 = self.layer_4(ou3) # ou4 : torch.Size([32, 128, 7, 7])
        # print('ou4 :', ou4.shape)
        # print('x4:', x.shape)
        x4 = self.avg_pool4(ou4)
        x4 = x4.contiguous().view(x4.size(0), -1)
        # print('x4:', x4.shape)
        ou5 = self.layer_5(ou4) # [1, 64, 4, 4]
        # print('ou5 :', ou5.shape)
        ou5 = self.conv_1x1_exp(ou5)
        # print('ou5 :', ou5.shape)
        # print(x.shape)
        # x = self.conv7(x)
        # print(x.shape)
        # x5 = self.conv8(x)
        # print('x5:', x.shape)
        x5 = self.avg_pool5(ou5)
        x5 = x5.contiguous().view(x5.size(0), -1)  # x5: torch.Size([32, 16, 56, 56])
        # print('x5:', x.shape)
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)

        # pre_yaw = self.fc_yaw(multi_scale)
        # pre_pitch = self.fc_pitch(multi_scale)
        # pre_roll = self.fc_roll(multi_scale)

        # pre_yaw = self.fc_yaw(x5)
        # pre_pitch = self.fc_pitch(x5)
        # pre_roll = self.fc_roll(x5)


        # print('mu:',multi_scale.shape)
        h = self.fc(multi_scale)

        # landmarks = self.dropout(self.relu(self.landmarks(h)))
        # # landmarks = self.landmarks1(landmarks)
        # pose = self.dropout(self.relu(self.pose(h)))
        # # pose = self.pose1(pose)
        # return pre_yaw, pre_pitch, pre_roll,h
        return h, ou2


def mobile_vit_xx_small(width_factor=1,input_size=112, landmark_number=98):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, input_size=112, landmark_number=98)
    return m


def mobile_vit_x_small(width_factor=1,input_size=112, landmark_number=98):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, input_size=112 , landmark_number=98)
    return m


def mobile_vit_small(width_factor=1,input_size=112, landmark_number=98):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small1")
    m = MobileViT(config,input_size,landmark_number)
    return m


class PFLD_mobileVIT_AuxiliaryNet(Module):
    def __init__(self, width_factor=1):
        super(PFLD_mobileVIT_AuxiliaryNet, self).__init__()
        self.conv1 = Conv_Block(int(64 * width_factor), int(128 * width_factor), 3, 2, 0)
        self.conv2 = Conv_Block(int(128 * width_factor), int(128 * width_factor), 3, 1, 0)
        self.conv3 = Conv_Block(int(128 * width_factor), int(32 * width_factor), 3, 2, 0)
        self.conv4 = Conv_Block(int(32 * width_factor), int(128 * width_factor), 3, 1, 0)

        self.avg1 = nn.AvgPool2d(3)
        self.avg2 = nn.AvgPool2d(3)
        self.avg3 = nn.AvgPool2d(3)

        # self.fc1 = nn.Linear(512, num_bins)
        # self.fc2 = nn.Linear(512, num_bins)
        # self.fc3 = nn.Linear(512, num_bins)

        # self.softmax1 = nn.Softmax(dim=1)
        # self.softmax2 = nn.Softmax(dim=1)
        # self.softmax3 = nn.Softmax(dim=1)
        self.fc11 = nn.Linear(128, 32)
        self.fc22 = nn.Linear(32, 3)
        self.max_pool1 = nn.MaxPool2d(3)
    def forward(self, x):

        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # # print('x1:',x.shape)
        # x = self.conv4(x)
        # # print('x2:',x.shape)
        # out1 = self.avg1(x)
        # # print('out1:',out1.shape)
        # out1 = out1.view(x.size(0), -1)
        # # print('out1:', out1.shape)
        # out1 = self.fc1(out1)
        # # print('out1:', out1.shape)
        #
        # out2 = self.avg2(x)
        # # print('out2:', out2.shape)
        # out2 = out2.view(x.size(0), -1)
        # # print('out3:', out2.shape)
        # out2 = self.fc2(out2)
        # # print('out3:', out2.shape)
        #
        #
        # out3 = self.avg3(x)
        # # print('out3:', out3.shape)
        # out3 = out3.view(x.size(0), -1)
        # # print('out3:', out3.shape)
        # out3 = self.fc3(out3)
        # # print('out3:', out3.shape)
        # # print(out1)
        print('x1:', x.shape)
        x = self.conv1(x)
        print('x2:',x.shape)
        x = self.conv2(x)
        print('x3:', x.shape)
        x = self.conv3(x)
        print('x4:', x.shape)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        print('x5:', x.shape)
        x = self.fc11(x)
        print('x6:', x.shape)
        x = self.fc22(x)
        return x

class AuxiliaryNet(nn.Module):

    def __init__(self, input_channels, nums_class=3, activation=nn.ReLU, first_conv_stride=2):
        super(AuxiliaryNet, self).__init__()
        self.input_channels = input_channels
        # self.num_channels = [128, 128, 32, 128, 32]
        self.num_channels = [512, 512, 512, 512, 1024]
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels[0], kernel_size=3, stride=first_conv_stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels[0])

        self.conv2 = nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels[1])

        self.conv3 = nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels[2])

        self.conv4 = nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm2d(self.num_channels[3])

        self.fc1 = nn.Linear(in_features=self.num_channels[3], out_features=self.num_channels[4])
        self.fc2 = nn.Linear(in_features=self.num_channels[4], out_features=nums_class)

        self.activation = activation(inplace=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # print('input:',input.shape)
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)
        # print('out:', out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        # print('out:', out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)
        # print('out:', out.shape)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.activation(out)
        # print('out:', out.shape)
        out = functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        #print(out.size())
        # out = out.view(out.size(0), -1)
        # print('out:', out.shape)
        out = self.fc1(out)
        # print('out:', out.shape)
        euler_angles_pre = self.fc2(out)

        return euler_angles_pre
