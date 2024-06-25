import torchvision.models._utils as _utils
from models.ghostnet import *
from models.mobilenetv3 import *
from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
from models.convnext import *

class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        # 将最后的预测结果即通道调整到最后一维
        out = out.permute(0, 2, 3, 1).contiguous()
        # 第一维度batchsize, 第二维度所有的先验框，第三维度为每一个先验框是否包含人脸即包含人脸的概率
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        # 最后一维为每一个先验框的调整参数
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        # 最后一维为所有先验框5个人脸关键点的调整参数
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                print('new_state_dict:', new_state_dict)
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        elif cfg['name'] == 'ghostnet':
            backbone = ghostnet()
        elif cfg['name'] == 'mobilev3':
            backbone = MobileNetV3()
        elif cfg['name'] == 'convnexttiny':
            backbone = convnext_tiny(num_classes=2)
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/convnext_tiny_1k_224_ema.pth", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                # 删除有关分类类别的权重
                for k in list(checkpoint.keys()):
                    if "head" in k:
                        del checkpoint[k]
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v

                # load params
                backbone.load_state_dict(new_state_dict)
        # 获取主干特征提取网络的三个有效特征层，C3,C4,C5
        # print(cfg['return_layers'])
        # print('backbone:',backbone)
        # print(_utils.IntermediateLayerGetter(backbone,  cfg['return_layers']))
        self.body = _utils.IntermediateLayerGetter(backbone,  cfg['return_layers'])

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        # 利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 2，num_anchors每一个网格点先验框的数量默认为2，用于代表每个先验框内部包含人脸的概率。
        # 若num_anchors x 2中2里面序号为1的值较大，意味着先验框内部包含人脸，若序号为0的值大。则不包含人脸
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        # 利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 4，用于代表每个先验框的调整参数。4个参数中，前两个对先验框的中心进行调整，获得预测框的中心
        # 后两个对先验框的宽高进行调整。获得最终的预测框
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        # 利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 10（num_anchors x 5 x 2），用于代表每个先验框的每个人脸关键点的调整。
        # 2 为5个人脸关键点的调整参数，调整参数对先验框的中心进行调整获得5个人脸关键点的位置
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        # print('input的长度',len(inputs))
        # print('input:',inputs)
        # 做一个前向传递
        out = self.body(inputs)

        # FPN
        # 如果是调用mb0.25或者rn，则FPN需要改成fpn（小写），具体原因排查中，前向传递的结果传入fpn中
        fpn = self.fpn(out)

        # SSH
        # 构建3个SSH,分别对特征融合得到的三个有效特征层进行处理，获得最终的三个有效特征层
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        # 对每一个特征层都进行三个head的构建
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
