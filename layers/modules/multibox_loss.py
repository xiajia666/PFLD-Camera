import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
from ..box import match, log_sum_exp
from ..box import match_ious, bbox_overlaps_iou, bbox_overlaps_giou, bbox_overlaps_diou, bbox_overlaps_ciou, decode, match
import math
GPU = cfg_mnet['gpu_train']

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # 输入参数inputs是网络的输出，形状为(N, C)，其中N表示batch size，C表示类别数量。targets是真实的目标标签，形状为(N,)。
        print('input:',inputs)
        print('input',inputs.shape)
        print('target:',targets)
        print('target:', targets.shape)
        N = inputs.size(0)
        print('N:', N)
        C = inputs.size(1)
        print('C:', C)
        P = torch.sigmoid(inputs) # 根据inputs使用softmax函数计算预测概率矩阵P，形状为(N, C) torch.Size([7824, 2])
        print('P:', P)
        class_mask = inputs.data.new(N, 2).fill_(0)  # 根据targets构建类别掩码class_mask，将对应类别的位置设为1，其余位置为0。
        print('class_mask:',class_mask)
        print('class_mask:', class_mask.shape)
        class_mask = Variable(class_mask)  # 将类别掩码张量转换为Variable对象
        ids = targets.view(-1, 1)  # 目标类别标签 targets 变形为 (N, 1) 的张量 ids，其中 N 是批次大小。\
        print('ids:',ids)
        print('ids:', ids.shape)
        class_mask.scatter_(1, ids.data, 1.) # torch.Size([7824, 2])  是将类别掩码中对应样本的类别位置设为1。
        if inputs.is_cuda and not self.alpha.is_cuda:
            print('self.alpha:',self.alpha)
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        #通过矩阵相乘，计算每个样本属于对应类别的概率P乘以类别掩码class_mask的和，得到每个样本的预测概率probs，形状为(N, 1)。
        probs = (P * class_mask).sum(1).view(-1, 1) # 只会保留对应类别的预测概率，其他位置的概率将被置零。
        log_p = probs.log()
        # print('alpha:',alpha)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class IouLoss(nn.Module):

    def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Ciou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss

# class MultiBoxLoss(nn.Module):
#     """SSD Weighted Loss Function
#     Compute Targets:
#         1) Produce Confidence Target Indices by matching  ground truth boxes
#            with (default) 'priorboxes' that have jaccard index > threshold parameter
#            (default threshold: 0.5).
#         2) Produce localization target by 'encoding' variance into offsets of ground
#            truth boxes and their matched  'priorboxes'.
#         3) Hard negative mining to filter the excessive number of negative examples
#            that comes with using a large number of default bounding boxes.
#            (default negative:positive ratio 3:1)
#     Objective Loss:
#         L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
#         weighted by α which is set to 1 by cross val.
#         Args:
#             c: class confidences,
#             l: predicted boxes,
#             g: ground truth boxes
#             N: number of matched default boxes
#         See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#     """
#
#     def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,loss_name ='SmoothL1'):
#         super(MultiBoxLoss, self).__init__()
#         self.num_classes = num_classes
#         self.threshold = overlap_thresh
#         self.background_label = bkg_label
#         self.encode_target = encode_target
#         self.use_prior_for_matching = prior_for_matching
#         self.do_neg_mining = neg_mining
#         self.negpos_ratio = neg_pos
#         self.neg_overlap = neg_overlap
#         self.variance = [0.1, 0.2]
#         self.focalloss = FocalLoss(self.num_classes, gamma=2, size_average=False)
#         self.loss = loss_name
#         self.gious = IouLoss(pred_mode='Center', size_sum=True, variances=self.variance, losstype=self.loss)
#         if self.loss != 'SmoothL1' or self.loss != 'Ciou':
#             assert Exception("THe loss is Error, loss name must be SmoothL1 or Giou")
#
#         else:
#             match_ious(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#
#     def forward(self, predictions, priors, targets):
#         """Multibox Loss
#         Args:
#             predictions (tuple): A tuple containing loc preds, conf preds,
#             and prior boxes from SSD net.
#                 conf shape: torch.size(batch_size,num_priors,num_classes)
#                 loc shape: torch.size(batch_size,num_priors,4)
#                 priors shape: torch.size(num_priors,4)
#
#             ground_truth (tensor): Ground truth boxes and labels for a batch,
#                 shape: [batch_size,num_objs,5] (last idx is the label).
#         """
#
#         loc_data, conf_data, landm_data = predictions
#         priors = priors
#         num = loc_data.size(0)
#         num_priors = (priors.size(0))
#         # print(loc_data.shape, conf_data.shape, landm_data.shape, num, num_priors)
#         # match priors (default boxes) and ground truth boxes
#         loc_t = torch.Tensor(num, num_priors, 4)
#         landm_t = torch.Tensor(num, num_priors, 10)
#         conf_t = torch.LongTensor(num, num_priors)
#         for idx in range(num):
#             truths = targets[idx][:, :4].data
#             labels = targets[idx][:, -1].data
#             landms = targets[idx][:, 4:14].data
#             defaults = priors.data
#             if self.loss == 'SmoothL1':
#                 match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#             else:
#                 match_ious(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#         if GPU:
#             loc_t = loc_t.cuda()
#             conf_t = conf_t.cuda()
#             landm_t = landm_t.cuda()
#
#         zeros = torch.tensor(0).cuda()
#         # landm Loss (Smooth L1)
#         # Shape: [batch,num_priors,10]
#         pos1 = conf_t > zeros
#         num_pos_landm = pos1.long().sum(1, keepdim=True)
#         N1 = max(num_pos_landm.data.sum().float(), 1)
#         pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
#         landm_p = landm_data[pos_idx1].view(-1, 10)
#         landm_t = landm_t[pos_idx1].view(-1, 10)
#         loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
#
#
#         pos = conf_t != zeros
#         conf_t[pos] = 1
#
#         # Localization Loss (Smooth L1)
#         # Shape: [batch,num_priors,4]
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_p = loc_data[pos_idx].view(-1, 4)
#         loc_t = loc_t[pos_idx].view(-1, 4)
#         # loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#
#         if self.loss == 'SmoothL1':
#             loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#         else:
#             giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
#             loss_l = self.gious(loc_p,loc_t,giou_priors[pos_idx].view(-1, 4))
#         # Compute max conf across batch for hard negative mining
#         batch_conf = conf_data.view(-1, self.num_classes)
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
#
#         # Hard Negative Mining
#         loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
#         loss_c = loss_c.view(num, -1)
#         _, loss_idx = loss_c.sort(1, descending=True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim=True)
#         num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#         neg = idx_rank < num_neg.expand_as(idx_rank)
#
#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#         neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#         conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
#         targets_weighted = conf_t[(pos+neg).gt(0)]
#         loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
#
#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         N = max(num_pos.data.sum().float(), 1)
#         loss_l /= N
#         loss_c /= N
#         loss_landm /= N1
#
#         return loss_l, loss_c, loss_landm
# class MultiBoxLoss(nn.Module):
#     """SSD Weighted Loss Function
#     Compute Targets:
#         1) Produce Confidence Target Indices by matching  ground truth boxes
#            with (default) 'priorboxes' that have jaccard index > threshold parameter
#            (default threshold: 0.5).
#         2) Produce localization target by 'encoding' variance into offsets of ground
#            truth boxes and their matched  'priorboxes'.
#         3) Hard negative mining to filter the excessive number of negative examples
#            that comes with using a large number of default bounding boxes.
#            (default negative:positive ratio 3:1)
#     Objective Loss:
#         L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
#         weighted by α which is set to 1 by cross val.
#         Args:
#             c: class confidences,
#             l: predicted boxes,
#             g: ground truth boxes
#             N: number of matched default boxes
#         See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#     """
#
#     def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,loss_name = 'Ciou'):
#         super(MultiBoxLoss, self).__init__()
#         self.num_classes = num_classes
#         self.threshold = overlap_thresh
#         self.background_label = bkg_label
#         self.encode_target = encode_target
#         self.use_prior_for_matching = prior_for_matching
#         self.do_neg_mining = neg_mining
#         self.negpos_ratio = neg_pos
#         self.neg_overlap = neg_overlap
#         self.variance = [0.1, 0.2]
#         self.focalloss = FocalLoss(self.num_classes, gamma=2, size_average=False)
#         self.wingloss = WingLoss()
#         self.loss = loss_name
#         self.cious = IouLoss(pred_mode='Center', size_sum=True, variances=self.variance, losstype=self.loss)
#         if self.loss != 'SmoothL1' or self.loss != 'Ciou':
#             assert Exception("THe loss is Error, loss name must be SmoothL1 or Ciou")
#
#         else:
#             match_ious(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#
#     def forward(self, predictions, priors, targets):
#         """Multibox Loss
#         Args:
#             predictions (tuple): A tuple containing loc preds, conf preds,
#             and prior boxes from SSD net.
#                 conf shape: torch.size(batch_size,num_priors,num_classes)
#                 loc shape: torch.size(batch_size,num_priors,4)
#                 priors shape: torch.size(num_priors,4)
#
#             ground_truth (tensor): Ground truth boxes and labels for a batch,
#                 shape: [batch_size,num_objs,5] (last idx is the label).
#         """
#         # 坐标，前背景，landmarks
#         # [1,16800,4],[1,16800,2],[1,16800,10]
#         # print('predictions:', predictions.shape)
#         loc_data, conf_data, landm_data = predictions
#         # [16800,4]
#         priors = priors
#         num = loc_data.size(0)  # batchsize
#         num_priors = (priors.size(0))  # num_anchors
#         # print('num_priors :', num_priors.shape)
#         # match priors (default boxes) and ground truth boxes
#         # 这几是预备存储内容的
#         loc_t = torch.Tensor(num, num_priors, 4)
#         landm_t = torch.Tensor(num, num_priors, 10)
#         conf_t = torch.LongTensor(num, num_priors)
#         for idx in range(num):
#             # 对batch中的每一个内容
#             truths = targets[idx][:, :4].data  # 坐标
#             labels = targets[idx][:, -1].data  # 置信度
#             landms = targets[idx][:, 4:14].data  # landmarks
#             defaults = priors.data
#             # 最后结果都在loc_t,conf_t,landm_t里面
#             # 一句话，找到每个anchors该负责的gt框，
#             # 并将该gt框转化为对应与该anchors坐标的归一化参数
#             # 通过overlap阈值过滤的方式，找到了那些被选中的anchors，并在他们的位置上标志为非0.
#             if self.loss == 'SmoothL1':
#                 match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#             else:
#                 match_ious(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#         if GPU:
#             loc_t = loc_t.cuda()
#             conf_t = conf_t.cuda()
#             landm_t = landm_t.cuda()
#         # print('loc_t:',loc_t)
#         # print('conf_t:',conf_t)
#         # print('landm_t:',landm_t)
#         zeros = torch.tensor(0).cuda() # 创建了一个值为0的张量zeros，并将其放置在GPU上。
#         # landm Loss (Smooth L1)
#         # Shape: [batch,num_priors,10]
#         # 目标检测任务中的分类标签conf_t和zeros进行比较，生成一个布尔类型的掩码pos1。对于conf_t中大于zeros的元素，对应位置的值为True，否则为False。
#         # 这个掩码用于筛选出正样本（类别标签大于0的样本）。
#         pos1 = conf_t > zeros
#         # 拿到anchors中，和gt框的overlap大于阈值的anchors数目
#         num_pos_landm = pos1.long().sum(1, keepdim=True)  # 统计每个样本中正样本的数量，并将结果保留为一个1维张量num_pos_landm
#         # 拿到所有batch中，最大的num_pos_landm
#         N1 = max(num_pos_landm.data.sum().float(), 1)  # 计算正样本的总数N1，即num_pos_landm中所有元素之和,和为0，则将N1设置为1，以避免除以0的错误。
#         # 把1*16800变成1*16800*10
#         # 单纯的复制了每一位置10遍
#         pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)  # 将正样本的掩码扩展到与landm_data的形状相同。
#         # 根据pos_idx1从landm_data和landm_t中选择正样本的预测值和目标值，并将它们展平成形状为(-1, 10)的张量landm_p和landm_t。
#         # 过滤，并拿到预测结果中，对应的landmarks
#         landm_p = landm_data[pos_idx1].view(-1, 10)
#         # 过滤，并拿到labels中对应的数据
#         landm_t = landm_t[pos_idx1].view(-1, 10)
#         # print('landm_p:',landm_p)
#         # print('landm_t:',landm_t)
#         # 根据预测结果和target结果，计算landmarks的loss
#         # loss_landm = self.wingloss(landm_p, landm_t)
#         loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
#
#         # 根据目标检测任务中的分类标签conf_t和zeros进行比较，生成一个布尔类型的掩码pos。对于conf_t中不等于zeros的元素，对应位置的值为True，否则为False。
#         # 这个掩码用于筛选出正样本（类别标签不等于0的样本）。在人脸检测中，通常将真实标签中置信度不为零的位置视为正样本。
#
#         pos = conf_t != zeros
#         conf_t[pos] = 1
#         num_pos = pos.sum(dim=1, keepdim=True)
#         # conf_t[pos] = 1 # 将conf_t中对应正样本位置的值设置为1，以标记这些正样本。
#         # Localization Loss (Smooth L1)
#         # Shape: [batch,num_priors,4]
#         # 根据pos生成与loc_data相同形状的掩码pos_idx，将正样本的掩码扩展到与loc_data的形状相同。
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_p = loc_data[pos_idx].view(-1, 4)
#         loc_t = loc_t[pos_idx].view(-1, 4)
#         # loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#
#         if self.loss == 'SmoothL1':
#             loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#         else:
#             giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
#             loss_l = self.cious(loc_p,loc_t,giou_priors[pos_idx].view(-1, 4))
#
#         # # class loss(FocalLoss)
#         # # print('conf_data:',conf_data.tolist())
#         # batch_conf = conf_data.view(-1, self.num_classes)
#         # # print('batch_conf:',batch_conf.tolist())
#         # # print('conf_t:',conf_t.tolist())
#         # loss_c = self.focalloss(batch_conf, conf_t)
#         #
#         # num_pos = pos.long().sum(1, keepdim=True)  # 正样本的数量
#
#         '''
#                 1、根据阈值过滤，拿到正样本的位置pos和数量num_pos
#                 2、num_pos*7拿到负样本数量num_neg。
#                 3、根据loss_c，拿到loss最大的num_neg个anchors的位置neg。
#                 4、从conf_t和conf_d中，拿出pos和neg对应的anchors，计算cla_loss
#         '''
#         # batch_conf = conf_data.view(-1, self.num_classes)
#         # loss_c = self.focalloss(batch_conf, conf_t)
#         # # Compute max conf across batch for hard negative mining
#         batch_conf = conf_data.view(-1, self.num_classes) # 网络输出的置信度（confidence）预测结果，通过 view 函数将其转换为大小为 (batch_size, num_classes) 的张量。
#         '''
#                     这个是用来排序用的
#                     可以理解为，每个anchors会预测两个概率，【p1,p2】，
#                     并且，根据gt，我们知道它应该得到的是[0,1]
#                     那么，这个anchors的loss=log(e^p1+e^p2)-p2
#                     loss_c就存了这样的16800个这样的loss
#          '''
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
#         # Hard Negative Mining
#         # 总不能用所有的16800个loss来计算
#
#         # 被选中的anchors的loss写为0,将 loss_c 中对应于正样本位置为 True 的分类损失设置为零。
#         # 这样做的目的是将真实标签中的正样本的分类损失排除在计算范围之外，因为正样本已经被确定为正确分类，不需要再计算分类损失。
#         loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
#         loss_c = loss_c.view(num, -1)  # num 是批次中样本的数量
#         '''
#                 a = torch.Tensor([[10,50,70,20,30,40]])
#                 _,b = a.sort(1,descending=True) # b=tensor([[2, 1, 5, 4, 3, 0]])
#                 _,c = b.sort(1) # c=tensor([[5, 1, 0, 4, 3, 2]])
#         '''
#         # 每个batch单独,按照value大小，降序排序
#         # 对每张图的priorbox的conf loss从大到小排序，每一列的值为prior box的index；相当于最不是前景的排在第一个
#         _, loss_idx = loss_c.sort(1, descending=True)  # 降序对 loss_c 进行排序，得到排序后的索引 loss_idx 和排序的排名 idx_rank
#         _, idx_rank = loss_idx.sort(1)  # 对上面每一列，按照存储内容大小进行排序，对 loss_idx 张量进行排序，idx_rank。这个索引张量表示了每个元素在初始 loss_c 张量中的排名。
#         num_pos = pos.long().sum(1, keepdim=True)  # 正样本的数量
#         num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)  # 将正样本数量乘以负正样本比例 self.negpos_ratio 并进行上限限制，得到负样本的数量。
#         '''
#                     neg中拿到的是idx_rank中从0~num_neg的这些下标
#                     这些下标对应的是loss_idx中loss比较的位置
#         '''
#         neg = idx_rank < num_neg.expand_as(idx_rank)  # 利用排名 idx_rank 判断哪些样本为负样本，将小于 num_neg 的索引置为True，得到负样本的布尔张量 neg。
#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # 将正样本的索引 pos 进行扩展，形状与 conf_data 一致，并且通过索引可以选择相应位置的元素。得到 pos_idx
#         neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#         # 根据正负样本的索引，从 conf_data 中提取对应的置信度预测结果 conf_p,形状为 (num_pos + num_neg, num_classes)
#         # 拿到prediction中的数据，得到的结果是一个布尔张量，其中 True 表示正样本或负样本的位置，False 表示其他位置。
#         conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes) # torch.Size([7824, 2]),每个正样本的数量都不一样
#         # 同时从真实标签 conf_t 中提取对应的权重，并形状与 conf_p 一致，得到 targets_weighted。
#         # 拿到labels中的数据
#         targets_weighted = conf_t[(pos+neg).gt(0)] # 得到正样本和负样本对应位置的真实标签，torch.Size([7824])
#         # loss_c = self.focalloss(conf_p, targets_weighted)
#         loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
#
#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         N = max(num_pos.data.sum().float(), 1) # num_pos正样本的数量，确保正样本的数量至少为1，以避免除以0的情况。
#
#         # 将定位损失值loss_l和分类损失值loss_c都除以正样本的数量N，以获得它们的平均损失。
#         loss_l /= N
#         loss_c /= N
#         loss_landm /= N1
#         return loss_l, loss_c, loss_landm


# class IouLoss(nn.Module):
#
#     def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Diou'):
#         super(IouLoss, self).__init__()
#         self.size_sum = size_sum
#         self.pred_mode = pred_mode
#         self.variances = variances
#         self.loss = losstype
#
#     def forward(self, loc_p, loc_t, prior_data):
#         num = loc_p.shape[0]
#
#         if self.pred_mode == 'Center':
#             decoded_boxes = decode(loc_p, prior_data, self.variances)
#         else:
#             decoded_boxes = loc_p
#         if self.loss == 'Iou':
#             loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
#         else:
#             if self.loss == 'Giou':
#                 loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
#             else:
#                 if self.loss == 'Diou':
#                     loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
#                 else:
#                     loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))
#
#         if self.size_sum:
#             loss = loss
#         else:
#             loss = loss / num
#         return loss



# class MultiBoxLoss(nn.Module):
#     """SSD Weighted Loss Function
#     Compute Targets:
#         1) Produce Confidence Target Indices by matching  ground truth boxes
#            with (default) 'priorboxes' that have jaccard index > threshold parameter
#            (default threshold: 0.5).
#         2) Produce localization target by 'encoding' variance into offsets of ground
#            truth boxes and their matched  'priorboxes'.
#         3) Hard negative mining to filter the excessive number of negative examples
#            that comes with using a large number of default bounding boxes.
#            (default negative:positive ratio 3:1)
#     Objective Loss:
#         L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
#         weighted by α which is set to 1 by cross val.
#         Args:
#             c: class confidences,
#             l: predicted boxes,
#             g: ground truth boxes
#             N: number of matched default boxes
#         See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#     """
#
#     def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
#         super(MultiBoxLoss, self).__init__()
#         self.num_classes = num_classes
#         self.threshold = overlap_thresh
#         self.background_label = bkg_label
#         self.encode_target = encode_target
#         self.use_prior_for_matching = prior_for_matching
#         self.do_neg_mining = neg_mining
#         self.negpos_ratio = neg_pos
#         self.neg_overlap = neg_overlap
#         self.variance = [0.1, 0.2]
#
#     def forward(self, predictions, priors, targets):
#         """Multibox Loss
#         Args:
#             predictions (tuple): A tuple containing loc preds, conf preds,
#             and prior boxes from SSD net.
#                 conf shape: torch.size(batch_size,num_priors,num_classes)
#                 loc shape: torch.size(batch_size,num_priors,4)
#                 priors shape: torch.size(num_priors,4)
#
#             ground_truth (tensor): Ground truth boxes and labels for a batch,
#                 shape: [batch_size,num_objs,5] (last idx is the label).
#         """
#
#         loc_data, conf_data, landm_data = predictions
#         priors = priors
#         num = loc_data.size(0)
#         num_priors = (priors.size(0))
#
#         # match priors (default boxes) and ground truth boxes
#         loc_t = torch.Tensor(num, num_priors, 4)
#         landm_t = torch.Tensor(num, num_priors, 10)
#         conf_t = torch.LongTensor(num, num_priors)
#         for idx in range(num):
#             truths = targets[idx][:, :4].data
#             labels = targets[idx][:, -1].data
#             landms = targets[idx][:, 4:14].data
#             defaults = priors.data
#             match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
#         if GPU:
#             loc_t = loc_t.cuda()
#             conf_t = conf_t.cuda()
#             landm_t = landm_t.cuda()
#
#         zeros = torch.tensor(0).cuda()
#         # landm Loss (Smooth L1)
#         # Shape: [batch,num_priors,10]
#         pos1 = conf_t > zeros
#         num_pos_landm = pos1.long().sum(1, keepdim=True)
#         N1 = max(num_pos_landm.data.sum().float(), 1)
#         pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
#         landm_p = landm_data[pos_idx1].view(-1, 10)
#         landm_t = landm_t[pos_idx1].view(-1, 10)
#         loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
#
#
#         pos = conf_t != zeros
#         conf_t[pos] = 1
#
#         # Localization Loss (Smooth L1)
#         # Shape: [batch,num_priors,4]
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_p = loc_data[pos_idx].view(-1, 4)
#         loc_t = loc_t[pos_idx].view(-1, 4)
#         loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
#
#         # Compute max conf across batch for hard negative mining
#         batch_conf = conf_data.view(-1, self.num_classes)
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
#
#         # Hard Negative Mining
#         loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
#         loss_c = loss_c.view(num, -1)
#         _, loss_idx = loss_c.sort(1, descending=True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim=True)
#         num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#         neg = idx_rank < num_neg.expand_as(idx_rank)
#
#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#         neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#         conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
#         targets_weighted = conf_t[(pos+neg).gt(0)]
#         loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
#
#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         N = max(num_pos.data.sum().float(), 1)
#         loss_l /= N
#         loss_c /= N
#         loss_landm /= N1
#
#         return loss_l, loss_c, loss_landm
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm