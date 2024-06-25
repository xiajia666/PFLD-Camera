# -*- coding: utf-8 -*-
import torch
import math


def bbox_overlaps_diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return cious


def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
            bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1 + area2 - inter_area
    ious = inter_area / union
    ious = torch.clamp(ious, min=0, max=1.0)
    if exchange:
        ious = ious.T
    return ious


def bbox_overlaps_giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
            bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious, min=-1.0, max=1.0)
    if exchange:
        ious = ious.T
    return ious


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    将先验框从中心-大小表示转换为(xmin, ymin, xmax, ymax)表示，以便与点形式的真实标注框数据进行比较
    """
    # print(boxes)
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B]. 第一列乘以第二列 交集区域在 x 方向上的边长，交集区域在 y方向上的边长
    """
    # print(box_a)
    # print(box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    #  每个真实的框与所有的先验框进行比较
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0) # 边长为负值（即没有重叠），则会被限制为0，而正值则保持不变。两个框框之间的交集区域的边长
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]  每个标注框的四个坐标值。
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]  表示每个先验框的四个坐标值。
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)  # [A,B] B的值要不是正值，要不为0
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match_ious(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    loc_t[idx] = point_form(priors)
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    landm = landms[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc_t[idx] = matches  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


# def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
#     """Match each prior box with the ground truth box of the highest jaccard
#     overlap, encode the bounding boxes, then return the matched indices
#     corresponding to both confidence and location preds.
#     Args:
#         threshold: (float) The overlap threshold used when mathing boxes.
#         用于匹配边界框时的重叠阈值，表示两个边界框之间的Jaccard重叠必须大于该阈值才能进行匹配。
#         truths: (tensor) Ground truth boxes, Shape: [num_obj, 4]. 表示图像中的目标数量，num_priors表示先验框的数量。
#         priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4]. 先验框的数量
#         variances: (tensor) Variances corresponding to each prior coord,
#             Shape: [num_priors, 4]. 对应于每个先验框坐标的方差张量
#         labels: (tensor) All the class labels for the image, Shape: [num_obj]. 每个目标对应一个类别标签。
#         loc_t: (tensor) Tensor to be filled w/ endcoded location targets.用于存储编码后的位置目标的张量，将被填充。表示每个先验框的位置偏移量
#         conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.用于存储匹配索引的置信度目标的张量，将被填充。表示每个先验框的置信度标签
#         idx: (int) current batch index 当前批次的索引，用于区分不同批次的数据
#     Return:
#         将真实标注框与先验框进行匹配，并生成用于训练目标检测模型的位置和置信度目标。同时，它还处理了一些特殊情况，如过滤掉与背景类别重叠较小的匹配。
#         The matched indices corresponding to 1)location and 2)confidence preds.
#     """
#     # jaccard index
#     # print('truth:',truths)
#     # print('truth:',truths.shape)
#     # print('priors:',priors)
#     # print('priors:', priors.shape)
#
#     overlaps = jaccard(
#         truths,
#         point_form(priors)
#     )
#     # print('overlaps:',overlaps)
#     # print('overlaps:',overlaps.shape)
#     # 得到每个真实框与所有先验框的交并比，0或者正值 [num_obj,num_priors]
#     # (Bipartite Matching)
#     # [1,num_objects] best prior for each ground truth
#     # 对overlaps进行按行操作，返回每行中的交并比的最大值和对应的索引。[num_obj, 1]，[num_obj, 1]
#     best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
#     # print('best_prior_overlap:',best_prior_overlap)
#     # print('best_prior_overlap:',best_prior_overlap.shape)
#     # print('best_prior_idx:',best_prior_idx)
#     # print('best_prior_idx:', best_prior_idx.shape)
#     valid_gt_idx = best_prior_overlap[:,0] >= 0.2  # 是一个布尔类型的张量[num_obj, 1]
#     # print('valid_gt_idx:',valid_gt_idx)
#     # print('valid_gt_idx:', valid_gt_idx.shape)
#     best_prior_idx_filter = best_prior_idx[valid_gt_idx,:]  # [num_valid_gt, 1]，满足条件的ground truth box对应的prior box的索引
#     # print('best_prior_idx_filter:',best_prior_idx_filter)
#     # print('best_prior_idx_filter:', best_prior_idx_filter.shape)
#     if best_prior_idx_filter.shape[0] <= 0:
#         loc_t[idx] = 0
#         conf_t[idx] = 0
#         return
#     # [1,num_priors] best ground truth for each prior
#     # 对overlaps进行按行操作，返回每列中的交并比的最大值和对应的索引。[num_obj, 1]，[num_obj, 1]
#     # 码找到每个prior box对应的最佳（具有最大jaccard overlap）ground truth box的索引和相应的overlap值。
#     best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)   # torch.Size([1, 19116])
#
#     # print('best_truth_overlap:',best_truth_overlap)
#     # print('best_truth_overlap:',best_truth_overlap.shape)
#     # print('best_truth_idx:', best_truth_idx)
#     # print('best_truth_idx:', best_truth_idx.shape)
#     best_truth_idx.squeeze_(0) # 从1维压缩为0维，即去除多余的维度。  torch.Size([num_priors])
#     # print('best_truth_idx2:', best_truth_idx)
#     # print('best_truth_idx2:', best_truth_idx.shape)
#     best_truth_overlap.squeeze_(0)  # torch.Size([num_priors])
#     # print('best_truth_overlap:',best_truth_overlap)
#     # print('best_truth_overlap:',best_truth_overlap.shape)
#     best_prior_idx.squeeze_(1) # 从2维压缩为1维，即去除多余的维度。 num_obj
#     # print('best_prior_idx:',best_prior_idx)
#     # print('best_prior_idx:', best_prior_idx.shape)
#     best_prior_idx_filter.squeeze_(1)
#     # print('best_prior_idx_filter:',best_prior_idx_filter)
#     # print('best_prior_idx_filter:', best_prior_idx_filter.shape)
#     best_prior_overlap.squeeze_(1)  # num_obj
#     # print('best_prior_overlap:',best_prior_overlap)
#     # print('best_prior_overlap:',best_prior_overlap.shape)
#     # 中索引为best_prior_idx_filter对应的位置填充为2。这个操作的目的是确保每个prior box都匹配到其对应的ground truth box，因为在之前
#     # 的操作中，可能有些prior box没有匹配到任何有效的ground truth boxbest_truth_overlap值设置为2，可以确保这些prior box在后续的处理中被视为与ground truth box匹配。
#     best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior  torch.Size([num_priors])
#     # print('best_truth_overlap:',best_truth_overlap)
#     # print('best_truth_overlap:',best_truth_overlap.shape)
#     # TODO refactor: index  best_prior_idx with long tensor
#     # ensure every gt matches with its prior of max overlap
#     for j in range(best_prior_idx.size(0)):   # [num_obj, 1]
#         best_truth_idx[best_prior_idx[j]] = j   # torch.Size([num_priors])
#     # print('best_truth_idx3:', best_truth_idx)
#     # print('best_truth_idx3:', best_truth_idx.shape)
#     matches = truths[best_truth_idx]  # Shape: [num_priors,4]
#     # print('match:', matches)
#     # print('match:', matches.shape)
#
#     conf = labels[best_truth_idx] + 1  # Shape: [num_priors]  # torch.Size([num_priors])
#     # print('labels:',labels)
#     # print('labels:', labels.shape)
#     # print('conf:', conf)
#     # print('conf:', conf.shape)
#     conf[best_truth_overlap < threshold] = 0  # label as background
#     # print('conf2:', conf)
#     # print('conf2:', conf.shape)
#     loc = encode(matches, priors, variances)
#
#     matches_landm = landms[best_truth_idx]
#     landm = encode_landm(matches_landm,priors,variances)
#     loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
#     conf_t[idx] = conf  # [num_priors] top class label for each prior
#     landm_t[idx] = landm


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].与每个先验框匹配的真实框的坐标（以点形式表示）,与先验框的交并比最大的那个真实框的坐标
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
        将真实坐标值相对于先验框的偏移和尺度信息进行编码
    """

    # dist b/t match center and prior's center 匹配真实框中心和先验框中心之间的距离 ，先验框是中心-大小的形式
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 匹配真实框的宽度和高度与先验框宽度和高度的比例
    g_wh = torch.log(g_wh) / variances[1]  # 取对数，并除以方差列表中的第二个值（variances[1]）来编码方差。
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4] 它包含了匹配真实框的中心偏移和宽高比例的编码信息，用于后续的平滑L1损失计算。

def encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
        将匹配的真实框的关键点与对应的先验框进行编码，以便在损失函数计算中使用。编码后的关键点坐标包含了关键点与先验框中心的偏移和尺度信息。
    """

    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))  # 其中5表示每个先验框有5个关键点，2表示每个关键点有2个坐标。
    # 分别表示先验框的中心坐标、宽度和高度，并扩展维度使其与matched的形状相匹配。
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    # print('priors:',priors)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]  # 计算关键点坐标与先验框中心坐标的差异
    # encode variance 对差异进行编码
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]  将编码后的关键点坐标展平为(num_priors, 10)的形状
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # print(boxes)
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

# def match_ious(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
#     """Match each prior box with the ground truth box of the highest jaccard
#     overlap, encode the bounding boxes, then return the matched indices
#     corresponding to both confidence and location preds.
#     Args:
#         threshold: (float) The overlap threshold used when mathing boxes.
#         truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
#         priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
#         variances: (tensor) Variances corresponding to each prior coord,
#             Shape: [num_priors, 4].
#         labels: (tensor) All the class labels for the image, Shape: [num_obj].
#         loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
#         conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
#         idx: (int) current batch index
#     Return:
#         The matched indices corresponding to 1)location and 2)confidence preds.
#     """
#     # jaccard index
#     loc_t[idx] = point_form(priors)
#     overlaps = jaccard(
#         truths,
#         point_form(priors)
#     )
#     # (Bipartite Matching)
#     # [1,num_objects] best prior for each ground truth
#     best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
#     # [1,num_priors] best ground truth for each prior
#     best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
#     best_truth_idx.squeeze_(0)
#     best_truth_overlap.squeeze_(0)
#     best_prior_idx.squeeze_(1)
#     best_prior_overlap.squeeze_(1)
#     best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
#     # TODO refactor: index  best_prior_idx with long tensor
#     # ensure every gt matches with its prior of max overlap
#     for j in range(best_prior_idx.size(0)):
#         best_truth_idx[best_prior_idx[j]] = j
#     matches = truths[best_truth_idx]          # Shape: [num_priors,4]
#     conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
#     conf[best_truth_overlap < threshold] = 0  # label as background
#     loc_t[idx] = matches    # [num_priors,4] encoded offsets to learn
#     conf_t[idx] = conf  # [num_priors] top class label for each prior
#
#
# def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
#     """Match each prior box with the ground truth box of the highest jaccard
#     overlap, encode the bounding boxes, then return the matched indices
#     corresponding to both confidence and location preds.
#     Args:
#         threshold: (float) The overlap threshold used when mathing boxes.
#         truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
#         priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
#         variances: (tensor) Variances corresponding to each prior coord,
#             Shape: [num_priors, 4].
#         labels: (tensor) All the class labels for the image, Shape: [num_obj].
#         loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
#         conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
#         idx: (int) current batch index
#     Return:
#         The matched indices corresponding to 1)location and 2)confidence preds.
#     """
#     # jaccard index
#     overlaps = jaccard(
#         truths,
#         point_form(priors)
#     )
#     # (Bipartite Matching)
#     # [1,num_objects] best prior for each ground truth
#     best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
#     # [1,num_priors] best ground truth for each prior
#     best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
#     best_truth_idx.squeeze_(0)
#     best_truth_overlap.squeeze_(0)
#     best_prior_idx.squeeze_(1)
#     best_prior_overlap.squeeze_(1)
#     best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
#     # TODO refactor: index  best_prior_idx with long tensor
#     # ensure every gt matches with its prior of max overlap
#     for j in range(best_prior_idx.size(0)):
#         best_truth_idx[best_prior_idx[j]] = j
#     matches = truths[best_truth_idx]          # Shape: [num_priors,4]
#     conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
#     conf[best_truth_overlap < threshold] = 0  # label as background
#     loc = encode(matches, priors, variances)
#     loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
#     conf_t[idx] = conf  # [num_priors] top class label for each prior
#
#
# def encode(matched, priors, variances):
#     """Encode the variances from the priorbox layers into the ground truth boxes
#     we have matched (based on jaccard overlap) with the prior boxes.
#     Args:
#         matched: (tensor) Coords of ground truth for each prior in point-form
#             Shape: [num_priors, 4].
#         priors: (tensor) Prior boxes in center-offset form
#             Shape: [num_priors,4].
#         variances: (list[float]) Variances of priorboxes
#     Return:
#         encoded boxes (tensor), Shape: [num_priors, 4]
#     """
#
#     # dist b/t match center and prior's center
#     g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
#     # encode variance
#     g_cxcy /= (variances[0] * priors[:, 2:])
#     # match wh / prior wh
#     g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
#     g_wh = torch.log(g_wh) / variances[1]
#     # return target for smooth_l1_loss
#     return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
#
#
# # Adapted from https://github.com/Hakuyume/chainer-ssd
# def decode(loc, priors, variances):
#     """Decode locations from predictions using priors to undo
#     the encoding we did for offset regression at train time.
#     Args:
#         loc (tensor): location predictions for loc layers,
#             Shape: [num_priors,4]
#         priors (tensor): Prior boxes in center-offset form.
#             Shape: [num_priors,4].
#         variances: (list[float]) Variances of priorboxes
#     Return:
#         decoded bounding box predictions
#     """
#
#     boxes = torch.cat((
#         priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
#     boxes[:, :2] -= boxes[:, 2:] / 2
#     boxes[:, 2:] += boxes[:, :2]
#     #print(boxes)
#     return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def diounms(boxes, scores, overlap=0.5, top_k=200, beta1=1.0):
    """Apply DIoU-NMS at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
        beta1: (float) DIoU=IoU-R_DIoU^{beta1}.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        inx1 = torch.clamp(xx1, min=x1[i])
        iny1 = torch.clamp(yy1, min=y1[i])
        inx2 = torch.clamp(xx2, max=x2[i])
        iny2 = torch.clamp(yy2, max=y2[i])
        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (xx1 + xx2) / 2
        center_y2 = (yy1 + yy2) / 2
        d = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
        cx1 = torch.clamp(xx1, max=x1[i])
        cy1 = torch.clamp(yy1, max=y1[i])
        cx2 = torch.clamp(xx2, min=x2[i])
        cy2 = torch.clamp(yy2, min=y2[i])
        c = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
        u = d / c
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = inx2 - inx1
        h = iny2 - iny1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union - u ** beta1  # store result in diou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


