# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms

from mmcv.runner import force_fp32
import numpy as np
from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,bbox_overlaps,
                        build_assigner, build_sampler, multi_apply,multiclass_nms,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from mmcv.cnn import normal_init, kaiming_init
from mmcv.ops import DeformConv2dPack
from mmcv.cnn import build_norm_layer

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res[0] if return_single else res

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)

def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas
    
def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return (x1, y1, x2, y2)

def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight)[None] / avg_factor

def ct_focal_loss(pred, gt, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos

def ct_focal_loss2(pred, gt, gamma=2.0,thr=0.1):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    pos_inds = (gt>=thr).float()
    neg_inds = (gt<thr).float()

    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos

@HEADS.register_module()
class SegDetHead(BaseDenseHead, BBoxTestMixin):

    def __init__(self,
                in_channels=[32, 64, 160, 256],
                feat_channels=128,
                 down_ratio=4,
                 stacked_convs=2,
                 num_classes=81,
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 alpha=0.54,
                 beta=0.54,
                 iou_branch=False,
                 use_sigmoid=True,
                 use_depthwise = False,
                 dcn_on_last_conv =False,
                 freeze_backbone=False,
                 norm_cfg=dict(type='BN'),
                 loss_cls=dict(
                     type='ct_focal_loss',
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox= dict(  type='giou_loss' ,loss_weight=5.0),
                 loss_iou = dict(type='L1Loss', reduction='mean', loss_weight=1.0),
                train_cfg=None,
                 test_cfg=None,):
        super(BaseDenseHead, self).__init__()
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.fp16_enabled = False
        self.down_ratio =down_ratio
        self.num_classes=num_classes
        self.wh_planes = 4 
        self.base_loc = None
        if not use_sigmoid:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls=loss_cls
        self.loss_bbox=loss_bbox
        self.loss_iou =build_loss(loss_iou)
        
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.in_channels =in_channels
        self.feat_channels =feat_channels
        self.norm_cfg =norm_cfg
        self.act_cfg = dict(type='ReLU')
        self.interpolate_mode= 'bilinear'
        self.align_corners = False
        self.iou_branch =iou_branch
        self.stacked_convs = stacked_convs
        self.freeze_backbone =freeze_backbone
        self.conv_cfg=None
        self.conv_bias='auto'
        self.use_sigmoid=use_sigmoid
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv =dcn_on_last_conv
        num_inputs = len(self.in_channels)
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.feat_channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.feat_channels * num_inputs,
            out_channels=self.feat_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.cls_convs=self._build_stacked_convs()
        self.reg_convs =self._build_stacked_convs()
        self.conv_cls, self.conv_reg, self.conv_obj = self._build_predictor()
        
        
    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule  if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)
    
    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        if self.use_sigmoid:
            conv_cls = nn.Conv2d(self.feat_channels, self.num_classes, 1)
        else:
            conv_cls = nn.Conv2d(self.feat_channels, self.num_classes+1, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self):
        bias_init = bias_init_with_prob(0.01)

        self.conv_cls.bias.data.fill_(bias_init)
        self.conv_obj.bias.data.fill_(bias_init)
        
        for _, m in self.convs.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        
        for _, m in self.fusion_conv.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        outs = []
        for idx in range(len(feats)):
            x = feats[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=feats[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        x = self.fusion_conv(torch.cat(outs, dim=1))

        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        cls_score = self.conv_cls(cls_feat)
        bbox_pred =F.relu( self.conv_reg(reg_feat))* self.wh_offset_base
        if self.iou_branch:
            iou =  self.conv_obj(reg_feat).sigmoid()
            bbox_pred = torch.cat((bbox_pred,iou),dim=1)
        return cls_score, bbox_pred
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   rescale=False):
        
        batch, cat, H, W= cls_scores.size()
        #获取anchor点
        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=cls_scores.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=cls_scores.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)
            
        result_list = []
        #对每个图像分别计算
        for img_id in range(len(img_metas)):
            cls_score =  cls_scores[img_id].detach() 
            bbox_pred =   bbox_preds[img_id].detach()
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            
            if self.use_sigmoid:
                scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            else:
                scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes+1)
                scores = F.softmax(scores, dim=1)[:,1:]
                
            points = self.base_loc.permute(1, 2, 0).reshape(-1, 2)
            if self.iou_branch:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            else:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            cfg =self.test_cfg
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores ).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            
            if rescale:
                bboxes /= bboxes.new_tensor(scale_factor)
 
            padding = scores.new_zeros(scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, scores], dim=1)
            if self.iou_branch:
                ious = bbox_pred[:,4]
                det_bboxes, det_labels = multiclass_nms(
                bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,score_factors=ious)
            else:
                det_bboxes, det_labels = multiclass_nms(
                bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
            result_list.append([det_bboxes, det_labels])
        return result_list
    

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        # hm_loss, wh_loss,iou_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets)
        return self.loss_calc(pred_heatmap, pred_wh, *all_targets)

    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  heatmap,
                  box_target,
                  wh_weight,
                  labels):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        num_imgs,num_class,H, W = pred_hm.shape
        if not self.use_sigmoid:
            labels=labels.long().squeeze(1)
            hm_loss = self.loss_cls(pred_hm, labels)
        elif self.loss_cls['type']=='ct_focal_loss':
            pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
            hm_loss = ct_focal_loss(pred_hm, heatmap) * self.loss_cls['loss_weight']
        elif self.loss_cls['type']=='ct_focal_loss2':
            pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
            hm_loss = ct_focal_loss2(pred_hm, heatmap,self.loss_cls.thr) * self.loss_cls['loss_weight']
        elif self.loss_cls['type']=='VarifocalLoss':
            pos_inds = torch.where(heatmap > 1e-2)[0]
            num_pos = len(pos_inds)+1.0
            flatten_cls_scores = pred_hm.permute(0, 2, 3, 1).reshape(-1, num_class).contiguous()
            cls_iou_targets = heatmap.permute(0, 2, 3, 1).reshape(-1, num_class).contiguous()
            hm_loss =varifocal_loss(flatten_cls_scores,  cls_iou_targets, avg_factor=num_pos)* self.loss_cls['loss_weight']
        elif self.loss_cls['type']=='FocalLoss':
            pos_inds = torch.where(heatmap > 1e-2)[0]
            pos_inds =torch.where(labels > 0)[0] 
            num_pos = len(pos_inds)+1.0
            flatten_cls_scores = pred_hm.permute(0, 2, 3, 1).reshape(-1, num_class).contiguous()
            cls_iou_targets = labels.reshape(-1).contiguous().long()
            fcos_loss = build_loss(self.loss_cls)
            hm_loss =fcos_loss(flatten_cls_scores,  cls_iou_targets, avg_factor=num_pos)* self.loss_cls['loss_weight']
        else:
            print('error')

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4
        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)
        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        if self.loss_bbox['type']=='giou_dis_loss':
            wh_loss = giou_dis_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) *  self.loss_bbox['loss_weight']
        elif self.loss_bbox['type']=='giou_loss':
            wh_loss = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) *  self.loss_bbox['loss_weight']

        loss_dict = dict( loss_cls=hm_loss, loss_bbox=wh_loss)
        if self.iou_branch:
            #计算iou_loss
            flatten_target_boxes = boxes.reshape(-1, 4)
            # pos_inds = (heatmap.max(1)[0].reshape(-1,1)>=self.pos_threshold).squeeze()
            pos_inds = (mask>0).reshape(-1,1).squeeze()
            flatten_pred_boxes =  pred_boxes.reshape(-1, 4)
            flatten_pred_ious = pred_wh[:, 4:5].permute(0, 2, 3, 1).reshape(-1, 1)
            flatten_pred_boxes = flatten_pred_boxes[pos_inds]
            flatten_target_boxes = flatten_target_boxes[pos_inds]
            flatten_pred_ious = flatten_pred_ious[pos_inds]
            target_ious= bbox_overlaps(  flatten_pred_boxes, flatten_target_boxes.detach(), is_aligned=True)
            # iou_loss = torch.abs(flatten_pred_ious - target_ious).mean()*self.loss_iou['loss_weight']
            iou_loss = self.loss_iou(flatten_pred_ious.squeeze(),target_ious)
            loss_dict.update(loss_iou=iou_loss)
        if not self.use_sigmoid:
            acc_seg= accuracy(pred_hm, labels)
            loss_dict.update(acc_seg=acc_seg)
        return loss_dict
    
    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian2D(self, shape, sigma_x=1, sigma_y=1, dtype=torch.float32, device='cpu'):
        """Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma (int): Sigma of gaussian function. Default: 1.
            dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
            device (str): Device of gaussian tensor. Default: 'cpu'.

        Returns:
            h (Tensor): Gaussian kernel with a
                ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        x = torch.arange(-n, n + 1, dtype=dtype, device=device).view(1, -1)
        y = torch.arange(-m, m + 1, dtype=dtype, device=device).view(-1, 1)
        h = (-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y))).exp()
        # h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y,dtype=heatmap.dtype, device=heatmap.device)
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_classes

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        labels = gt_boxes.new_zeros((output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian: #False
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_gaussian:#True
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 1e-3
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:#True
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:#True
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()
            labels[box_target_inds] = box_target_inds.float()[box_target_inds]*(gt_labels[k] )
        return heatmap, box_target, reg_weight,labels

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight,labels = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            labels =  torch.stack(labels, dim=0).detach()
            return heatmap, box_target, reg_weight, labels

