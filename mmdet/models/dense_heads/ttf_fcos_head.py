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
from .anchor_free_head import AnchorFreeHead
# from mmdet.models.necks.dyhead_mmdet import DyHead
INF = 1e8


@HEADS.register_module()
class TTF_Fcos_Head(AnchorFreeHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 channels=128,
                 down_ratio=4,
                 head_conv=128,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_kernel=3,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 iou_branch=False,
                 dcn_on_last_conv=False,
                 use_dyhead=False,
                 dyhead_NUM_CONVS=6,
                 alpha=1.0,
                 beta=1.0,
                 debug=False,
                 pos_thr_cls=1e-2,
                 max_radius=16,
                 strides=[4],
                 regress_ranges=((-1, 1024)),
                 loss_cls=dict(
                     type='ct_focal_loss',
                     thr=0.1,
                     loss_weight=1.0),
                 loss_bbox= dict(  type='giou_loss',loss_weight=5.0),
                 loss_centerness = dict(type='L1Loss', reduction='mean', loss_weight=1.0),
                 freeze_backbone=False,
                 use_sigmoid=True,
                train_cfg=None,
                 test_cfg=None,):
        super().__init__(
            num_classes,
            128,  strides=strides,)
        assert len(planes) in [2, 3, 4]
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.planes = planes
        self.head_conv = head_conv
        self.channels=channels
        self.num_classes = num_classes
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.fp16_enabled = False
        self.debug =debug
        self.pos_thr =pos_thr_cls
        self.iou_branch =iou_branch
        self.dcn_on_last_conv=dcn_on_last_conv
        self.use_dyhead =use_dyhead
        self.down_ratio = down_ratio
        self.wh_planes = 4 
        self.base_loc = None
        self.loss_cls=build_loss(loss_cls)
        self.loss_bbox=build_loss(loss_bbox)
        self.loss_centerness =build_loss(loss_centerness)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.freeze_backbone=freeze_backbone
        self.use_sigmoid=use_sigmoid
        self.max_radius =max_radius
        self.pos_anchor=[]
        
        self.strides = strides
        self.regress_ranges = regress_ranges
        # repeat upsampling n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self.build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
            self.build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self.build_upsample(planes[i - 1], planes[i], norm_cfg=norm_cfg))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self.build_shortcut(
            inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
            kernel_size=shortcut_kernel, padding=padding)
        
        if self.use_dyhead:
            self.DyHead=DyHead(   in_channels = channels,
                    channels = channels,
                    NUM_CONVS=dyhead_NUM_CONVS,
                    strides = [down_ratio],
                    out_features =['p2'],
                    out_feature_channels ={'p2': channels},
                    size_divisibility = 4)
        # heads
        self.cls_convs = self.feature_convs(wh_head_conv_num, head_conv)
        self.reg_convs = self.feature_convs(hm_head_conv_num,wh_conv)
        self.hm = self.build_head(head_conv,self.num_classes)
        self.wh = self.build_head(wh_conv, self.wh_planes)
        if self.iou_branch:
            self.iou = self.build_head(wh_conv,1)
        self.pos_anchor=[]
        self.center_sampling = False
        self.norm_on_bbox = False
        
    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes, norm_cfg=None):
        mdcn = DeformConv2dPack(inplanes, planes, 3, stride=1,
                                       padding=1, dilation=1, deformable_groups=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        return nn.Sequential(*layers)

    def feature_convs(self,  conv_num=1, head_conv_plane=None):
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            inp = self.channels if i == 0 else head_conv_plane
            if self.dcn_on_last_conv and i == conv_num - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = None
            head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1,conv_cfg=conv_cfg,))
        return nn.Sequential(*head_convs)
    
    def build_head(self, inp, out_channel,):
        head_convs = []
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)
    
    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:`
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]

        for i, upsample_layer in enumerate(self.deconv_layers):
            x = upsample_layer(x)
            if i < len(self.shortcut_layers):
                shortcut = self.shortcut_layers[i](feats[-i - 2])
                x = x + shortcut

        if self.use_dyhead:
            feature_dict={0:x }
            x=self.DyHead(feature_dict)[0]
            
        cls_feature = self.cls_convs(x)
        reg_feature =  self.reg_convs(x)
        
        hm = self.hm(cls_feature) 
        wh = F.relu(self.wh(reg_feature)) * self.wh_offset_base
        iou = self.iou(reg_feature)
         
        return [hm], [wh],[iou]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        self.featmap_sizes = featmap_sizes
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            # pos_decoded_target_preds = self.bbox_coder.decode(
            #     pos_points, pos_bbox_targets)
            pos_decoded_target_preds = pos_bbox_targets#
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        # labels_list1, bbox_targets_list1 = multi_apply(
        #     self._get_target_single,
        #     gt_bboxes_list,
        #     gt_labels_list,
        #     points=concat_points,
        #     regress_ranges=concat_regress_ranges,
        #     num_points_per_lvl=num_points)

        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single_ga,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points
            )
        
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        # self.pos_anchor_box.append(gt_bboxes)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
     
        self.pos_anchor.append(inside_gt_bbox_mask.sum(0))
        
        return labels, bbox_targets

    def _get_target_single_ga(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
                
        def bbox_areas(bboxes, keep_axis=False):
            x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            areas = (y_max - y_min + 1) * (x_max - x_min + 1)
            if keep_axis:
                return areas[:, None]
            return areas

        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        feat_shape=self.featmap_sizes
        # self.pos_anchor_box.append(gt_bboxes)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
                   
        output_h, output_w = feat_shape[0][0], feat_shape[0][1]
        heatmap_channel = self.num_classes

        heatmap = gt_bboxes.new_zeros((heatmap_channel, output_h, output_w))
        labels = gt_bboxes.new_ones((output_h, output_w))*self.num_classes
        fake_heatmap = gt_bboxes.new_zeros((output_h, output_w))
        bbox_targets = gt_bboxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        
        boxes_areas_log = bbox_areas(gt_bboxes).log()

        _, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        gt_bboxes = gt_bboxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_bboxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
        
        feat_hs =torch.clamp(feat_hs, min=0, max=self.max_radius/self.down_ratio)
        feat_ws =torch.clamp(feat_ws, min=0, max=self.max_radius/self.down_ratio)
        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2,
                                (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] 
            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            box_target_inds = fake_heatmap >= self.pos_thr
            # self.pos_anchor.append(box_target_inds.sum()[None])
            bbox_targets[:, box_target_inds] = gt_bboxes[k][:, None]
            labels[box_target_inds] = box_target_inds.float()[box_target_inds]*(cls_id)
    
        labels=labels.reshape(-1)
        labels = labels.long()
        
        bbox_targets = bbox_targets.permute(1,2,0).reshape(-1,4)
        
        # xs, ys = points[:, 0], points[:, 1]
        # left = xs - bbox_targets[..., 0]
        # right = bbox_targets[..., 2] - xs
        # top = ys - bbox_targets[..., 1]
        # bottom = bbox_targets[..., 3] - ys
        # bbox_targets = torch.stack((left, top, right, bottom), -1)
        return labels, bbox_targets

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

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
