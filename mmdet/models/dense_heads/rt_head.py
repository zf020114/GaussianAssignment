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
                        build_assigner, build_sampler, multi_apply,multiclass_nms,multiclass_nms_rotated_bbox,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from mmcv.cnn import normal_init, kaiming_init
from mmcv.ops import DeformConv2dPack
from mmcv.cnn import build_norm_layer
# from mmdet.models.necks.dyhead_mmdet import DyHead
from mmcv.ops import box_iou_rotated
from mmdet.core.ttf_core import *

@HEADS.register_module()
class RTHead(BaseDenseHead, BBoxTestMixin):

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
                 degug =False,
                 iou_branch=False,
                 dcn_on_last_conv=False,
                 use_dyhead=False,
                 
                 dyhead_NUM_CONVS=6,
                 ues_rotated_boxes=True,
                 alpha=1.0,
                 beta=1.0,
                 max_radius=16,
                 pos_thr=0.1,
                 loss_cls=dict(
                     type='ct_focal_loss',
                     thr=0.1,
                     loss_weight=1.0),
                 loss_bbox= dict(  type='giou_loss',loss_weight=5.0),
                 loss_iou = dict(type='L1Loss', reduction='mean', loss_weight=1.0),
                 freeze_backbone=False,
                 use_sigmoid=True,
                train_cfg=None,
                 test_cfg=None,):
        super(BaseDenseHead, self).__init__()
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
        
        self.iou_branch =iou_branch
        self.dcn_on_last_conv=dcn_on_last_conv
        self.use_dyhead =use_dyhead
        self.down_ratio = down_ratio
        self.wh_planes = 5 if ues_rotated_boxes else 4 
        self.base_loc = None
        self.loss_cls=loss_cls
        self.loss_bbox=build_loss(loss_bbox)
        self.loss_iou =build_loss(loss_iou)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.freeze_backbone=freeze_backbone
        self.use_sigmoid=use_sigmoid
        self.ues_rotated_boxes=ues_rotated_boxes
        self.debug =degug
        self.max_radius =max_radius
        self.pos_thr = pos_thr
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
        # wh = F.relu(self.wh(reg_feature)) * self.wh_offset_base
        wh =self.wh(reg_feature)
        wh[:, 0:2, :, :] *=  self.wh_offset_base
        wh[:, 2:4, :, :] = wh[:, 2:4, :, :] .exp() * self.wh_offset_base
        if self.iou_branch:
            iou = self.iou(reg_feature).sigmoid()
            wh = torch.cat((wh,iou),dim=1)
        return hm, wh

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
            
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
                
            points = self.base_loc.permute(1, 2, 0).reshape(-1, 2)
            if self.iou_branch:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 6)
            else:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)

            cfg =self.test_cfg
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores ).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            # bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            bbox_pred[:, [0, 1]]+=points
            bboxes = bbox_pred
         
            if rescale:
                bboxes[...,0:4] /= bboxes.new_tensor(scale_factor)
 
            padding = scores.new_zeros(scores.shape[0], 1)
            mlvl_scores = torch.cat([ scores,padding], dim=1)
            # mlvl_scores=scores

            ious = bbox_pred[:,5]
            det_bboxes,  det_labels = multiclass_nms_rotated_bbox(bboxes[:,0:5],
                                                    mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img,score_factors=ious)
            #因为后面的参数 是rbox 和ploypoint
            det_bboxes=det_bboxes[:,0:5]
    
            result_list.append([det_bboxes, det_labels])
        return result_list
    
    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_masks,
             gt_bboxes_ignore=None):
        
        #将gt_mask转换维rbox
        gt_mask_arrays=[]
        for i in  gt_masks:
            gt_mask_array=np.array(i.masks).squeeze().astype(float)
            gt_mask_array=gt_bboxes[0].new_tensor(gt_mask_array)
            if gt_mask_array.shape[0]==0:
                gt_mask_array=gt_bboxes[0].new_tensor(gt_mask_array)
            else:
                gt_mask_array = poly_to_rotated_box(gt_mask_array.reshape(-1,8))
            gt_mask_arrays.append(gt_mask_array)
        gt_masks=gt_mask_arrays
        
        all_targets = self.target_generator(gt_bboxes, gt_labels, gt_masks,img_metas)
        if self.debug:
            self.heatmap, self.box_target, self.reg_weight,self.labels = all_targets
            self.cls_scores = pred_heatmap
            self.bbox_preds = pred_wh
        return  self.loss_calc(pred_heatmap, pred_wh, *all_targets)

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
        num_imgs,_,H, W = pred_hm.shape
        loss_dict={}
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        if self.loss_cls['type']=='diceloss':
             hm_loss = dice_loss(pred_hm, heatmap) * self.loss_cls['loss_weight']
        elif self.loss_cls['type']=='ct_focal_loss':
            hm_loss = ct_focal_loss(pred_hm, heatmap, self.loss_cls.thr) * self.loss_cls['loss_weight']
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
        pred_boxes = torch.cat((self.base_loc + pred_wh[:, [0, 1]],
                               pred_wh[:, [2, 3,4]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        pos_mask = mask > 0
        pos_pred_boxes = pred_boxes[pos_mask]
        pos_target_boxes = boxes[pos_mask]
        weight = mask[pos_mask].float()[:,None].repeat(1,self.wh_planes)
        wh_loss=self.loss_bbox(pos_pred_boxes,   pos_target_boxes.detach(),   weight=weight,  avg_factor=avg_factor)
        loss_dict.update(loss_cls=hm_loss)
        loss_dict.update(loss_box=wh_loss)
        
        if self.iou_branch:      #计算iou_loss
            if pos_pred_boxes.shape[0]==0:
                iou_loss= pred_boxes.sum() * 0
            else:
                flatten_pred_ious = pred_wh[:, 5:6].permute(0, 2, 3, 1)
                flatten_pred_ious = flatten_pred_ious[pos_mask]
                target_ious=  box_iou_rotated(
                    pos_pred_boxes,
                    pos_target_boxes.detach(),
                    aligned=True)
                iou_loss = self.loss_iou(flatten_pred_ious.squeeze(),target_ious)
            loss_dict.update(loss_iou=iou_loss)
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

    def rot_img(self,x, theta):
        # theta = torch.tensor(theta)theta
        theta=theta*(-1.0)#.to(device)#表示这里是反的顺时针旋转 是负号
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]).to(x.device)
        rot_mat=rot_mat[None, ...]
        out_size = torch.Size((1, 1,  x.size()[0],  x.size()[1]))
        grid = F.affine_grid(rot_mat, out_size).float()
        rotate = F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid)
        return rotate.squeeze()
    
    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius,angle,k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y, device=heatmap.device)
        #这里添加转换函数
        max_size=max(gaussian.size()[0],gaussian.size()[1])
        gaussian_expand=torch.zeros((max_size,max_size),device=heatmap.device)#.to(device)
        if gaussian.size()[0]==max_size:
            start=int((max_size-gaussian.size()[1])/2)
            gaussian_expand[:,start:start+gaussian.size()[1]]=gaussian
        else:
            start=int((max_size-gaussian.size()[0])/2)
            gaussian_expand[start:start+gaussian.size()[0],:]=gaussian

        rotated_im = self.rot_img(gaussian_expand, angle) 

        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        max_radius=max(w_radius,h_radius)
        left, right = min(x, max_radius), min(width - x, max_radius + 1)#为了不越界
        top, bottom = min(y, max_radius), min(height - y, max_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        if max_size==1:
            masked_gaussian=rotated_im.unsqueeze(0).unsqueeze(0)#[y - top:y + bottom]
        else:
            masked_gaussian = rotated_im[max_radius - top:max_radius + bottom,
                                max_radius - left:max_radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian.float() * k, out=masked_heatmap)
        return heatmap


    def target_single_image(self, gt_boxes, gt_labels, gt_rboxes, feat_shape):
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
        reg_weight = gt_boxes.new_zeros((1, output_h, output_w))

        if gt_rboxes.shape[0]==0:
            return heatmap, box_target, reg_weight,labels
        
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
        gt_rboxes = gt_rboxes[boxes_ind]

        # 设置画中心点热图的参数
        gt_rboxes_down =gt_rboxes/self.down_ratio
        ct_ints =gt_rboxes_down[:,0:2].to(torch.int)
        feat_hs, feat_ws = (gt_rboxes_down[:,3],  gt_rboxes_down[:,2])
        #限制最大半径
        feat_hs =torch.clamp(feat_hs, min=0, max=self.max_radius/self.down_ratio)
        feat_ws =torch.clamp(feat_ws, min=0, max=self.max_radius/self.down_ratio)
        
        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        angles=gt_rboxes[:,4]
        
        #对于单张图像开始遍历图上的所有bbox
        for k in range(gt_rboxes.shape[0]):
            cls_id = gt_labels[k] 
            #生成椭圆旋转高斯热图
            fake_heatmap = fake_heatmap.zero_()
            fake_heatmap = self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item(),angles[k])
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)
            box_target_inds = fake_heatmap >self.pos_thr
            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            
            box_target[:, box_target_inds] = gt_rboxes[k].reshape(5,-1).repeat(1, box_target_inds.sum())
            reg_weight[0, box_target_inds] = local_heatmap / ct_div
            labels[box_target_inds] = box_target_inds.float()[box_target_inds]*( gt_labels[k] )
        return heatmap, box_target, reg_weight,labels

    def target_generator(self, gt_boxes, gt_labels, gt_masks, img_metas):
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
                gt_masks,
                feat_shape=feat_shape
            )
            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            labels =  torch.stack(labels, dim=0).detach()
            return heatmap, box_target, reg_weight, labels

  
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) 
        losses = self.loss(*loss_inputs, gt_masks=gt_masks, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
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
