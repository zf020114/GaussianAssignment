# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import cv2
import os
import numpy as np
from torchvision import transforms

@DETECTORS.register_module()
class TTFNet(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        if self.bbox_head.freeze_backbone:
            with torch.no_grad():
                x = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        if self.bbox_head.debug:
            if not os.path.isdir('./debug/'):
                os.makedirs('./debug/') 
            for i in range(img.shape[0]):
                img_name=os.path.splitext(img_metas[i]['ori_filename'])[0]
    
                image_show=self.imshow_gpu_tensor(img[i])
                image_heatmap = self.imshow_gpu_tensor_mask(self.bbox_head.heatmap[i].max(0)[0]*255)
                image_weight = self.imshow_gpu_tensor_mask(self.bbox_head.reg_weight[i][0]*255)
                box_target_l = self.imshow_gpu_tensor_mask(self.bbox_head.box_target[i][0]*20+20)
                label_target =  self.imshow_gpu_tensor_mask((self.bbox_head.labels[i])*200)
                                
                heatmap = self.imshow_gpu_tensor_mask(self.bbox_head.cls_scores[i].max(0)[0]*255)
                bbox_pred =self.imshow_gpu_tensor_mask(self.bbox_head.bbox_preds[i][0]*255)
                
                image_heatmap = cv2.resize(image_heatmap,image_show.shape[0:2])
                image_weight = cv2.resize(image_weight,image_show.shape[0:2])
                box_target = cv2.resize(box_target_l,image_show.shape[0:2])
                heatmap = cv2.resize(heatmap,image_show.shape[0:2] )
                bbox_pred = cv2.resize(bbox_pred,image_show.shape[0:2] )
                label_target = cv2.resize(label_target,image_show.shape[0:2] )
                
                cv2.imwrite('./debug/{}.jpg'.format(img_name), image_show)
                cv2.imwrite('./debug/{}_heatmap.jpg'.format(img_name), image_heatmap)
                cv2.imwrite('./debug/{}_mask.jpg'.format(img_name), image_weight)
                cv2.imwrite('./debug/{}_labels.jpg'.format(img_name), label_target)

                # cv2.imwrite('./debug/{}_boxtarget.jpg'.format(img_name), box_target)
                cv2.imwrite('./debug/{}_heatmap_pre.jpg'.format(img_name), heatmap)
                cv2.imwrite('./debug/{}_boxpred.jpg'.format(img_name), bbox_pred)
                
        return losses
    
    

    def imshow_gpu_tensor_mask(self, tensor):#调试中显示表标签图

        device=tensor[0].device
        mean= torch.tensor([0, 0, 0])
        std= torch.tensor([1,1,1])
        mean=mean.to(device)
        std=std.to(device)
        tensor = (tensor.squeeze() * std[:,None,None]) + mean[:,None,None]
        tensor=tensor[0:1]
        if len(tensor.shape)==4:
            image = tensor.permute(0,2, 3,1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2,0).cpu().clone().detach().numpy()
        image = image.astype(np.uint8).squeeze()
        return image

    def imshow_gpu_tensor(self, tensor):#调试中显示表标签图
        device=tensor[0].device
        mean= torch.tensor([123.675, 116.28, 103.53])
        std= torch.tensor([58.395, 57.12, 57.375])
        mean=mean.to(device)
        std=std.to(device)
        tensor = (tensor.squeeze() * std[:,None,None]) + mean[:,None,None]
        if len(tensor.shape)==4:
            image = tensor.permute(0,2, 3,1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2,0).cpu().clone().detach().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        return img