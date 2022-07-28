# Copyright (c) OpenMMLab. All rights reserved.
import random
import cv2,os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
import numpy as np
from torchvision import transforms
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result

@DETECTORS.register_module()
class YOLOTTF(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size.
            Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None):
        super(YOLOTTF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      ):
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
        # Multi-scale training
        # img, gt_bboxes = self._preprocess(img, gt_bboxes)
 
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        # # random resizing
        # if (self._progress_in_iter + 1) % self._random_size_interval == 0:
        #     self._input_size = self._random_resize() 
        # self._progress_in_iter += 1
 
        if self.bbox_head.debug:
            if not os.path.isdir('./debug/'):
                os.makedirs('./debug/') 
            for i in range(img.shape[0]):
                img_name=os.path.splitext(img_metas[i]['ori_filename'])[0]
    
                image_show=self.imshow_gpu_tensor(img[i])
                image_heatmap = self.imshow_gpu_tensor_mask(self.bbox_head.heatmap[i].max(0)[0])*255
                image_weight = self.imshow_gpu_tensor_mask(self.bbox_head.reg_weight[i])*255
                box_target_l = self.imshow_gpu_tensor_mask(self.bbox_head.box_target[i][0]+1)*20
                                
                heatmap = self.imshow_gpu_tensor_mask(self.bbox_head.cls_scores[i].max(0)[0])*255
                bbox_pred =self.imshow_gpu_tensor_mask(self.bbox_head.bbox_preds[i][0])*255
                
                image_heatmap = cv2.resize(image_heatmap,image_show.shape[0:2])
                image_weight = cv2.resize(image_weight,image_show.shape[0:2])
                box_target = cv2.resize(box_target_l,image_show.shape[0:2])
                heatmap = cv2.resize(heatmap,image_show.shape[0:2] )
                bbox_pred = cv2.resize(bbox_pred,image_show.shape[0:2] )
                
                cv2.imwrite('./debug/{}.jpg'.format(img_name), image_show)
                cv2.imwrite('./debug/{}_heatmap.jpg'.format(img_name), image_heatmap)
                cv2.imwrite('./debug/{}_mask.jpg'.format(img_name), image_weight)
                cv2.imwrite('./debug/{}_boxtarget.jpg'.format(img_name), box_target)
                
                cv2.imwrite('./debug/{}_heatmap_pre.jpg'.format(img_name), heatmap)
                cv2.imwrite('./debug/{}_boxpred.jpg'.format(img_name), bbox_pred)
        return losses

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self):
        tensor = torch.LongTensor(2).cuda()

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_multiplier * size, self._size_multiplier * size)
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
 
    
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
        # tensor = (tensor.squeeze() * std[:,None,None]) + mean[:,None,None]
        tensor=tensor[0:1]
        if len(tensor.shape)==4:
            image = tensor.permute(0,2, 3,1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2,0).cpu().clone().detach().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        return img
    