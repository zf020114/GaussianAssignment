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


@DETECTORS.register_module()
class YOLOSeg(SingleStageDetector):
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
        super(YOLOSeg, self).__init__(backbone, neck, bbox_head, train_cfg,
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
                      gt_masks=None,
                      gt_contours=None):
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

        # losses = super(YOLOSeg, self).forward_train(img, img_metas, gt_bboxes,
        #                                           gt_labels, gt_bboxes_ignore,gt_contours)
        if self.bbox_head.freeze_backbone:
            with torch.no_grad():
                x = self.extract_feat(img)
        else: 
            x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,gt_contours)
        # random resizing
        # if (self._progress_in_iter + 1) % self._random_size_interval == 0:
        #     self._input_size = self._random_resize()
        # self._progress_in_iter += 1
 
        if self.bbox_head.debug:
            if not os.path.isdir('./debug/'):
                os.makedirs('./debug/') 
            for i in range(img.shape[0]):
                img_name=os.path.splitext(img_metas[i]['ori_filename'])[0]
                pos_bbox_targets = self.bbox_head.pos_bbox_targets
                pos_img_ids = self.bbox_head.pos_img_ids
                pos_points=self.bbox_head.pos_points

                img_id_mask=pos_img_ids ==i
                img_box_masks = pos_bbox_targets[img_id_mask]
                for j in range(img_box_masks.shape[0]):
                    #获得原始图像
                    image_show=self.imshow_gpu_tensor(img[i])
                    #获得目标框,mask 和正样本点
                    img_bbox_targets = pos_bbox_targets[img_id_mask][j]
                    img_contour_targets = self.bbox_head.pos_target_contors[img_id_mask][j].long()
                    img_points = pos_points[img_id_mask][j]
                    #获得预测的mask
                    img_mask_preds = self.bbox_head.pos_decoded_mask_preds[img_id_mask][j]

                     #画anchor点
                    image_show = cv2.circle(image_show, (int(img_points[0]), int(img_points[1])), 2, (255,0,0), 2) 

                    # #画真值框
                    # [xmin,ymin,xmax,ymax]=img_bbox_targets
                    # cv2.line(image_show, (int(xmin), int(ymin)), (int(xmin), int(ymax)), (0, 0, 255) , 1)
                    # cv2.line(image_show, (int(xmin), int(ymax)), (int(xmax), int(ymax)), (0, 0, 255) , 1)
                    # cv2.line(image_show, (int(xmax), int(ymax)), (int(xmax), int(ymin)), (0, 0, 255) , 1)
                    # cv2.line(image_show, (int(xmax), int(ymin)), (int(xmin), int(ymin)), (0, 0, 255) , 1)

                    ##绘制GT 轮廓
                    pts = img_contour_targets.reshape((-1, 1, 2)).cpu().numpy()
                    cv2.polylines(image_show, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
                    #绘制预测的36个点
                    b=img_mask_preds.permute(1,0).long().cpu().numpy()
                    # cv2.polylines(image_show, [b], isClosed=True, color=(0, 0, 255), thickness=1)
                    for k in range(self.bbox_head.angles_num):
                        image_show = cv2.circle(image_show, (b[k,0], b[k,1]), 1, (0,255,0), 1)  #画anchor点

                    
                    # [xmin,ymin,xmax,ymax]=img_bbox_pred#画预测获得的框
                    # cv2.line(image_show, (int(xmin), int(ymin)), (int(xmin), int(ymax)), (0, 255, 255) , 1)
                    # cv2.line(image_show, (int(xmin), int(ymax)), (int(xmax), int(ymax)), (0, 255, 255) , 1)
                    # cv2.line(image_show, (int(xmax), int(ymax)), (int(xmax), int(ymin)), (0, 255, 255) , 1)
                    # cv2.line(image_show, (int(xmax), int(ymin)), (int(xmin), int(ymin)), (0, 255, 255) , 1)
                    
                    # for k in range(self.bbox_head.angles_num):
                    #     img_points=img_pred_points[:,k]
                    #     image_show = cv2.circle(image_show, (int(img_points[0]), int(img_points[1])), 1, (0,255,0), 2)  #画anchor点

                    cv2.imwrite('./debug/{}{}_{}.jpg'.format(img_name, i, j), image_show)
                    # cv2.imwrite('./debug/{}{}_{}_mask.jpg'.format(img_name, i, j), image_mask*128)
                    # cv2.imwrite('./debug/{}{}_{}_mask_pred.jpg'.format(img_name, i, j), img_pred_masks*128)
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
 
    
    def simple_test(self, img, img_meta, rescale=False):
        
        def empty_results(results, cls_scores):
            """Generate a empty results."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results
        
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        results = [
            self.bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
            for det_bboxes, det_labels, det_masks in bbox_list]

        bbox_results = [results[0][0]]
        mask_results = [results[0][1]]
        

        return list(zip(bbox_results, mask_results))
    
    def bbox_mask2result(self, bboxes, masks, labels, num_classes, img_meta):
        '''bbox and mask 转成result mask要画图'''
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (Tensor): shape (n, 5)
            masks (Tensor): shape (n, 2, 36)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class 
        """
        ori_shape = img_meta['ori_shape']
        img_h, img_w, _ = ori_shape
        mask_results = [[] for _ in range(num_classes )]

        for i in range(masks.shape[0]):
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask = [masks[i].transpose(1,0).unsqueeze(1).int().data.cpu().numpy()]
            im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
            # rle = mask_util.encode(
            #     np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            label = labels[i]
            mask_results[label].append(im_mask)

        if bboxes.shape[0] == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes )
            ]
            return bbox_results, mask_results
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            bbox_results = [bboxes[labels == i, :] for i in range(num_classes )]
            return bbox_results, mask_results
        
    
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
        tensor=tensor[0:1]
        if len(tensor.shape)==4:
            image = tensor.permute(0,2, 3,1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2,0).cpu().clone().detach().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        return img
    