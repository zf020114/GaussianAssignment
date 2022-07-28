# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from .pipelines import Compose
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
import cv2
from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset
from mmdet.core import multi_apply
import torch
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module()
class iSaidContourDataset(CocoDataset):
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 max_points=1440):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.max_points=max_points
        
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)
    
    CLASSES = ( 'ship',
        'storage_tank',
        'baseball_diamond',
        'tennis_court',
        'basketball_court',
        'Ground_Track_Field',
        'Bridge',
        'Large_Vehicle',
        'Small_Vehicle',
        'Helicopter',
        'Swimming_pool',
        'Roundabout',
        'Soccer_ball_field',
        'plane',
        'Harbor')
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        data=self.pipeline(results)
        # img = data['img'].data
        # pad_shape=img.shape[1:3]
        # gt_labels = data['gt_labels'].data
        gt_masks=data['gt_masks'].data
        gt_bboxes=data['gt_bboxes'].data
        self.device=gt_bboxes.device
        _, _ ,contours =self.gt_mask2bbox_torch(gt_masks)
        # data['gt_bboxes'] = contours
        data['gt_contours'] = DC(contours)
        return data
    
    def gt_mask2bbox_torch(self,mask_img):
        # H,W =self.img_shape#计算单张图的mask2box 和 mask
        #将gtmask转换为tensor
        # masks = mask_img.expand(H, W, 0, 0)
        # masks = masks.to_tensor(dtype=torch.float16, device=self.device)
        # masks=torch.nn.functional.interpolate(masks[None,:,...], scale_factor=1/self.out_stride, mode='bilinear')[0]
        masks = None
        mask_img=mask_img.masks
        mask2box, contour_fix = multi_apply(self.mask2bbox,mask_img)
        mask2box = torch.tensor(np.array(mask2box),dtype=torch.float32, device=self.device)[:,0,:]
        contour = torch.tensor(np.array(contour_fix),dtype=torch.float32, device=self.device)[:,0,:]
        return (mask2box,masks,contour)

    def mask2bbox(self,mask):   
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_np=np.zeros((0,2))
        for i in contour:
            contour_np=np.vstack((contour_np,i[:,0]))
        if contour_np.shape[0]>0:
            left, right ,top, down = contour_np.min(), contour_np.max(), contour_np.min(), contour_np.max()
            contour_fix = cv2.resize(contour_np,(2,self.max_points))[None,:]
        else:
            left, right ,top, down=0,0,0,0
            contour_fix = np.zeros((self.max_points,2))[None,:]
        _mask2box=np.array([left, top, right,  down])[None,:]
           
        return (_mask2box,contour_fix)