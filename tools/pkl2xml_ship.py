import cv2
import mmcv
import numpy as np
import os
import cv2
from mmdet.datasets.dota_k import DotaKDataset
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv import Config, DictAction

configs='/media/zf/E/Dataset/2021ZKXT_aug_2/dardet_r50_DCN_fpn_2x_class10.py'
pkl_path='/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_DCN_2x_class10/test/test_class10_epoch13.pkl'
work_dir = '/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_DCN_2x_class10/test'

# configs='/media/zf/E/Dataset/2021ZKXT_aug_2/dardet_r50_fpn_2x.py'
# pkl_path='/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_2x/test_ISPRS/result_trainval_epoch12.pkl'
# work_dir = '/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_2x'



gt_dir = '/media/zf/E/Dataset/2021ZKXT_aug_2/annotations/class_10/GT_class10'


def pkl2xml(configs,pkl_path,work_dir,gt_dir):
    # dataset=DotaKDataset
    cfg = Config.fromfile(configs)
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    results=mmcv.load(pkl_path)
    dataset.evaluate_rbox(results, work_dir, gt_dir)
    
pkl2xml(configs,pkl_path,work_dir,gt_dir)