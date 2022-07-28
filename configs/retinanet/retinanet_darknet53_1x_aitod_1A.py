_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
fp16 = dict(loss_scale=512.)
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    #  backbone=dict(
    #     type='DarknetV3',
    #     layers=[1, 2, 8, 8, 4],
    #     inplanes=[3, 32, 64, 128, 256, 512],
    #     planes=[32, 64, 128, 256, 512, 1024],
    #     norm_cfg=dict(type='BN'),
    #     out_indices=(1, 2, 3, 4),
    #     frozen_stages=0,
    #     norm_eval=False),
    # backbone=dict(
    #     type='Darknet',
    #     depth=53,
    #     out_indices=(1,2, 3, 4),
    #     init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=128,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=1),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=8,
        in_channels=128,
        stacked_convs=4,
        feat_channels=256,
        freeze_backbone=False,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,#4
            scales_per_octave=1,#3
            ratios=[1.0],#0.5, 1.0, 2.0
            strides=[4]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# dataset settings
dataset_type = 'AiTodDataset'
data_root = '/media/zf/E/Dataset/AI-TOD/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=3, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 24])  # the real step is [18*5, 24*5]
runner = dict(max_epochs=36)  # the real epoch is 28*5=140
work_dir = '/media/zf/E/mmdetection219/work_dir/retinanet_darknet53_3x_aitod_1A'
load_from ='/media/zf/E/checkpoint/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'
resume_from =None#  '/media/zf/E/mmdetection219/work_dirs/retinanet_r50_fpn_1x_aitod_1A/latest.pth'
workflow = [('train', 1)]
