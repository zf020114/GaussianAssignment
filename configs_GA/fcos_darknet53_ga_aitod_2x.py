# model settings
fp16 = dict(loss_scale=512.)
model = dict(
    type='TTFNet',
     backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=0,
        norm_eval=False),
    # backbone=dict(
    #     type='Darknet',
    #     depth=53,
    #     out_indices=(2,3, 4, 5),
    #     init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=None,
    bbox_head=dict( 
        type='TTF_Fcos_Head',
        inplanes=(128, 256, 512, 1024),
        planes=(512, 256, 128),
        down_ratio=4,
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=8,
        wh_offset_base=4,
        strides=[4],
        regress_ranges=[(-1, 1024)],
        wh_agnostic=True,
        wh_gaussian=True,
        debug=False,
        shortcut_cfg=(1, 2, 3),
        alpha=1.0,
        beta=1.0,
        pos_thr_cls=1e-2,
        freeze_backbone=False,
        dcn_on_last_conv=False,
        iou_branch=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),

    # training and testing settings
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
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=3,
        score_thr=0.01,
        nms=dict(type='nms', iou_thr=0.65),
        max_per_img=300))

# dataset settings 
dataset_type = 'AiTodDataset'
data_root = '/media/zf/E/Dataset/AI-TOD/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=False),
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
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=6e-2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_train2017.json',
    #     img_prefix=data_root + 'train2017/',
    #     pipeline=train_pipeline),
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
# optimizer
optimizer = dict(type='SGD', lr=0.0125, momentum=0.9, weight_decay=0.0004)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[24, 33])
checkpoint_config = dict(interval=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workdir/ttffcos53_2x_800_aitod_iou_ga'
load_from ='/media/zf/E/mmdetection219/workdir/ttfnet53_2x_800_aitod_iou/epoch_36.pth'
resume_from = None#'/media/zf/E/mmdetection219/workdir/ttffcos53_2x_800_aitod_iou_ga/latest.pth'
workflow = [('train', 1)]
evaluation = dict(interval = 3,  metric=['bbox'])