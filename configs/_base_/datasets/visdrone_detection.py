# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/home/ubuntu/visdrone/data/'
data_root = '/home/stu010/datasets/VisDrone/VisDrone2019_DET/'
classes = (
 "pedestrian",
 "people",
 "bicycle",
 "car",
 "van",
 "truck",
 "tricycle",
 "awning-tricycle",
 "bus",
 "motor",
)
img_norm_cfg = dict(
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # ref from: https://blog.csdn.net/mary_0830/article/details/103361664  *255
    mean = [96.56190315, 98.26577235, 94.6931076], std = [48.83620515, 46.656687 , 49.58866425], to_rgb = True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 800), (1333, 400)], keep_ratio=True),
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
        img_scale=(1333, 800),
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        #ann_file=data_root + 'annotations/train2019.json',
        #img_prefix=data_root + 'images/train2019/',
        ann_file=data_root + 'COCO_version/annotations/VisDrone2019-DET_train_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'annotations/val2019.json',
        # img_prefix=data_root + 'images/val2019/',
        ann_file=data_root + 'COCO_version/annotations/VisDrone2019-DET_val_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'annotations/testdev2019.json',
        # img_prefix=data_root + 'images/testdev2019/',
        ann_file=data_root + 'COCO_version/annotations/VisDrone2019-DET_test-dev_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-test-dev/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')