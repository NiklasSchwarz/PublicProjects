"""
Hadamard SegFormer Configuration for Cityscapes Dataset
This configuration file is for training a SegFormer model with Hadamard codes on the Cityscapes dataset.
It includes custom Hadamard-specific components and configurations for training and evaluation.

This file is part of the Hadamard experiments and is designed to work with the mmsegmentation framework.

The first part of the file contains the necessary imports and configurations
for the Hadamard-specific components, followed by the model, data, and training settings.
"""

NUM_CLASSES         = 19                                                        # Number of classes in Cityscapes dataset
OUTPUT_CHANNELS     = 32                                                        # Hadamard output channels (size of Hadamard codes)
HADAMARD_SIZE       = (NUM_CLASSES, OUTPUT_CHANNELS)                            # Size of Hadamard codes
AUGMENTED           = False
USE_SIMPLEX         = False                                                      # Whether to use augmented Hadamard codes
ACCOMULATIVE_COUNTS = 2                                                         # Number of iterations to accumulate gradients

hadamard_codec = dict(
    type="HadamardCodec",
    hadamard_size=HADAMARD_SIZE,
    use_simplex=USE_SIMPLEX,
    activation_function="tanh",
    use_all_one_codeword=False,
    augmented=AUGMENTED
)

CUSTOM_IMPORT = dict(
    imports=[ 
        'models.encoder_decoder_hadamard', 
        'decode_heads.segformer_head_hadamard',
    ],
    allow_failed_imports=False
)

CROP_SIZE = (1024, 1024)                                                # Crop size for training and validation
STRIDE = (768, 768)                                                     # Stride for sliding window inference

"""
Base configuration files for SegFormer with Hadamard codes
These files include the model architecture, dataset settings, runtime configurations, and training schedules.

"""
_base_ = [
    'mmseg::_base_/models/segformer_mit-b0.py',
    'mmseg::_base_/datasets/cityscapes_1024x1024.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]


CHECKPOINT          = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'
MODEL_IN_CHANNELS   = [64, 128, 320, 512]
MODEL_CHANNELS      = 256
EMBED_DIMS          = 64
NUM_LAYERS          = [2, 2, 2, 2]


"""
Optimizer and learning rate scheduler configurations
These settings define the optimizer type, learning rate, and parameter-wise configurations for training.
"""
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }),
    accumulative_counts=ACCOMULATIVE_COUNTS
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500*ACCOMULATIVE_COUNTS),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500*ACCOMULATIVE_COUNTS,
        end=160000*ACCOMULATIVE_COUNTS,
        by_epoch=False,
    )
]

"""
Data loading and preprocessing configurations
These settings define the data loading pipelines for training and validation,
including image loading, resizing, cropping, flipping, and Hadamard-specific transformations.
"""
train_pipeline = [
    dict(type='LoadImageFromFile'), 
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=CROP_SIZE, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    #dict(type='HadamardCodeTransform', hadamard_codec=hadamard_codec),    # Custom Hadamard transformation
    #dict(type='PackSegInputsHadamard')                                  # Custom packing for Hadamard codes
    dict(type='PackSegInputs')                                          # Standard packing (no Hadamard for test)

]

test_pipeline = [
    dict(type='LoadImageFromFile'),                                     # Load test images
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),           # Resize to fixed scale
    dict(type='LoadAnnotations'),                                       # Load ground truth
    dict(type='PackSegInputs')                                          # Standard packing (no Hadamard for test)
]

"""
Data loaders for training, validation, and testing
These settings define the batch size, number of workers, and data samplers for each phase.
The training data loader uses an infinite sampler for continuous training,
while the validation and test data loaders use a default sampler for evaluation.
"""
train_dataloader = dict(
    batch_size=4,
    num_workers=2,                     # Tuning siehe unten
    persistent_workers=True,           # Worker bleiben zwischen Iterationen aktiv
    pin_memory=True,                   # Page-locked Host RAM -> schnellere H2D Copies
    prefetch_factor=4,                 # pro Worker vorgepufferte Batches (Default=2)
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        ignore_index=255,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

"""
Data preprocessor for image normalization and resizing
This preprocessor applies mean and standard deviation normalization,
converts BGR to RGB, and resizes images to the specified crop size.
"""
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=CROP_SIZE,
    pad_val=0,
    seg_pad_val=255
)

"""
Model configuration for SegFormer with Hadamard codes
This section defines the model architecture, including the backbone, decode head,
and training/testing configurations.
The decode head is specifically designed for Hadamard codes, using Hadamard-specific loss functions
for training.
"""
model = dict(
    type='EncoderDecoderHadamard',
    data_preprocessor=data_preprocessor,
    hadamard_codec=hadamard_codec,                                       # Hadamard codes size
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=CHECKPOINT),
        with_cp=False,
        embed_dims=EMBED_DIMS,
        num_layers=NUM_LAYERS
    ),
    decode_head=dict(
        type='SegformerHeadHadamard',
        in_channels=MODEL_IN_CHANNELS,
        in_index=[0, 1, 2, 3],
        channels=MODEL_CHANNELS,
        dropout_ratio=0.1,
        hadamard_codec=hadamard_codec,                                    # Hadamard codes size
        hadamard_size=HADAMARD_SIZE,
        num_classes=OUTPUT_CHANNELS,                                      # Hadamard output channels
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            #  dict(
            #      type='CrossEntropyLoss', 
            #      loss_name='hadamard_bce_codes_loss', 
            #      use_sigmoid=True,
            #      loss_weight=0.25),
            dict(
                type='HadamardCrossEntropyLoss',
                loss_name='hadamard_ce_loss',
                loss_weight=1.0,
                ignore_index=255,
                use_simplex=USE_SIMPLEX),
            #  dict(
            #      type='HadamardMSELoss',
            #      loss_name='hadamard_mse_loss',
            #      loss_weight=0.1,
            #      ignore_index=255),
            # dict(
            #     type='HadamardL1Loss',
            #     loss_name='hadamard_l1_loss',
            #     loss_weight=0.1,
            #     ignore_index=255),
            # dict(
            #     type='HadamardCodesMSELoss',
            #     loss_name='hadamard_mse_codes_loss',
            #     loss_weight=0.25,
            #     ignore_index=255),
            # dict(
            #      type='HadamardCodesL1Loss',
            #      loss_name='hadamard_l1_codes_loss',
            #      loss_weight=0.25,
            #      ignore_index=255),
        ]
    ),
    train_cfg=dict(),                                                   # Training configuration (empty for default)
    test_cfg=dict(mode='slide', crop_size=CROP_SIZE, stride=STRIDE)     # Sliding window inference settings
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000*ACCOMULATIVE_COUNTS, val_interval=16000*ACCOMULATIVE_COUNTS
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50*ACCOMULATIVE_COUNTS, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000*ACCOMULATIVE_COUNTS),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)