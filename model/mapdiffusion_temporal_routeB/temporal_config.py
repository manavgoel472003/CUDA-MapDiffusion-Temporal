checkpoint_config = dict(interval=3480, create_symlink=False)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss'
load_from = None
resume_from = None
workflow = [('train', 1)]
type = 'Mapper'
plugin = True
plugin_dir = 'plugin/'
custom_imports = dict(
    imports=[
        'plugin.models.mapers.MapDiffusionTemporal',
        'plugin.models.heads.MapDetectorHeadDiffuseTemporal'
    ],
    allow_failed_imports=False)
data_root = '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_h = 480
img_w = 800
img_size = (480, 800)
num_gpus = 1
batch_size = 8
num_iters_per_epoch = 3480
num_epochs = 32
temporal_start_epoch = 1
num_epochs_single_frame = 1
temporal_start_iter = 3480
total_iters = 111360
num_queries = 100
scheduler = 'cosine'
total_steps = 1000
cat2id = dict(ped_crossing=0, divider=1, boundary=2)
num_class = 3
roi_size = (60, 30)
bev_h = 50
bev_w = 100
pc_range = [-30.0, -15.0, -3, 30.0, 15.0, 5]
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_points = 20
permute = True
model = dict(
    type='MapDiffusionTemporal',
    roi_size=(60, 30),
    bev_h=50,
    bev_w=100,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=(60, 30),
        bev_h=50,
        bev_w=100,
        use_grid_mask=True,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d'),
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs=True,
            num_outs=3,
            norm_cfg=dict(type='BN2d'),
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=[-30.0, -15.0, -3, 30.0, 15.0, 5],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=3),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=50,
            col_num_embed=100)),
    head_cfg=dict(
        type='MapDetectorHeadDiffuseTemporal',
        num_queries=100,
        embed_dims=512,
        num_classes=3,
        in_channels=256,
        num_points=20,
        roi_size=(60, 30),
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        streaming_cfg=None,
        use_temporal_query_fusion=True,
        temporal_query_fusion_cfg=dict(
            embed_dims=512, num_heads=8, dropout=0.1, use_ffn=True),
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=20,
            coord_dim=2,
            encoder=dict(type='PlaceHolderEncoder', embed_dims=512),
            decoder=dict(
                type='MapTransformerDecoderDiffuse',
                num_layers=6,
                timestep_embed=512,
                prop_add_stage=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=512,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=512,
                            num_heads=8,
                            num_levels=1,
                            num_points=20,
                            dropout=0.1)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=512,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0),
        loss_reg=dict(type='LinesL1Loss', loss_weight=50.0, beta=0.01),
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=5.0),
                reg_cost=dict(
                    type='LinesL1Cost', weight=50.0, beta=0.01,
                    permute=True)))),
    streaming_cfg=dict(
        streaming_bev=True,
        batch_size=8,
        fusion_cfg=dict(type='ConvGRU', out_channels=256)),
    temporal_query_cfg=dict(
        enable=True,
        train=True,
        test=True,
        batch_size=8,
        warmup_iters=3480,
        temporal_consistency_loss=True,
        temporal_consistency_weight=0.5,
        temporal_score_thr=0.25,
        same_class_match=True,
        strict_temporal_sequence=True,
        expected_sample_delta=1,
        debug_assert_temporal_delta=False),
    model_name='MapDiffusion_Temporal_Scratch_32e_Start1')
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=2,
        roi_size=(60, 30),
        sample_num=20,
        normalize=True,
        permute=True),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='ResizeMultiViewImages', size=(480, 800), change_intrinsics=True),
    dict(
        type='Normalize3D',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(
        type='Collect3D',
        keys=['img', 'vectors', 'gts'],
        meta_keys=('token', 'ego2img', 'sample_idx', 'ego2global_translation',
                   'ego2global_rotation', 'img_shape', 'scene_name'))
]
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(
        type='ResizeMultiViewImages', size=(480, 800), change_intrinsics=True),
    dict(
        type='Normalize3D',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(
        type='Collect3D',
        keys=['img'],
        meta_keys=('token', 'ego2img', 'sample_idx', 'ego2global_translation',
                   'ego2global_rotation', 'img_shape', 'scene_name'))
]
eval_config = dict(
    type='NuscDataset',
    data_root=
    '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/',
    ann_file=
    '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/nuscenes_map_infos_val_newsplit.pkl',
    meta=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False,
        output_format='vector'),
    roi_size=(60, 30),
    cat2id=dict(ped_crossing=0, divider=1, boundary=2),
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=2,
            simplify=True,
            normalize=False,
            roi_size=(60, 30)),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
    ],
    interval=1)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='NuscDataset',
        data_root=
        '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/',
        ann_file=
        '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/nuscenes_map_infos_train_newsplit.pkl',
        meta=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            output_format='vector'),
        roi_size=(60, 30),
        cat2id=dict(ped_crossing=0, divider=1, boundary=2),
        pipeline=[
            dict(
                type='VectorizeMap',
                coords_dim=2,
                roi_size=(60, 30),
                sample_num=20,
                normalize=True,
                permute=True),
            dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='ResizeMultiViewImages',
                size=(480, 800),
                change_intrinsics=True),
            dict(
                type='Normalize3D',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImages', size_divisor=32),
            dict(type='FormatBundleMap'),
            dict(
                type='Collect3D',
                keys=['img', 'vectors', 'gts'],
                meta_keys=('token', 'ego2img', 'sample_idx',
                           'ego2global_translation', 'ego2global_rotation',
                           'img_shape', 'scene_name'))
        ],
        seq_split_num=1),
    val=dict(
        type='NuscDataset',
        data_root=
        '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/',
        ann_file=
        '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/nuscenes_map_infos_val_newsplit.pkl',
        meta=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            output_format='vector'),
        roi_size=(60, 30),
        cat2id=dict(ped_crossing=0, divider=1, boundary=2),
        pipeline=[
            dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
            dict(
                type='ResizeMultiViewImages',
                size=(480, 800),
                change_intrinsics=True),
            dict(
                type='Normalize3D',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImages', size_divisor=32),
            dict(type='FormatBundleMap'),
            dict(
                type='Collect3D',
                keys=['img'],
                meta_keys=('token', 'ego2img', 'sample_idx',
                           'ego2global_translation', 'ego2global_rotation',
                           'img_shape', 'scene_name'))
        ],
        eval_config=dict(
            type='NuscDataset',
            data_root=
            '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/',
            ann_file=
            '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/nuscenes_map_infos_val_newsplit.pkl',
            meta=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False,
                output_format='vector'),
            roi_size=(60, 30),
            cat2id=dict(ped_crossing=0, divider=1, boundary=2),
            pipeline=[
                dict(
                    type='VectorizeMap',
                    coords_dim=2,
                    simplify=True,
                    normalize=False,
                    roi_size=(60, 30)),
                dict(type='FormatBundleMap'),
                dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
            ],
            interval=1),
        test_mode=True,
        seq_split_num=1),
    test=dict(
        type='NuscDataset',
        data_root=
        '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/',
        ann_file=
        '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/nuscenes_map_infos_val_newsplit.pkl',
        meta=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            output_format='vector'),
        roi_size=(60, 30),
        cat2id=dict(ped_crossing=0, divider=1, boundary=2),
        pipeline=[
            dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
            dict(
                type='ResizeMultiViewImages',
                size=(480, 800),
                change_intrinsics=True),
            dict(
                type='Normalize3D',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImages', size_divisor=32),
            dict(type='FormatBundleMap'),
            dict(
                type='Collect3D',
                keys=['img'],
                meta_keys=('token', 'ego2img', 'sample_idx',
                           'ego2global_translation', 'ego2global_rotation',
                           'img_shape', 'scene_name'))
        ],
        eval_config=dict(
            type='NuscDataset',
            data_root=
            '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/',
            ann_file=
            '/home/018198687/Mapping/mmdetection3d/datasets/nuScenes/v1.0-trainval/nuscenes_map_infos_val_newsplit.pkl',
            meta=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False,
                output_format='vector'),
            roi_size=(60, 30),
            cat2id=dict(ped_crossing=0, divider=1, boundary=2),
            pipeline=[
                dict(
                    type='VectorizeMap',
                    coords_dim=2,
                    simplify=True,
                    normalize=False,
                    roi_size=(60, 30)),
                dict(type='FormatBundleMap'),
                dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
            ],
            interval=1),
        test_mode=True,
        seq_split_num=1),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        num_iters_to_seq=3480,
        random_drop=0.0),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.1),
            temporal_query_fusion=dict(lr_mult=3.0))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.003)
evaluation = dict(
    interval=3480,
    eval_diffusion_eta=0.5,
    eval_diffusion_sampling_timesteps=5,
    eval_diffusion_query_threshold=0.5)
find_unused_parameters = True
runner = dict(type='IterBasedRunner', max_iters=111360)
SyncBN = False
gpu_ids = range(0, 1)
