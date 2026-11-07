data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/nuscenes/gts"
input_shape = (704, 256)
batch_size = 1

dataset_coord_debug = dict(
    enabled=True,
    interval=50,
    max_print=12,
    show_matrices=False,
    show_label_hist=True,
)
occ_pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
visibility_pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 5.4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadPointFromFileLiDAR", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweepsLiDAR",
        sweeps_num=10,
        load_dim=5,
        use_dim=6,                
        visibility_use_dim=7,     
        pad_empty_sweeps=True,
        remove_close=True,
        add_decay=True,
        decay_lambda=1.0,
        decay_normalize=True,
        add_is_current=True,
        point_cloud_range=visibility_pc_range,
        crop_points=True,
        sort_sweeps_by_time=True,
        debug=False,
    ),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadMultiViewDepthFromFiles", is_to_depth_map=True, map_size=None),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="LoadOccupancyOcc3d",
        occ_path=occ_path,
        semantic=True,
        use_ego=True,
        use_occ3d_mask=True,
        pc_range=occ_pc_range,
        visibility_pc_range=visibility_pc_range,
        points_pc_range=visibility_pc_range,
        use_lidar=True,
        use_mask_training=True,
    ),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]

test_pipeline = [
    dict(type="LoadPointFromFileLiDAR", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweepsLiDAR",
        sweeps_num=10,
        load_dim=5,
        use_dim=6,                
        visibility_use_dim=7,     
        pad_empty_sweeps=True,
        remove_close=True,
        add_decay=True,
        decay_lambda=1.0,
        decay_normalize=True,
        add_is_current=True,
        point_cloud_range=visibility_pc_range,
        crop_points=True,
        sort_sweeps_by_time=True,
        debug=False,
    ),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadMultiViewDepthFromFiles", is_to_depth_map=True, map_size=None),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="LoadOccupancyOcc3d",
        occ_path=occ_path,
        semantic=True,
        use_ego=True,
        use_occ3d_mask=True,
        pc_range=occ_pc_range,
        visibility_pc_range=visibility_pc_range,
        points_pc_range=visibility_pc_range,
        use_lidar=True,
        use_mask_training=True,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]

data_aug_conf = dict(
    H=900,
    W=1600,
    final_dim=(256, 704),
    resize_lim=(0.44, 0.44),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    rand_flip=False,
)

train_dataset_config = dict(
    type="NuScenesOcc3DDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_gf3d_occ3d_train.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase="train",
    respect_valid_flag=False,
    coord_debug=dataset_coord_debug,
    sample_interval=14,
    sample_offset=0,
)

val_dataset_config = dict(
    type="NuScenesOcc3DDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_gf3d_occ3d_val.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase="val",
    respect_valid_flag=False,
    coord_debug=dict(enabled=True, interval=50, max_print=6, show_matrices=False, show_label_hist=True),
    sample_interval=1,
    sample_offset=0,
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2,
)
