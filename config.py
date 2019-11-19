conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "8,9",
    "data": {
        'dataset_path': "/home1/lrj/moo/dataset/GaitSet_dataset",
		'posedata_path': "/home1/wxh/dataset/contour_bbx_images_all_part_pose/",
        'resolution': '128',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 5,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 272,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 16),
        'restore_iter': 0,
        'total_iter': 100,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 32,
        'model_name': 'rg',
    },
}
