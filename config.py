conf = {
    "type": "silh",
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "data": {
        'train_list': "/home1/wxh/Gaitset-code/list/train.txt",
        'test_list': "/home1/wxh/Gaitset-code/list/test.txt",
        'silhdata_dir': "/home1/wxh/dataset/OUMVLP_silhouette_normalization/",
        'ofdata_dir': "",
		'posedata_dir': "/home1/wxh/dataset/OUMVLP_pose_normalization/",
        # 'resolution': '128',
        'dataset': 'OUMVLP',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'train_pid': 5153,   #personid for training
        'test_pid': 5154,   #personid for training
        'frame_num': 20,
        # 'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (16, 16),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 20,
        'model_name': 'set',
    },
}

cbconf = {
    "type": "silh",
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'train_list': "/home1/wxh/Gaitset-code/list/cb_train74.txt",
        'test_list': "/home1/wxh/Gaitset-code/list/cb_test50.txt",
        'silhdata_dir': "/home/wxh/dataset/silhouette_normalization/",
        'ofdata_dir': "",
        'posedata_dir': "/home1/wxh/dataset/contour_bbx_images_all_part_pose/",
        # 'resolution': '128',
        'dataset': 'CASIA',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'train_pid': 74,   #personid for training
        'test_pid': 50,   #personid for training
        'frame_num': 30,
        # 'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 30,
        'model_name': 'cbset',
    },
}

cbofconf = {
    "type": "silh",
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "data": {
        'train_list': "/home1/wxh/Gaitset-code/list/cb_train74.txt",
        'test_list': "/home1/wxh/Gaitset-code/list/cb_test50.txt",
        # 'silhdata_dir': "/home/wxh/dataset/silhouette_normalization/",
		'silhdata_dir': "",
        # 'ofdata_dir': "/home/wxh/dataset/CASIAB_OF_normalization/",
        'ofdata_dir': "/home/wxh/dataset/CASIAB_OF_silh_normalization/",
        'posedata_dir': "/home1/wxh/dataset/contour_bbx_images_all_part_pose/",
        # 'resolution': '128',
        'dataset': 'CASIA',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'train_pid': 74,   #personid for training
        'test_pid': 50,   #personid for training
        'frame_num': 32,
        # 'pid_shuffle': False, 
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2, 
        'num_workers': 4,
        'frame_num': 30,
        'model_name': 'cbof3set',
    }, 
}

poseconf = {
    "type": "pose",
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "4",
    "data": {
        'train_list': "/home1/wxh/Gaitset-code/list/train.txt",
        'test_list': "/home1/wxh/Gaitset-code/list/test.txt",
        'silhdata_dir': "",
		'ofdata_dir': "",
        # 'posedata_dir': "/home1/anweizhi/data/openpose_raw/OUMVLP4/",
		'posedata_dir': "/home1/wxh/dataset/OUMVLP_pose_normalization/",
        # 'resolution': '128',
        'dataset': 'OUMVLP',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'train_pid': 5153,   #personid for training
        'test_pid': 5154,   #personid for training
        'frame_num': 20,
        # 'pid_shuffle': False, 
    },
    "model": {
        'hidden_dim': 512,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (16, 20),
        'restore_iter': 0,
        'total_iter': 150000,	#~500iter/epoch
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 20,
        'model_name': 'posepair',
    },
}
