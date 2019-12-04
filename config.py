conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'train_list': "/home/yagilab-1/Gaitset-code/list/train.txt",
        'test_list': "/home/yagilab-1/Gaitset-code/list/test.txt",
        'silhdata_dir': "/home/yagilab-1/data/OUMVLP/",
		'posedata_dir': "/home/yagilab-1/data/openpose_correct/OUMVLP4/",
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
        'hidden_dim': 272,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 4),
        'restore_iter': 19000,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 20,
        'model_name': 'rgp',
    },
}
