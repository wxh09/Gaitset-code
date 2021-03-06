import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, posedata_path=None, cache=True):
	seq_dir = list()
	view = list()
	seq_type = list()
	label = list()
	if posedata_path is not None:
		pose_dir = list() 

	for _label in sorted(list(os.listdir(dataset_path))):
		# In CASIA-B, data of subject #5 is incomplete.
		# Thus, we ignore it in training.
		if dataset == 'CASIA-B' and _label == '005':
			continue
		label_path = osp.join(dataset_path, _label)
		for _seq_type in sorted(list(os.listdir(label_path))):
			seq_type_path = osp.join(label_path, _seq_type)
			for _view in sorted(list(os.listdir(seq_type_path))):
				_seq_dir = osp.join(seq_type_path, _view)
				seqs = os.listdir(_seq_dir)
				if len(seqs) > 0:
					seq_dir.append([_seq_dir])
					# print(_seq_dir)
					# print(seq_dir)
					if posedata_path is not None:	
						_pose_dir = _label + '-' + _seq_type + '-' + _view					#拼接字符串_pose_dir
						_pose_dir =  osp.join(posedata_path,_pose_dir)
						pose_dir.append([_pose_dir])
						# print(_pose_dir)
						# print(pose_dir)
					label.append(_label)
					seq_type.append(_seq_type)
					view.append(_view)

	pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
		dataset, pid_num, pid_shuffle))
	if not osp.exists(pid_fname):
		pid_list = sorted(list(set(label)))
		if pid_shuffle:
			np.random.shuffle(pid_list)
		pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
		os.makedirs('partition', exist_ok=True)
		np.save(pid_fname, pid_list)

	pid_list = np.load(pid_fname)
	train_list = pid_list[0]
	test_list = pid_list[1]
	train_source = DataSet(
		[seq_dir[i] for i, l in enumerate(label) if l in train_list],
		[pose_dir[i] for i, l in enumerate(label) if l in train_list], #增加pose目录
		[label[i] for i, l in enumerate(label) if l in train_list],
		[seq_type[i] for i, l in enumerate(label) if l in train_list],
		[view[i] for i, l in enumerate(label)
		 if l in train_list],
		cache, resolution)
	test_source = DataSet(
		[seq_dir[i] for i, l in enumerate(label) if l in test_list],
		[pose_dir[i] for i, l in enumerate(label) if l in train_list], #增加pose目录
		[label[i] for i, l in enumerate(label) if l in test_list],
		[seq_type[i] for i, l in enumerate(label) if l in test_list],
		[view[i] for i, l in enumerate(label)
		 if l in test_list],
		cache, resolution)

	return train_source, test_source

def load_OU_data(dataset_path, resolution, pid_num, pid_shuffle, posedata_path, dataset='OU', cache=True):
	seq_dir = list()
	view = list()
	seq_type = list()
	label = list()
	pose_dir = list() 

	for _label in sorted(list(os.listdir(posedata_path))):
		label_path = osp.join(posedata_path, _label, "RGB")
		for _seq_type in sorted(list(os.listdir(label_path))):
			_seq_dir = osp.join(label_path, _seq_type)
			_view = _seq_type.split("_")[0]

			seqs = os.listdir(_seq_dir)
			if len(seqs) > 0:
				pose_dir.append([_seq_dir])
				_silh_dir =  osp.join(dataset_path, 'Silhouette_' + _view + '-' + _seq_type.split("_")[1], _label)		#拼接字符串_silh_dir
				seq_dir.append([_silh_dir])
				label.append(_label)
				seq_type.append(_seq_type)
				view.append(_view)

	pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
		dataset, pid_num, pid_shuffle))
	if not osp.exists(pid_fname):
		pid_list = sorted(list(set(label)))
		if pid_shuffle:
			np.random.shuffle(pid_list)
		pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
		os.makedirs('partition', exist_ok=True)
		np.save(pid_fname, pid_list)

	pid_list = np.load(pid_fname, allow_pickle=True)
	train_list = pid_list[0]
	test_list = pid_list[1]
	train_source = DataSet(
		[seq_dir[i] for i, l in enumerate(label) if l in train_list],
		[pose_dir[i] for i, l in enumerate(label) if l in train_list], #增加pose目录
		[label[i] for i, l in enumerate(label) if l in train_list],
		[seq_type[i] for i, l in enumerate(label) if l in train_list],
		[view[i] for i, l in enumerate(label) if l in train_list],
		cache, resolution)
	test_source = DataSet(
		[seq_dir[i] for i, l in enumerate(label) if l in test_list],
		[pose_dir[i] for i, l in enumerate(label) if l in train_list], #增加pose目录
		[label[i] for i, l in enumerate(label) if l in test_list],
		[seq_type[i] for i, l in enumerate(label) if l in test_list],
		[view[i] for i, l in enumerate(label) if l in test_list],
		cache, resolution)

	return train_source, test_source
