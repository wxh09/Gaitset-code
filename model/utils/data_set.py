import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr
import re
import json

def img2xarray(dataset, flie_path, frame_set=None):
	# print(flie_path)
	if dataset == 'CASIA':
		imgs = sorted(list(os.listdir(flie_path)))
		frame_list = [np.reshape(
			cv2.resize(cv2.imread(osp.join(flie_path, _img_path)), (128,128)),
			[self.resolution, self.resolution, -1])[:, :, 0] / 255.0
					  for _img_path in imgs
					  if osp.isfile(osp.join(flie_path, _img_path))]
		num_list = [(int)(re.split(r'[-.]',_img_path)[-2]) for _img_path in imgs] #pose
		# id_list = list(range(len(frame_list)))
		data_dict = xr.DataArray(
			frame_list,
			coords={'frame': num_list},
			dims=['frame', 'img_y', 'img_x'],
		)
		return data_dict

	elif dataset == 'OUMVLP':
		imgs = [str(frame).zfill(4) + '.png' for frame in frame_set]

		frame_list = [cv2.resize(
			cv2.imread(osp.join(flie_path, _img_path)), (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)[:, :, 0] / 255.0
					  for _img_path in imgs
					  if osp.isfile(osp.join(flie_path, _img_path))]
		if len(frame_list) < 20:
			# print(flie_path)
			# print(len(frame_list))
			frame_list.extend([np.zeros((96,128)) for i in range(len(frame_list),20)])
		return frame_list

def pose2array(dataset, path, frame_set=None):
	if dataset == 'CASIA':
		popaths = [re.split(r'/',path)[-1] + '.' + str(frame).zfill(4) + '.txt' for frame in frame_set]
		# print(popaths)

		pose_list = []
		for popath in popaths:
			popath = osp.join(path, popath)
			tarray=np.zeros((28))
			if os.path.exists(popath):
				tarray = np.loadtxt(popath) 
			all_pose=tarray.reshape(14,2)
			pose=np.zeros((5,2))
			pose[0]=all_pose[0]-[15,0]
			pose[1]=all_pose[3]-[15,0]
			pose[2]=all_pose[6]-[15,0]
			pose[3]=(all_pose[8]+all_pose[9])/2-[15,0]
			pose[4]=(all_pose[11]+all_pose[12])/2-[15,0]
			pose_list.append(pose)
		return pose_list

	elif dataset == 'OUMVLP':
		pose_list=[]
		num_list=[]
		poses = sorted(list(os.listdir(path)))

		for pose_json in poses:
			jsonFile = osp.join(path, pose_json)
			with open(jsonFile,'r') as f:
				json_dict = json.load(f)
				if len(json_dict['people']) == 0:
					continue
				keypoints = json_dict['people'][0]['pose_keypoints_2d']
				# print(keypoints)
				keypoints = np.reshape(np.array(keypoints), [18,3])[:,:2] / 10.0
				# print(keypoints)

			pose_list.append(keypoints)
			num_list.append(int(pose_json.split('_')[0]))

		if len(pose_list) < 20:
			# print("bu quan poses")
			# print(len(pose_list))
			pose_list.extend([np.zeros((18,2)) for i in range(len(pose_list),20)])
			num_list.extend([0 for i in range(len(num_list),20)])
			# print(len(pose_list))

		
		data_dict = xr.DataArray(
			pose_list,
			coords={'frame': num_list},
			dims=['frame', 'img_y', 'img_x'],
		)
		return data_dict	
	
class DataSet(tordata.Dataset):
	def __init__(self, seq_dir, pose_dir, label, seq_type, view, cache, resolution):
		self.seq_dir = seq_dir
		self.pose_dir = pose_dir #增加pose
		self.view = view
		self.seq_type = seq_type
		self.label = label
		self.cache = cache
		self.resolution = int(resolution)
		self.cut_padding = int(float(resolution)/64*10)
		self.data_size = len(self.label)
		self.data = [None] * self.data_size
		self.pose_data = [None] * self.data_size #增加pose
		self.frame_set = [None] * self.data_size

		self.label_set = set(self.label)
		self.seq_type_set = set(self.seq_type)
		self.view_set = set(self.view)
		_ = np.zeros((len(self.label_set),
					  len(self.seq_type_set),
					  len(self.view_set))).astype('int')
		_ -= 1
		self.index_dict = xr.DataArray(
			_,
			coords={'label': sorted(list(self.label_set)),
					'seq_type': sorted(list(self.seq_type_set)),
					'view': sorted(list(self.view_set))},
			dims=['label', 'seq_type', 'view'])

		for i in range(self.data_size):
			_label = self.label[i]
			_seq_type = self.seq_type[i]
			_view = self.view[i]
			self.index_dict.loc[_label, _seq_type, _view] = i

	def load_all_data(self):
		print(self.data_size)
		for i in range(self.data_size):
			print(i)
			self.load_data(i)

	def load_data(self, index):
		return self.__getitem__(index)

	# def __loader__(self, path):
	# 	return img2xarray(
	# 		path)[:, :, self.cut_padding:-self.cut_padding].astype(
	# 		'float32') / 255.0
			
	# def pose_loader(self, path, frame_set):
	# 	return pose2array(path,frame_set)

	def __getitem__(self, index):
		# pose sequence sampling
		if not self.cache:
			pose_data = [pose2array('OUMVLP',_path) for _path in self.pose_dir[index]] #增加pose		
			frame_set = [set(feature.coords['frame'].values.tolist()) for feature in pose_data]
			frame_set = list(set.intersection(*frame_set))			#真实帧数		
			data = [img2xarray('OUMVLP',_path,frame_set) for _path in self.seq_dir[index]]

		elif self.data[index] is None:
			pose_data = [pose2array('OUMVLP',_path) for _path in self.pose_dir[index]] #增加pose	
			frame_set = [set(feature.coords['frame'].values.tolist()) for feature in pose_data]
			frame_set = list(set.intersection(*frame_set))			#真实帧数		
			data = [img2xarray('OUMVLP',_path,frame_set) for _path in self.seq_dir[index]]
			# print(data)

			self.data[index] = data
			self.pose_data[index] = pose_data
			self.frame_set[index] = frame_set
		else:
			data = self.data[index]
			pose_data = self.pose_data[index]
			frame_set = self.frame_set[index]
			
		return data, frame_set, self.view[
			index], self.seq_type[index], self.label[index], pose_data

	'''
	def img2xarray(self, flie_path):
		# print(flie_path)
		imgs = sorted(list(os.listdir(flie_path)))
		frame_list = [np.reshape(
			cv2.resize(cv2.imread(osp.join(flie_path, _img_path)), (128,128)),
			[self.resolution, self.resolution, -1])[:, :, 0]
					  for _img_path in imgs
					  if osp.isfile(osp.join(flie_path, _img_path))]
		# print(frame_list[0].shape)
		num_list = [(int)(re.split(r'[-.]',_img_path)[-2]) for _img_path in imgs] #pose
		# id_list = list(range(len(frame_list)))
		# print(num_list)
		data_dict = xr.DataArray(
			frame_list,
			coords={'frame': num_list},
			dims=['frame', 'img_y', 'img_x'],
		)
		# print(data_dict)
		return data_dict

	def pose2array(self, path, frame_set):
		# print(path)
		popaths = [re.split(r'/',path)[-1] + '.' + str(frame).zfill(4) + '.txt' for frame in frame_set]
		# print(popaths)

		pose_list = []
		for popath in popaths:
			popath = osp.join(path, popath)
			tarray=np.zeros((28))
			if os.path.exists(popath):
				tarray = np.loadtxt(popath) 
			all_pose=tarray.reshape(14,2)
			pose=np.zeros((5,2))
			pose[0]=all_pose[0]-[15,0]
			pose[1]=all_pose[3]-[15,0]
			pose[2]=all_pose[6]-[15,0]
			pose[3]=(all_pose[8]+all_pose[9])/2-[15,0]
			pose[4]=(all_pose[11]+all_pose[12])/2-[15,0]
			pose_list.append(pose)
		return pose_list
	'''
	def __len__(self):
		return len(self.label)
