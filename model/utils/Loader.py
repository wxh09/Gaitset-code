# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
# from PIL import Image
import pickle
import random
import cv2
import os
import scipy.io as sio
import json
import re
import os.path as osp


def pose_loader(dataset, data_dir, flie_path, framenum):
	pose_list=[]
	num_list=[]
	path = data_dir+flie_path
	poses = sorted(list(os.listdir(path)))

	if dataset == 'OUMVLP':
		size = (18,2)
	elif dataset == 'CASIA':
		size = (6,2)

	for _pose in poses:
		if dataset == 'OUMVLP':
			keypoints,num = OUpose(path, _pose)
		elif dataset == 'CASIA':
			keypoints,num = CBpose(path, _pose)

		pose_list.append(keypoints)
		num_list.append(num)

	if len(pose_list) < framenum:
		pose_list.extend([np.zeros(size) for i in range(len(pose_list),framenum)])
		num_list.extend([1 for i in range(len(num_list),framenum)])
	
	poses = pose_list
	return poses,num_list	

def silh_loader(dataset, data_dir, flie_path, framenum, num_list=None):
	frame_list=[]
	flie_path = osp.join(data_dir, flie_path)

	if dataset == 'OUMVLP':
		size = (64,64)
	elif dataset == 'CASIA':
		size = (128,128)

	for frame in num_list:
		if dataset == 'OUMVLP':
			img = OUsilh(flie_path,frame)
		elif dataset == 'CASIA':
			img = CBsilh(flie_path,frame,size)

		frame_list.append(img)

	if len(frame_list) < framenum:
		frame_list.extend([np.zeros(size) for i in range(len(frame_list),framenum)])

	seqs = frame_list
	return seqs

def OF_loader(dataset, data_dir, flie_path, framenum, num_list=None):
	frame_list=[]
	flie_path = osp.join(data_dir, flie_path)

	if dataset == 'CASIA':
		size = (2,64,64)
		
	of_video_y, of_video_x = CBofs1(flie_path)
	# of_video_y, of_video_y1, of_video_x, of_video_x1 = CBofs(flie_path)
	
	for frame in num_list:
		if dataset == 'CASIA':
			img = CBof2(of_video_y, of_video_x, frame, size)
			# img = CBof2(of_video_y, of_video_y1, of_video_x, of_video_x1, frame, size)

		frame_list.append(img)

	if len(frame_list) < framenum:
		frame_list.extend([np.zeros(size) for i in range(len(frame_list),framenum)])

	seqs = frame_list
	return seqs
	
def OF_silh_loader(dataset, silhdata_dir, ofdata_dir, flie_path, framenum, num_list=None):
	frame_list=[]
	silh_path = osp.join(silhdata_dir, flie_path)
	of_path = osp.join(ofdata_dir, flie_path)

	if dataset == 'CASIA':
		size = (2,64,64)
		
	of = np.zeros((1,64,64))
	silh = np.zeros((1,64,64))

	of_video_y, of_video_y1, of_video_x, of_video_x1 = CBofs(flie_path)
	
	for frame in num_list:
		if dataset == 'CASIA':
			of[0] = CBof(of_video_y, of_video_y1, of_video_x, of_video_x1, frame)
			silh[0] = CBsilh(silh_path,frame)
		img = np.concatenate((silh,of),axis=0)
		frame_list.append(img)

	if len(frame_list) < framenum:
		frame_list.extend([np.zeros(size) for i in range(len(frame_list),framenum)])

	seqs = frame_list
	return seqs

def OUstr(str):
	strs = re.split(r'[/_]',str)
	silhpath = os.path.join(strs[1],strs[4],strs[3])
	# silhpath = os.path.join('Silhouette_' + strs[3] + '-' + strs[4], strs[1])
	return silhpath,silhpath,strs[3],strs[4]

def OUpose(path, _pose):
	_filepath = osp.join(path, _pose)
	try:
		with open(_filepath,'r') as f:
			keypoints = np.loadtxt(f)[:,:2]
		num = int(_pose.split('.')[0])
	except:
		return np.zeros((18,2)),1
	else:
		return keypoints, num

def OUsilh(path, frame):
	_img_path = str(frame).zfill(4) + '.png'
	_img_path = osp.join(path, _img_path)
	try:
		img = cv2.imread(_img_path)[:, :, 0] / 255.0
		# img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)[:, :, 0]
	except:
		return np.zeros((64,64))
	else:
		return img

def CBstr(str):
	strs = re.split(r'[/.]',str)
	impath = strs[3]
	imstrs = re.split(r'[-]',impath)
	return impath,impath,imstrs[3],strs[2]

def CBpose(path, _pose):
	_filepath = osp.join(path, _pose)
	try:
		tarray = np.loadtxt(_filepath)
		all_pose = tarray.reshape(14,2)
		# all_pose[:,0] = all_pose[:,0] - 15
		# keypoints = all_pose
		keypoints = np.zeros((6,2))
		keypoints[0] = all_pose[0]-[15,0]
		keypoints[1] = all_pose[3]-[15,0]
		keypoints[2] = all_pose[6]-[15,0]
		keypoints[3] = (all_pose[8]+all_pose[9])/2-[15,0]
		keypoints[4] = (all_pose[11]+all_pose[12])/2-[15,0]
		keypoints[5] = (all_pose[8] + all_pose[11]) / 2 - [15, 0]
		num = int(_pose.split('.')[1])
	except:
		return np.zeros((6,2)),1
	else: 
		return keypoints, num

def CBsilh(path, frame, size):	
	try:
		silhs = sorted(list(os.listdir(path)))
		_img_path = osp.join(path, silhs[frame-1])
		img = cv2.imread(_img_path) / 255.0
		img = img[:, :, 0]
		# img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)[:, :, 0]
	except:
		return np.zeros(size)
	else:
		return img

def CBofs(path):
	try:
		file_path = path + '.npz'
		of_video = np.load(file_path)['of']	
		of_video_y = of_video[:,1]
		
		# of_video_y1 = of_video_y
		# of_video_y[of_video_y > 2] = 0
		# of_video_y[of_video_y < -2] = 0
		# of_video_y = (of_video_y - np.min(of_video_y)) / (np.max(of_video_y)-np.min(of_video_y)) 
		
		of_video_x = of_video[:,0]
		
		# of_video_x1 = of_video_x
		# of_video_x[of_video_x > 2] = 0
		# of_video_x[of_video_x < -2] = 0
		# of_video_x = (of_video_x - np.min(of_video_x)) / (np.max(of_video_x)-np.min(of_video_x)) 
	except:
		zeros = np.zeros((1, 64, 64))
		return zeros, zeros
	else:
		return of_video_y, of_video_x
		# return of_video_y, of_video_y1, of_video_x, of_video_x1 

def CBofs1(path):
	try:
		file_path = path + '.npz'
		of_video = np.load(file_path)['of']	
		of_video_y = of_video[:,1]
		of_video_y = of_video_y / np.max(np.abs(of_video_y))
		
		of_video_x = of_video[:,0]
		of_video_x = of_video_x / np.max(np.abs(of_video_x))
		

	except:
		zeros = np.zeros((1, 64, 64))
		return zeros, zeros
	else:
		return of_video_y, of_video_x
		# return of_video_y, of_video_y1, of_video_x, of_video_x1 
			
class GAIT_Dataset(Dataset):
	def __init__(self, dataset, txtlist, spid, pid_num, frame_num, posedata_dir='', silhdata_dir='', ofdata_dir=''):
		fh = open(txtlist, 'r')
		data = []
		for line in fh:
			line = line.strip('\n')
			line = line.rstrip()
			words = line.split()
			label = words[1]

			if dataset == 'OUMVLP':
				posepath, silhpath, view, seq_type = OUstr(words[0])
			elif dataset == 'CASIA':
				posepath, silhpath, view, seq_type = CBstr(words[0])
			data.append((posepath,silhpath,int(label),view,seq_type))
			
		#print(words)
		self.data = data
		self.dataset = dataset
		self.posedata_dir = posedata_dir
		self.silhdata_dir = silhdata_dir
		self.ofdata_dir = ofdata_dir
		self.spid = spid
		self.pid_num = pid_num
		self.frame_num = frame_num

		self.label_set = set([i for i in range(self.spid,self.pid_num)])
		if dataset == 'OUMVLP':
			self.view = {'000','015','030','045','060','075','090','180','195','210','225','240','255','270'}
			self.seq_type = {'00','01'}
		elif dataset == 'CASIA':
			self.view = {'000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180'}
			self.seq_type = {'nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02'}

		_ = np.zeros((len(self.label_set),
					  len(self.seq_type),
					  len(self.view))).astype('int')
		_ -= 1
		self.index_dict = xr.DataArray(
			_,
			coords={'label': sorted(list(self.label_set)),
					'seq_type': sorted(list(self.seq_type)),
					'view': sorted(list(self.view))},
			dims=['label', 'seq_type', 'view'])

		for i, x in enumerate(data):
			_, _, _label, _view, _seq_type = x
			self.index_dict.loc[_label, _seq_type, _view] = i	

	def __getitem__(self, index):
		popath, silhpath, label, view, seq_type = self.data[index]
		
		poses,num_list,seqs = None,None,None
		
		if self.posedata_dir != '': 
			poses,num_list = pose_loader(self.dataset, self.posedata_dir, popath, self.frame_num)
		if self.silhdata_dir != '' and self.ofdata_dir == '' :
			seqs = silh_loader(self.dataset, self.silhdata_dir, silhpath, self.frame_num, num_list)
		elif self.ofdata_dir != '' and self.silhdata_dir == '':
			seqs = OF_loader(self.dataset, self.ofdata_dir, silhpath, self.frame_num, num_list)
		elif self.ofdata_dir != '' and self.silhdata_dir != '':
			seqs = OF_silh_loader(self.dataset, self.silhdata_dir, self.ofdata_dir, silhpath, self.frame_num, num_list)

		if poses is None:
			return seqs, label, view, seq_type
		if seqs is None:
			poses = poses[:self.frame_num]
			poses = np.array(poses)
			return poses, label, view, seq_type
			
		# print(poses.shape)
		return poses, seqs, label, view, seq_type

	def __len__(self):
		return len(self.data)
