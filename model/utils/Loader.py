# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from PIL import Image
import pickle
import random
import cv2
import os
import scipy.io as sio
import json
import re
import os.path as osp

def pose_loader(dataset, flie_path, framenum=20):
	pose_list=[]
	num_list=[]
	path = dataset+flie_path
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

	if len(pose_list) < framenum:
		pose_list.extend([np.zeros((18,2)) for i in range(len(pose_list),framenum)])
		num_list.extend([0 for i in range(len(num_list),framenum)])
	
	poses = pose_list[:framenum]
	poses = np.array(poses)
	return poses,num_list	

def silh_loader(dataset, flie_path, num_list=None, framenum=20):

	flie_path = osp.join(dataset, flie_path)
	imgs = [str(frame).zfill(4) + '.png' for frame in num_list]

	frame_list = [cv2.resize(
		cv2.imread(osp.join(flie_path, _img_path)), (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)[:, :, 0] / 255.0
				  for _img_path in imgs
				  if osp.isfile(osp.join(flie_path, _img_path))]
	if len(frame_list) < framenum:
		frame_list.extend([np.zeros((96,128)) for i in range(len(frame_list),framenum)])

	seqs = frame_list[:framenum]
	seqs = np.array(seqs)
	return seqs

def pose_loader1(dir,path,frame = 30):
	#print("pose_loader work")
	#print(path)
	#pos1=path.rfind('/')
	#pos2=path.rfind('.')
	#impath=path[pos1+1:pos2]
	pose_dir = dir+path+"/"
	#im_path = pose_dir + impath + '.0001.mat'
	#print(im_path)
	#im_path = path
	#print(pose_dir)
	image = np.zeros((36,frame),dtype='float')

	if os.path.exists(pose_dir):
		dirs = os.listdir(pose_dir)
		#print(dirs)
		#dirs = sorted(dirs)
		len_frames = len(dirs)
		#print(dirs)
		if len_frames == 0:
			image = np.zeros((36,frame),dtype='float')
			#print("erro")
		elif len_frames <= frame:
			for i in range(len_frames):
				pose_json = dirs[i]
				jsonFile = pose_dir + pose_json
				#print(jsonFile)
				with open(jsonFile,'r') as f:
					json_dict = json.load(f)
					if(len(json_dict['people']) > 0):
						keypoints = json_dict['people'][0]['pose_keypoints_2d']
						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]

						mid_hipX = (keypoints[24] + keypoints[33])/2
						mid_hipY = (keypoints[25] + keypoints[34])/2
						neckX = keypoints[3]
						neckY = keypoints[4]
						H_square = pow((mid_hipX - neckX),2) + pow((mid_hipY - neckY),2)
						H_body = pow(H_square,0.5)
						if(H_body == 0):
							for j in range(18):
							
								keypoints_X.append(0)
								keypoints_Y.append(0)
								keypoints_C.append(0)
								#print("exception")
						else:
							for j in range(18):
								if(keypoints[j*3] == 0):
									keypoints_X.append(0)
									keypoints_Y.append(0)
									keypoints_C.append(0)
								else:
									keypoints_X.append((keypoints[j*3]-neckX)/H_body)
									keypoints_Y.append((keypoints[j*3+1]-neckY)/H_body)
									keypoints_C.append(keypoints[j*3+2])
					else:
						# print("ok")
						# print(jsonFile)
						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]
						for j in range(18):

							keypoints_X.append(0)
							keypoints_Y.append(0)
							keypoints_C.append(0)

				
				image[0:18,i] =  keypoints_X
				image[18:36,i] = keypoints_Y
				#image[84:102,i] = keypoints_C
			


		elif len_frames > frame:
		#	print(">")

			count = 0
			rand = random.randint(17,len_frames)
			for i in range(rand-17,rand):
				
				pose_json = dirs[i]
				#print(pose_json)

				jsonFile = pose_dir + pose_json
				with open(jsonFile,'r') as f:
					json_dict = json.load(f)
					
					if(len(json_dict['people']) > 0):
						keypoints = json_dict['people'][0]['pose_keypoints_2d']

						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]

						mid_hipX = (keypoints[24] + keypoints[33])/2
						mid_hipY = (keypoints[25] + keypoints[34])/2
						neckX = keypoints[3]
						neckY = keypoints[4]
						H_square = pow((mid_hipX - neckX),2) + pow((mid_hipY - neckY),2)
						H_body = pow(H_square,0.5)
						if(H_body == 0):
							for j in range(18):
							
								keypoints_X.append(0)
								keypoints_Y.append(0)
								keypoints_C.append(0)
								#print("exception")
						else:
							for j in range(18):
								if(keypoints[j*3] == 0):
									keypoints_X.append(0)
									keypoints_Y.append(0)
									keypoints_C.append(0)
								else:
									keypoints_X.append((keypoints[j*3]-neckX)/H_body)
									keypoints_Y.append((keypoints[j*3+1]-neckY)/H_body)
									keypoints_C.append(keypoints[j*3+2])

						
						image[0:18,count] =  keypoints_X
						image[18:36,count] = keypoints_Y
						#image[84:102,count] = keypoints_C
						count = count + 1
			#image=np.array(image)
			#image=torch.from_numpy(image)
			#image = image.type(torch.FloatTensor)

				

					
						
	#print(image)
	image=torch.from_numpy(image)	
	#image = image.type(torch.FloatTensor)
	#image=image.permute(1,0).contiguous()
	#image = image.view(frame, 14, 9)
	#print("***")
	return image

def alphapose_loader(dir,path,frame = 128):
	#print("pose_loader work")
	#print(path)
	#pos1=path.rfind('/')
	#pos2=path.rfind('.')
	#impath=path[pos1+1:pos2]
	pose_dir = dir+path+"/"
	#im_path = pose_dir + impath + '.0001.mat'
	#print(im_path)
	#im_path = path
	#print(pose_dir)
	image = np.zeros((128,frame),dtype='float')

	if os.path.exists(pose_dir):
		dirs = os.listdir(pose_dir)
		len_frames = len(dirs)
		#print(dirs)
		if len_frames == 0:
			image = np.zeros((128,frame),dtype='float')
			#print("erro")
		elif len_frames <= frame:
			for i in range(len_frames):
				pose_json = dirs[i]
				jsonFile = pose_dir + pose_json
				#print(jsonFile)
				with open(jsonFile,'r') as f:
					json_dict = json.load(f)
					if(len(json_dict['people']) > 0):
						keypoints = json_dict['people'][0]['pose_keypoints_2d']
						#print(keypoints)
						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]

						mid_hipX = (keypoints[24] + keypoints[33])/2
						mid_hipY = (keypoints[25] + keypoints[34])/2
						neckX = keypoints[3]
						neckY = keypoints[4]
						H_square = pow((mid_hipX - neckX),2) + pow((mid_hipY - neckY),2)
						H_body = pow(H_square,0.5)
						if(H_body == 0):
							for j in range(18):
							
								keypoints_X.append(0)
								keypoints_Y.append(0)
								keypoints_C.append(0)
								#print("exception")
						else:
							for j in range(18):
								if(keypoints[j*3] == 0):
									keypoints_X.append(0)
									keypoints_Y.append(0)
									keypoints_C.append(0)
								else:
									keypoints_X.append((keypoints[j*3]-neckX)/H_body)
									keypoints_Y.append((keypoints[j*3+1]-neckY)/H_body)
									keypoints_C.append(keypoints[j*3+2])

				
				image[0:18,i] =  keypoints_X
				image[42:60,i] = keypoints_Y
				#image[84:102,i] = keypoints_C
			


		elif len_frames > frame:
		#	print(">")

			count = 0
			rand = random.randint(17,len_frames)
			for i in range(rand-17,rand):
				
				pose_json = dirs[i]
				#print(pose_json)

				jsonFile = pose_dir + pose_json
				with open(jsonFile,'r') as f:
					json_dict = json.load(f)
					
					if(len(json_dict['people']) > 0):
						keypoints = json_dict['people'][0]['pose_keypoints_2d']

						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]

						mid_hipX = (keypoints[24] + keypoints[33])/2
						mid_hipY = (keypoints[25] + keypoints[34])/2
						neckX = keypoints[3]
						neckY = keypoints[4]
						H_square = pow((mid_hipX - neckX),2) + pow((mid_hipY - neckY),2)
						H_body = pow(H_square,0.5)
						if(H_body == 0):
							for j in range(18):
							
								keypoints_X.append(0)
								keypoints_Y.append(0)
								keypoints_C.append(0)
								#print("exception")
						else:
							for j in range(18):
								if(keypoints[j*3] == 0):
									keypoints_X.append(0)
									keypoints_Y.append(0)
									keypoints_C.append(0)
								else:
									keypoints_X.append((keypoints[j*3]-neckX)/H_body)
									keypoints_Y.append((keypoints[j*3+1]-neckY)/H_body)
									keypoints_C.append(keypoints[j*3+2])

						
						image[0:18,count] =  keypoints_X
						image[42:60,count] = keypoints_Y
						#image[84:102,count] = keypoints_C
						count = count + 1
			#image=np.array(image)
			#image=torch.from_numpy(image)
			#image = image.type(torch.FloatTensor)

				

					
						
	#print(image)
	#image=torch.from_numpy(image)	
	#image = image.type(torch.FloatTensor)
	#image=image.permute(1,0).contiguous()
	#image = image.view(frame, 14, 9)
	#print("**")
	return image

def silh_data(txt):
	print("Loading data...")
	fh = open(txt, 'r')
	train_labels = []
	train_path = []
	for line in fh:
		line = line.strip('\n')
		line = line.rstrip()
		words = line.split()
		train_labels.append(words[1])	
		train_path.append(words[0])
	
	labels_set = set(np.array(train_labels))
	label_to_indices = {label: np.where(np.array(train_labels) == label)[0]
							 for label in train_labels}	
	#print(label_to_indices)
	train_path = np.array(train_path)
	train_labels = np.array(train_labels)

	return label_to_indices, train_path, train_labels, labels_set
	
def silh_testdata(mat):
	mat_file=sio.loadmat(mat)
	gallery=mat_file['gallery']
	probe=mat_file['probe']
	ga_path=[]
	ga_id=[]
	ga_angle=[]
	for i in range(gallery.shape[0]):
		# path = gallery[i,0][0]
		# strs = re.split(r'[/.]', path)
		# path = strs[0] + '/RGB/' + strs[1]
		ga_path.append(gallery[i,0][0])
		ga_id.append(gallery[i,1][0])
		ga_angle.append(gallery[i,2][0])
	ga_path = np.array(ga_path)
	ga_angle = np.array(ga_angle)
	ga_id = np.array(ga_id)
	
	angles = ['000','015','030','045','060','075','090','180','195','210','225','240','255','270']
	ga_subset = {angle: np.where(np.array(ga_angle) == angle)[0]
						 for angle in angles}
	#print(ga_subset)
						 
	for angle in angles:
		np.where(np.array(angles) == angle)[0][0]
	
	test_path=[]
	test_id=[]
	test_angle=[]
	for i in range(probe.shape[0]):
		# path = probe[i,0][0]
		# strs = re.split(r'[/.]', path)
		# path = strs[0] + '/RGB/' + strs[1]
		test_path.append(probe[i,0][0])
		test_id.append(probe[i,1][0])
		test_angle.append(probe[i,2][0])
		
	test_path = np.array(test_path)		
	test_id = np.array(test_id)		
	test_angle = np.array(test_angle)		
	
	return ga_path, ga_angle, ga_id, ga_subset, test_path, test_id, test_angle
	
class BPEI_Dataset(Dataset):
	def __init__(self, txtlist, posedata_dir, silhdata_dir, spid, pid_num, frame_num, transform=None):
		fh = open(txtlist, 'r')
		data = []
		for line in fh:
			line = line.strip('\n')
			line = line.rstrip()
			words = line.split()
			label = words[1]

			strs = re.split(r'[/_]',words[0])
			silhpath = os.path.join('Silhouette_' + strs[3] + '-' + strs[4], strs[1])
			data.append((words[0],silhpath,int(label),strs[3],strs[4]))
			
		#print(words)
		self.data = data
		self.posedata_dir = posedata_dir
		self.silhdata_dir = silhdata_dir
		self.spid = spid
		self.pid_num = pid_num
		self.frame_num = frame_num

		self.transform = transform

		self.label_set = set([i for i in range(self.spid,self.pid_num)])
		self.view = {'000','015','030','045','060','075','090','180','195','210','225','240','255','270'}
		self.seq_type = {'00','01'}

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
		
		poses,num_list = pose_loader(self.posedata_dir, popath, self.frame_num)
		seqs = silh_loader(self.silhdata_dir,silhpath, num_list, self.frame_num)

		return poses, seqs, label, view, seq_type

	def __len__(self):
		return len(self.data)

class Pair_Dataset(Dataset):
	def __init__(self, txt, data_dir, loader=silh_data):
		self.txt = txt
		# self.phrase = phrase
		self.label_to_indices, self.train_path, self.train_labels, self.labels_set = loader(txt)		
		self.dir = data_dir
		#self.loader = loader
		#self.transform = transform

	def __getitem__(self, index):
		target = np.random.randint(0, 2)
		pimg1, label1 = self.train_path[index], self.train_labels[index]
		if target == 1:		#positive pair
			siamese_index = index
			while siamese_index == index:
				siamese_index = np.random.choice(self.label_to_indices[label1])
		else:
			siamese_label = np.random.choice(list(self.labels_set - set([label1])))
			siamese_index = np.random.choice(self.label_to_indices[siamese_label])
		pimg2 = self.train_path[siamese_index]
		
		image = torch.zeros(2,36,30)
		image[0] = pose_loader(self.dir, pimg1)
		image[1] = pose_loader(self.dir, pimg2)
		
		return image, target
		

	def __len__(self):
		return len(self.train_path)
