import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
# import gc 		#内存管理（garbage collector）
# import psutil
from torch.utils.data import Dataset, DataLoader
# from .utils.Loader import GAIT_Dataset


from .network import TripletLoss, SetNet, BaseNet, Base1Net, RGPNet, RGP2Net, RGP3Net, RGP4Net, PoseBaseNet
from .utils import TripletSampler

def random_choices(list,k = 1):
	s = []
	for i in range(k):
		s.append(random.choice(list)) 
	return s


class Model:
	def __init__(self,
				 hidden_dim,
				 lr,
				 hard_or_full_trip,
				 margin,
				 num_workers,
				 batch_size,
				 restore_iter,
				 total_iter,
				 save_name,
				 dataset,
				 train_pid_num,
				 frame_num,
				 model_name,
				 train_source,
				 test_source,
				 img_size=64):

		self.save_name = save_name
		self.dataset = dataset
		self.train_pid_num = train_pid_num
		self.train_source = train_source
		self.test_source = test_source

		self.hidden_dim = hidden_dim
		self.lr = lr
		self.hard_or_full_trip = hard_or_full_trip
		self.margin = margin
		self.frame_num = frame_num
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.model_name = model_name
		self.P, self.M = batch_size

		self.restore_iter = restore_iter
		self.total_iter = total_iter

		self.img_size = img_size

		# self.encoder = SetNet(self.hidden_dim).float()
		# self.encoder = SetNet(self.hidden_dim, _set_channels=[64, 128, 256]).float() #OUMVLP
		# self.encoder = SetNet(self.hidden_dim, _set_in_channels=2).float()  #OF(x+y)
		# self.encoder = BaseNet(self.dataset, self.hidden_dim, _set_in_channels=2).float()
		# self.encoder = RGP4Net(self.dataset, self.hidden_dim).float() 
		# self.encoder = RGP4Net(self.dataset, self.hidden_dim, _set_in_channels=2).float()  #OF+Silh
		self.encoder = PoseBaseNet(self.hidden_dim).float()
		self.encoder = nn.DataParallel(self.encoder)
		# self.loss_func = nn.CrossEntropyLoss()
		self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
		self.triplet_loss = nn.DataParallel(self.triplet_loss)
		self.encoder.cuda()
		self.triplet_loss.cuda()

		self.optimizer = optim.Adam([
			{'params': self.encoder.parameters()},
		], lr=self.lr)

		self.loss = 0.
		self.hard_loss_metric = []
		self.full_loss_metric = []
		self.full_loss_num = []
		self.dist_list = []
		self.mean_dist = 0.01

		self.sample_type = 'all'

	def collate_fn(self, batch):
		batch_size = len(batch)
		feature_num = len(batch[0][0])
		poses = [batch[i][0] for i in range(batch_size)]
		seqs = [batch[i][1] for i in range(batch_size)]
		label = [batch[i][2] for i in range(batch_size)]
		view = [batch[i][3] for i in range(batch_size)]
		seq_type = [batch[i][4] for i in range(batch_size)]

		batch = [poses, seqs, label, view, seq_type, None]	#增加pose
		

		def select_frame(index):
			seq_sample = seqs[index]
			pose_sample = poses[index]
			
			frame_id_list = random.sample(list(range(len(seq_sample))), k=self.frame_num)	

			_seqs =[seq_sample[id] for id in frame_id_list]
			_poses =[pose_sample[id] for id in frame_id_list]
			return _seqs,_poses

		# seqs= list(map(select_frame, range(batch_size)))

		if self.sample_type == 'random':
			for i in range(batch_size):
				_seqs,_poses = select_frame(i)
				seqs[i]=_seqs
				poses[i]=_poses

			# seqs = np.array(seqs)
			# poses = np.array(poses)        #add for pose

		elif self.sample_type == 'consist':	#连续
			# seqs = np.asarray([seqs[i][0][:self.frame_num] for i in range(batch_size)])
			# poses = np.asarray([poses[i][0][:self.frame_num] for i in range(batch_size)])    #add for pose

			seq_size = seqs[0][0][0].shape
			pose_size = poses[0][0][0].shape
			new_seqs = np.zeros((batch_size, self.frame_num, seq_size[0], seq_size[1]))
			new_poses = np.zeros((batch_size, self.frame_num, pose_size[0], pose_size[1]))
			# print(new_seqs.shape)
			# print(new_seqs.shape)
			for i in range(batch_size):
				# print('naa')
				# print(len(seqs[i][0][:self.frame_num]))
				# print(len(poses[i][0][:self.frame_num]))
				new_seqs[i] = seqs[i][0][:self.frame_num]
				new_poses[i] = poses[i][0][:self.frame_num]      #add for pose
			
			seqs = new_seqs
			poses = new_poses

			# print(len(seqs))
			# print(seqs.shape)
			# print(len(poses))
			# print(poses.shape)

		elif self.sample_type == 'all':
			gpu_num = min(torch.cuda.device_count(), batch_size)
			batch_per_gpu = math.ceil(batch_size / gpu_num)
			batch_frames = [[
								len(seqs[i])
								for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
								if i < batch_size
								] for _ in range(gpu_num)]
			if len(batch_frames[-1]) != batch_per_gpu:
				for _ in range(batch_per_gpu - len(batch_frames[-1])):
					batch_frames[-1].append(0)
			max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
			seqs = [[
						np.concatenate([
										   seqs[i][j]
										   for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
										   if i < batch_size
										   ], 0) for _ in range(gpu_num)]
					for j in range(feature_num)]
			seqs = [np.asarray([
								   np.pad(seqs[j][_],
										  ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
										  'constant',
										  constant_values=0)
								   for _ in range(gpu_num)])
					for j in range(feature_num)]
			batch[5] = np.asarray(batch_frames)

		batch[0] = np.array(poses)
		batch[1] = np.array(seqs)        #add for pose
		batch[2] = np.asarray(label)        #add for pose
		return batch

	def fit(self):
		if self.restore_iter != 0:
			self.load(self.restore_iter)

		self.encoder.train()
		self.sample_type = 'random' #随机->连续
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

		triplet_sampler = TripletSampler(self.train_source, self.batch_size)
		train_loader = tordata.DataLoader(
			dataset=self.train_source,
			batch_sampler=triplet_sampler,
			collate_fn=self.collate_fn,
			num_workers=self.num_workers,
			# shuffle=True
			)

		_time1 = datetime.now()
		for x in train_loader:
			# print(x.size())
			poses, seqs, ids, view, seq_type, batch_frame = x
			self.restore_iter += 1
			self.optimizer.zero_grad()

			target_label = self.np2var(ids).long()
			seqs = self.np2var(seqs).float()
			poses = self.np2var(poses).float()		#add for pose
			# print(seqs.size())
			# print(poses.size())
			# print("done")

			# for i in range(len(seq)):
			# 	seq[i] = self.np2var(seq[i]).float()
			# 	pose[i] = self.np2var(pose[i]).float() 	
			# if batch_frame is not None:
			# 	batch_frame = self.np2var(batch_frame).int()

			feature, label_prob = self.encoder(x=seqs, px=poses)

			# loss = self.loss_func(label_prob, label)
			# target_label = [train_label_set.index(l) for l in label]
			# target_label = self.np2var(np.array(target_label)).long()

			triplet_feature = feature.permute(1, 0, 2).contiguous()
			triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
			(full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
			 ) = self.triplet_loss(triplet_feature, triplet_label)
			if self.hard_or_full_trip == 'hard':
				loss = hard_loss_metric.mean()
			elif self.hard_or_full_trip == 'full':
				loss = full_loss_metric.mean()

			self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
			self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
			self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
			self.dist_list.append(mean_dist.mean().data.cpu().numpy())

			if loss > 1e-9:
				loss.backward()
				self.optimizer.step()

			if self.restore_iter % 1000 == 0:
				print(datetime.now() - _time1)
				_time1 = datetime.now()

			if self.restore_iter % 1000 == 0:
				self.save()
				print('iter {}:'.format(self.restore_iter))
				print('hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)))
				print('full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)))
				print('full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)))
				self.mean_dist = np.mean(self.dist_list)
				print('mean_dist={0:.8f}'.format(self.mean_dist))
				print('lr=%f' % self.optimizer.param_groups[0]['lr'])
				print('hard or full=%r' % self.hard_or_full_trip)
				sys.stdout.flush()
				self.hard_loss_metric = []
				self.full_loss_metric = []
				self.full_loss_num = []
				self.dist_list = []

			# Visualization using t-SNE
			# if self.restore_iter % 500 == 0:
			#	 pca = TSNE(2)
			#	 pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
			#	 for i in range(self.P):
			#		 plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
			#					 pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
			#
			#	 plt.show()

			if self.restore_iter == self.total_iter:
				break

	def ts2var(self, x):
		return autograd.Variable(x).cuda()

	def np2var(self, x):
		return self.ts2var(torch.from_numpy(x))

	def ts2np(self, x):
		return x.numpy()

	def transform(self, flag, batch_size=1):
		self.encoder.eval()
		source = self.test_source if flag == 'test' else self.train_source
		# self.sample_type = 'all'
		self.sample_type = 'skip'

		data_loader = tordata.DataLoader(
			dataset=source,
			batch_size=batch_size,
			sampler=tordata.sampler.SequentialSampler(source),
			collate_fn=self.collate_fn,
			num_workers=self.num_workers)

		feature_list = list()
		view_list = list()
		seq_type_list = list()
		label_list = list()

		for i, x in enumerate(data_loader):
			poses, seqs, label, view, seq_type, batch_frame = x
			# poses, seqs, label, view, seq_type = x

			seqs = self.np2var(seqs).float()
			poses = self.np2var(poses).float()		#add for pose

			if i%1000==0:
				print('{}/{}'.format(i,len(data_loader)))
				print(seqs.size())
			# print(poses.size())
			# for j in range(len(seq)):
			# 	seq[j] = self.np2var(seq[j]).float()
			# if batch_frame is not None:
			# 	batch_frame = self.np2var(batch_frame).int()
			# print(batch_frame, np.sum(batch_frame))

			feature, _ = self.encoder(x=seqs, px=poses)
			# feature, _ = self.encoder(*seq, batch_frame)
			n, num_bin, _ = feature.size()
			feature_list.append(feature.view(n, -1).data.cpu().numpy())

			view_list += view
			seq_type_list += seq_type
			label = list(label)
			label_list += label

		return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

	def save(self):
		os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
		torch.save(self.encoder.state_dict(),
				   osp.join('checkpoint', self.model_name,
							'{}-{:0>5}-encoder.ptm'.format(
								self.save_name, self.restore_iter)))
		torch.save(self.optimizer.state_dict(),
				   osp.join('checkpoint', self.model_name,
							'{}-{:0>5}-optimizer.ptm'.format(
								self.save_name, self.restore_iter)))

	# restore_iter: iteration index of the checkpoint to load
	def load(self, restore_iter):
		self.encoder.load_state_dict(torch.load(osp.join(
			'checkpoint', self.model_name,
			'{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
		self.optimizer.load_state_dict(torch.load(osp.join(
			'checkpoint', self.model_name,
			'{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))

class PoseModel(Model):
	def fit(self):
		if self.restore_iter != 0:
			self.load(self.restore_iter)

		self.encoder.train()
		self.sample_type = 'skip' #随机->连续
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

		triplet_sampler = TripletSampler(self.train_source, self.batch_size)
		train_loader = tordata.DataLoader(
			dataset=self.train_source,
			batch_sampler=triplet_sampler,
			# collate_fn=self.collate_fn,
			num_workers=self.num_workers,
			# shuffle=True
			)

		_time1 = datetime.now()

		for poses, ids, _, _ in train_loader:
			# print(self.restore_iter)
			self.restore_iter += 1
			self.optimizer.zero_grad()

			target_label = self.ts2var(ids).long()
			poses = self.ts2var(poses).float()		#add for pose

			feature, label_prob = self.encoder(poses)

			triplet_feature = feature.permute(1, 0, 2).contiguous()
			triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
			(full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
			 ) = self.triplet_loss(triplet_feature, triplet_label)
			if self.hard_or_full_trip == 'hard':
				loss = hard_loss_metric.mean()
			elif self.hard_or_full_trip == 'full':
				loss = full_loss_metric.mean()

			self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
			self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
			self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
			self.dist_list.append(mean_dist.mean().data.cpu().numpy())

			if loss > 1e-9:
				loss.backward()
				self.optimizer.step()

			if self.restore_iter % 1000 == 0:
				print(datetime.now() - _time1)
				_time1 = datetime.now()

			if self.restore_iter % 1000 == 0:
				self.save()
				print('iter {}:'.format(self.restore_iter))
				print('hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)))
				print('full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)))
				print('full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)))
				self.mean_dist = np.mean(self.dist_list)
				print('mean_dist={0:.8f}'.format(self.mean_dist))
				print('lr=%f' % self.optimizer.param_groups[0]['lr'])
				print('hard or full=%r' % self.hard_or_full_trip)
				sys.stdout.flush()
				self.hard_loss_metric = []
				self.full_loss_metric = []
				self.full_loss_num = []
				self.dist_list = []

			if self.restore_iter == self.total_iter:
				break

	def transform(self, flag, batch_size=1):
		self.encoder.eval()
		source = self.test_source if flag == 'test' else self.train_source
		# self.sample_type = 'all'
		self.sample_type = 'skip'

		data_loader = tordata.DataLoader(
			dataset=source,
			batch_size=batch_size,
			sampler=tordata.sampler.SequentialSampler(source),
			# collate_fn=self.collate_fn,
			num_workers=self.num_workers)

		feature_list = list()
		view_list = list()
		seq_type_list = list()
		label_list = list()

		# print(len(data_loader))
		for i, x in enumerate(data_loader):
			# print(i)
			poses, label, view, seq_type = x
			# seq, view, seq_type, label, batch_frame = x

			poses = self.ts2var(poses).float()		#add for pose
			feature, _ = self.encoder(poses)

			n, num_bin, _ = feature.size()
			feature_list.append(feature.view(n, -1).data.cpu().numpy())

			view_list += view
			seq_type_list += seq_type
			label = list(self.ts2np(label))
			label_list += label

		return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
