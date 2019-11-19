import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import cv2
import torch.nn.functional as F

from .basic_blocks import SetBlock, BasicConv2d

class Base(nn.Module):
	def __init__(self, hidden_dim,_set_channels = [32, 64, 128]):
		super(Base, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_frame = None

		_set_in_channels = 1
		
		self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
		self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
		self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
		self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
		self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
		self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def forward(self, x):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		x = self.set_layer1(x)
		x = self.set_layer2(x)

		x = self.set_layer3(x)
		x = self.set_layer4(x)

		x = self.set_layer5(x)
		x = self.set_layer6(x)

		return x

class Base1(nn.Module):
	def __init__(self, hidden_dim,_set_channels = [32, 64, 128]):
		super(Base1, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_frame = None

		_set_in_channels = 1
		
		self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
		self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
		self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
		self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
		self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
		self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))
		self.conv1 = nn.Conv2d(in_channels=_set_channels[0]/2, out_channels=_set_channels[0], kernel_size=1)
		self.conv2 = nn.Conv2d(in_channels=_set_channels[1]/2, out_channels=_set_channels[1], kernel_size=1)

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def forward(self, x, last_x2=None, last_x4=None):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		x1 = self.set_layer1(x)
		x2 = self.set_layer2(x1)
		if last_x2 is not None:	
			n, s, c, h, w = last_x2.size()
			last_x2 = self.conv1(last_x2.view(-1,c,h,w))
			last_x2 = last_x2.view(n,s,last_x2.size(1),h,w)
			x2=x2+last_x2
		x3 = self.set_layer3(x2)
		x4 = self.set_layer4(x3)
		if last_x4 is not None:	
			n, s, c, h, w = last_x4.size()
			last_x4 = self.conv2(last_x4.view(-1,c,h,w))
			last_x4 = last_x4.view(n,s,last_x4.size(1),h,w)
			x4=x4+last_x4
		x5 = self.set_layer5(x4)
		x6 = self.set_layer6(x5)

		return x2, x4, x

		
def FrameAve(x, dim, keepdim):
	ave = x.sum(dim=dim, keepdim=keepdim)
	ave = ave/x.size(dim)
	#print(ave.shape)
	return ave
   
def f(i, Kernels, Paddings, Strides, pos):
	if (i==len(Kernels)):
		return pos, pos
	
	pos1, pos2 = f(i+1, Kernels, Paddings, Strides, pos)
	k = Kernels[i]
	p = Paddings[i]
	s = Strides[i]
	x1 = 0-p+pos1[0]*s
	y1 = 0-p+pos1[1]*s
	x2 = (k-1)-p+pos2[0]*s
	y2 = (k-1)-p+pos2[1]*s
	
	return [x1,y1], [x2,y2]

def RFbox(x, index):
	B, C, h, w = x.size()
	x = torch.abs(x)
	m = x.sum(dim = 1)
	#m = torch.max(x,1)[1]
	#print(m.size())	#(B,h,w)
	m = m.view(B, h*w)
	value, max_pos = m.sort(1,descending=True)
	rank = 3
	#max_pos = torch.max(m,1)[1]
	#print(max_pos.size())	#(B,)
	#print(max_pos)	#0~80

	Kernels = [5,3,2,3,3,2,3,3]
	Paddings = [2,1,0,1,1,0,1,1]
	Strides = [1,1,2,1,1,2,1,1]

	RoIs = torch.zeros(B,rank,5)
	for i in range(B):
		for j in range(rank):
			o_pos = max_pos[i,j]
			#o_pos = 175
			x = o_pos/9
			y = o_pos%9
			pos = [x,y]
			pos1, pos2 = f(index, Kernels, Paddings, Strides, pos)
			#print([pos1,pos2])
			RoIs[i,j,:]=torch.Tensor([i, pos1[0], pos1[1], pos2[0]+1, pos2[1]+1])	#?(index,w,h,w,h) or (index,x1,y1,x2,y2)
	
	return RoIs	

def hard_RFbox(px):
	B, rank, xy = px.size()
	bh=14
	bw=14
	RoIs = torch.zeros(B,rank,5)
	for i in range(B):
		for j in range(rank):
			#print([pos1,pos2])
			x1=px[i][j][0]-bw/2
			if(x1<0):
				x1=0
			y1=px[i][j][1]-bh/2
			if(y1<0):
				y1=0
			x2=px[i][j][0]+bw/2
			if(x2<0):
				x2=x1+1
			y2=px[i][j][1]+bh/2
			if(y2<0):
				y2=y1+1			
			RoIs[i,j,:]=torch.Tensor([i, x1, y1, x2, y2])	# (index,x1,y1,x2,y2)
	
	return RoIs	

def ROI(x,RoIs,rw,rh):
	B, S, C, h, w = x.size()
	x = x.view(B, S*C, h, w)
	Br, rank, n = RoIs.size()	#Br=B
	new_x = torch.zeros(B, rank, S*C, rw, rh).cuda()
	new_x = Variable(new_x)
	for i in range(B):
		#tmp = torch.zeros(rank, C, 70, 70)
		for j in range(rank):
			pos = RoIs[i,j]
			for k in range(1,5):
				if(pos[k]<0):
					pos[k]=0
			#print(pos)
			#print(x.size())
			ima = x[i,:,int(pos[1]):int(pos[3]),int(pos[2]):int(pos[4])]
			#print(ima.size())
			ima=ima.permute(1,2,0).contiguous()
			#print(ima.size())
			ima = ima.data.cpu().numpy()
			#print(ima.shape)
			reima = cv2.resize(ima,(rh, rw))
			#print(reima.shape)
			reima = Variable(torch.from_numpy(reima).cuda())
			reima=reima.permute(2,0,1).contiguous()
			new_x[i,j] = 	reima
			#print(tmp.size())
			
	new_x = new_x.view(B,rank,S,C, rw, rh)
	new_x = new_x.view(B*rank, S, C, rw, rh)
	return new_x

def create_adj(frames, rank):
	adj = torch.zeros(frames*rank, frames*rank)
	for i in range(frames):
		adj[3*i:3*(i+1),3*i:3*(i+1)]=1
		for j in range(rank):
			if i>0:
				adj[3*i+j,3*(i-1)+j] = 1
			if i<frames-1:
				adj[3*i+j,3*(i+1)+j] = 1
	return adj

def create_adj2(frames, rank):
	adj = torch.zeros(frames*rank, frames*rank)
	for i in range(frames):
		adj[rank*i:rank*(i+1),rank*i:rank*(i+1)]=1
		if i>0:
			adj[rank*i,rank*(i-1)] = 1
		if i<frames-1:
			adj[rank*i,rank*(i+1)] = 1
	return adj

def slow_sample(x, clip):
	len=x.size(1)
	x=x.permute(1, 0, 2, 3, 4).contiguous()
	x0,x1,x2,x3,x4=x.size()
	l_x=torch.zeros(round(len/clip)+1,x1,x2,x3,x4).cuda()
	l_x = Variable(l_x)
	j=0
	# print(len)
	# print(l_x.size(0))
	for i in range(0,len,clip):
		# print(i)
		# print(j)
		l_x[j]=x[i]
		j=j+1	
	l_x=l_x.permute(1, 0, 2, 3, 4).contiguous()			
	return l_x
	
class GraphAttention(nn.Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""

	def __init__(self, in_features, out_features, dropout, alpha, concat=True):
		super(GraphAttention, self).__init__()
		self.dropout = dropout
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat

		self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
		self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
		self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, input, adj):
		#print('a')
		#print(self.W)
		h = torch.mm(input, self.W)
		N = h.size()[0]
		f_1 = torch.matmul(h, self.a1)
		f_2 = torch.matmul(h, self.a2)
		e = self.leakyrelu(f_1 + f_2.transpose(0,1))

		#e1 = F.softmax(e, dim=1)
		attention = -9e15*torch.ones_like(e).cuda()
		#attention = torch.zeros_like(e).cuda()
		attention = Variable(attention)
		pos = adj>0
		attention[pos] = e[pos]

		#attention = torch.where(adj > 0, e, zero_vec)
		attention = F.softmax(attention, dim=-1)
		attention = F.dropout(attention, self.dropout, training=self.training)
		h_prime = torch.matmul(attention, h)

		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):

	def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
		super(GAT, self).__init__()
		self.dropout = dropout
		#self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
		self.attentions = GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)
		# for i, attention in enumerate(self.attentions):
		# 	self.add_module('attention_{}'.format(i), attention)
		self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

	def forward(self, x, adj):
		x = F.dropout(x, self.dropout, training=self.training)
		#x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
		x = self.attentions(x,adj)
		x = F.dropout(x, self.dropout, training=self.training)
		x = F.elu(self.out_att(x, adj))
		return F.log_softmax(x, dim=1)
	
class BaseNet(nn.Module):
	"""
	input_size(3,128,128) 
	"""
	def __init__(self, hidden_dim):
		super(BaseNet, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_frame = None
		
		self.base = Base(hidden_dim,_set_channels = [16, 32, 64]);
		self.gat = GAT(nfeat=256, nhid=8, nclass=16, dropout=0.6, nheads=1,alpha = 0.2)
		self.drop = nn.Dropout(p=0.5) 
		self.fc_g = nn.Linear(64 * 16 * 11, 256)  
		
		self.bin_num = [1, 2, 4, 8, 16]
		self.fc_bin = nn.ParameterList([
			nn.Parameter(
				nn.init.xavier_uniform_(
					torch.zeros(sum(self.bin_num) , 128, hidden_dim)))])

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def frame_max(self, x):
		if self.batch_frame is None:
			a = torch.max(x, 1)
			return a
		else:
			_tmp = [
				torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
				for i in range(len(self.batch_frame) - 1)
				]
			max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
			arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
			return max_list, arg_max_list

	def forward(self, silho, batch_frame=None):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		if batch_frame is not None:
			batch_frame = batch_frame[0].data.cpu().numpy().tolist()
			_ = len(batch_frame)
			for i in range(len(batch_frame)):
				if batch_frame[-(i + 1)] != 0:
					break
				else:
					_ -= 1
			batch_frame = batch_frame[:_]
			frame_sum = np.sum(batch_frame)
			if frame_sum < silho.size(1):
				silho = silho[:, :frame_sum, :, :]
			self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
		n = silho.size(0)
		x = silho.unsqueeze(2)
		del silho

		g_x = self.base(x)
		g_x = self.frame_max(g_x)[0]	
		g_x = g_x.view(g_x.size(0), -1)
		g_fc = self.fc_g(g_x)
	
		feature = g_fc.unsqueeze(1)

		return feature, None
		
class ROINet(nn.Module):
	"""
	input_size(3,128,128) 
	"""
	def __init__(self, hidden_dim):
		super(ROINet, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_frame = None
		
		self.base = Base(hidden_dim);
		self.gat = GAT(nfeat=256, nhid=8, nclass=16, dropout=0.6, nheads=1,alpha = 0.2)
		self.drop = nn.Dropout(p=0.5) 
		self.fc_g = nn.Linear(128 * 16 * 11, 256)  
		self.fc_l = nn.Linear(128*8*8, 256)  
		
		self.bin_num = [1, 2, 4, 8, 16]
		self.fc_bin = nn.ParameterList([
			nn.Parameter(
				nn.init.xavier_uniform_(
					torch.zeros(sum(self.bin_num) , 128, hidden_dim)))])

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def frame_max(self, x):
		if self.batch_frame is None:
			a = torch.max(x, 1)
			return a
		else:
			_tmp = [
				torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
				for i in range(len(self.batch_frame) - 1)
				]
			max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
			arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
			return max_list, arg_max_list

	def forward(self, silho, batch_frame=None):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		if batch_frame is not None:
			batch_frame = batch_frame[0].data.cpu().numpy().tolist()
			_ = len(batch_frame)
			for i in range(len(batch_frame)):
				if batch_frame[-(i + 1)] != 0:
					break
				else:
					_ -= 1
			batch_frame = batch_frame[:_]
			frame_sum = np.sum(batch_frame)
			if frame_sum < silho.size(1):
				silho = silho[:, :frame_sum, :, :]
			self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
		n = silho.size(0)
		x = silho.unsqueeze(2)
		del silho

		g_x = self.base(x)
		g_x = self.frame_max(g_x)[0]
		RoIs = RFbox(g_x, 0)
		B, rank, n = RoIs.size()
		#print(RoIs[0:100]) #[34,34]
		l_x = ROI(x,RoIs)
		Br, S, c, h, w=l_x.size()
		l_x = self.base(l_x)
		l_x = self.frame_max(l_x)[0]
		Br, c, h, w=l_x.size()
		l_x = l_x.view(B,rank,c,h,w)
		l_x = self.frame_max(l_x)[0]
		l_x = l_x.view(B, -1)
		l_fc = self.fc_l(l_x)
		
		
		g_x = g_x.view(g_x.size(0), -1)
		g_fc = self.fc_g(g_x)

		# gat_fc = torch.zeros(B,S*rank,16).cuda()
		# gat_fc = Variable(gat_fc)
		# adj = create_adj2(S, rank)
		# for i in range(B):
			# gat_fc[i] = self.gat(l_fc[i], adj)


		# l_fc = torch.cat([l_fc,gat_fc],2) 
		# l_fc = torch.max(l_fc, 1)[0]		
		feature = torch.cat([g_fc, l_fc],1)	
		#feature = g_fc
		#print(feature.size())
		feature = feature.unsqueeze(1)
		#print(feature.size())
		
		# feature = list()
		# n, c, h, w = g_x.size()
		# for num_bin in self.bin_num:
			# z = g_x.view(n, c, num_bin, -1)
			# z = z.mean(3) + z.max(3)[0]

			# #print(z.size()) #[128, 128, 1/2/4/8/16]
			# feature.append(z)
			# # z = gl.view(n, c, num_bin, -1)
			# # z = z.mean(3) + z.max(3)[0]
			# # feature.append(z)

		# feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
		# #print(feature.size()) #[31, 128, 128]
		# feature = feature.matmul(self.fc_bin[0])
		# #print(feature.size()) #[31, 128, 256]
		# feature = feature.permute(1, 0, 2).contiguous()

		return feature, None
			
class GATNet(nn.Module):
	"""
	input_size(3,128,128) 
	"""
	def __init__(self, hidden_dim):
		super(GATNet, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_frame = None
		
		self.base = Base(hidden_dim);
		self.gat = GAT(nfeat=256, nhid=8, nclass=16, dropout=0.6, nheads=1,alpha = 0.2)
		self.drop = nn.Dropout(p=0.5) 
		self.fc_g = nn.Linear(128 * 16 * 11, 256)  
		self.fc_l = nn.Linear(128*8*8, 256)  
		
		self.bin_num = [1, 2, 4, 8, 16]
		self.fc_bin = nn.ParameterList([
			nn.Parameter(
				nn.init.xavier_uniform_(
					torch.zeros(sum(self.bin_num) , 128, hidden_dim)))])

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def frame_max(self, x):
		if self.batch_frame is None:
			a = torch.max(x, 1)
			return a
		else:
			_tmp = [
				torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
				for i in range(len(self.batch_frame) - 1)
				]
			max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
			arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
			return max_list, arg_max_list

	def forward(self, silho, batch_frame=None):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		if batch_frame is not None:
			batch_frame = batch_frame[0].data.cpu().numpy().tolist()
			_ = len(batch_frame)
			for i in range(len(batch_frame)):
				if batch_frame[-(i + 1)] != 0:
					break
				else:
					_ -= 1
			batch_frame = batch_frame[:_]
			frame_sum = np.sum(batch_frame)
			if frame_sum < silho.size(1):
				silho = silho[:, :frame_sum, :, :]
			self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
		n = silho.size(0)
		x = silho.unsqueeze(2)
		del silho

		x = self.base(x)
		B, S, c, h, w=x.size()
		
		g_x = x.view(B*S,-1)
		g_fc = self.fc_g(g_x)
		g_fc = g_fc.view(B,S,-1)
		
		#print(x.size())
		x = self.frame_max(x)[0] #此行是主要的base提升（70->85）
		#print(g_x.size())
		x = x.view(x.size(0), -1)
		#print(g_x.size())
		fc = self.fc_g(x)


		gat_fc = torch.zeros(B,S,16).cuda()
		gat_fc = Variable(gat_fc)
		adj = create_adj2(S, 1)
		for i in range(B):
			gat_fc[i] = self.gat(g_fc[i], adj)
		#print(gat_fc.size())
		gat_fc = torch.max(gat_fc, 1)[0]
		fc = torch.cat([fc,gat_fc],1)
		

		# l_fc = torch.cat([l_fc,gat_fc],2) 
		# l_fc = torch.max(l_fc, 1)[0]		
		#feature = torch.cat([g_fc, l_fc],1)	
		#feature = g_fc
		#print(feature.size())
		feature = fc.unsqueeze(1)
		#print(feature.size())

		return feature, None
		
class RGNet(nn.Module):
	"""
	input_size(3,128,128) 
	"""
	def __init__(self, hidden_dim):
		super(RGNet, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_frame = None
		
		self.base = Base(hidden_dim);
		self.gat = GAT(nfeat=256, nhid=8, nclass=16, dropout=0.6, nheads=1,alpha = 0.2)
		self.drop = nn.Dropout(p=0.5) 
		self.fc_g = nn.Linear(128 * 16 * 11, 256)  
		self.fc_l = nn.Linear(128*8*8, 256)  
		
		self.bin_num = [1, 2, 4, 8, 16]
		self.fc_bin = nn.ParameterList([
			nn.Parameter(
				nn.init.xavier_uniform_(
					torch.zeros(sum(self.bin_num) , 128, hidden_dim)))])

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def frame_max(self, x):
		if self.batch_frame is None:
			a = torch.max(x, 1)
			return a
		else:
			_tmp = [
				torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
				for i in range(len(self.batch_frame) - 1)
				]
			max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
			arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
			return max_list, arg_max_list

	def forward(self, silho, batch_frame=None):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		if batch_frame is not None:
			batch_frame = batch_frame[0].data.cpu().numpy().tolist()
			_ = len(batch_frame)
			for i in range(len(batch_frame)):
				if batch_frame[-(i + 1)] != 0:
					break
				else:
					_ -= 1
			batch_frame = batch_frame[:_]
			frame_sum = np.sum(batch_frame)
			if frame_sum < silho.size(1):
				silho = silho[:, :frame_sum, :, :]
			self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
		n = silho.size(0)
		x = silho.unsqueeze(2)
		del silho

		g_x = self.base(x)
		g_x = self.frame_max(g_x)[0]
		RoIs = RFbox(g_x, 0)
		B, rank, n = RoIs.size()
		#print(RoIs[0:100]) #[34,34]
		l_x = ROI(x,RoIs,34,34)
		Br, S, c, h, w=l_x.size()
		l_x = self.base(l_x)
		l_x = l_x.view(Br*S, -1)
		l_fc = self.fc_l(l_x)
		l_fc = l_fc.view(B, S*rank, -1)
		
		g_x = g_x.view(g_x.size(0), -1)
		g_fc = self.fc_g(g_x)

		gat_fc = torch.zeros(B,S*rank,16).cuda()
		gat_fc = Variable(gat_fc)
		adj = create_adj2(S, rank)
		for i in range(B):
			gat_fc[i] = self.gat(l_fc[i], adj)


		l_fc = torch.cat([l_fc,gat_fc],2) 
		l_fc = torch.max(l_fc, 1)[0]		
		feature = torch.cat([g_fc, l_fc],1)	
		#feature = g_fc
		#print(feature.size())
		feature = feature.unsqueeze(1)
		#print(feature.size())
		
		# feature = list()
		# n, c, h, w = g_x.size()
		# for num_bin in self.bin_num:
			# z = g_x.view(n, c, num_bin, -1)
			# z = z.mean(3) + z.max(3)[0]

			# #print(z.size()) #[128, 128, 1/2/4/8/16]
			# feature.append(z)
			# # z = gl.view(n, c, num_bin, -1)
			# # z = z.mean(3) + z.max(3)[0]
			# # feature.append(z)

		# feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
		# #print(feature.size()) #[31, 128, 128]
		# feature = feature.matmul(self.fc_bin[0])
		# #print(feature.size()) #[31, 128, 256]
		# feature = feature.permute(1, 0, 2).contiguous()

		return feature, None
		
class RGPNet(nn.Module):
	"""
	input_size(3,128,128) 
	"""
	def __init__(self, hidden_dim):
		super(RGPNet, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_frame = None
		
		self.base = Base(hidden_dim);
		self.gat = GAT(nfeat=256, nhid=8, nclass=16, dropout=0.6, nheads=1,alpha = 0.2)
		self.drop = nn.Dropout(p=0.5) 
		self.fc_g = nn.Linear(128 * 16 * 11, 256)  
		self.fc_l = nn.Linear(128*8*8, 256)  
		
		self.bin_num = [1, 2, 4, 8, 16]
		self.fc_bin = nn.ParameterList([
			nn.Parameter(
				nn.init.xavier_uniform_(
					torch.zeros(sum(self.bin_num) , 128, hidden_dim)))])

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def frame_max(self, x):
		if self.batch_frame is None:
			a = torch.max(x, 1)
			return a
		else:
			_tmp = [
				torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
				for i in range(len(self.batch_frame) - 1)
				]
			max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
			arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
			return max_list, arg_max_list

	def forward(self, silho, px, batch_frame=None):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		if batch_frame is not None:
			batch_frame = batch_frame[0].data.cpu().numpy().tolist()
			_ = len(batch_frame)
			for i in range(len(batch_frame)):
				if batch_frame[-(i + 1)] != 0:
					break
				else:
					_ -= 1
			batch_frame = batch_frame[:_]
			frame_sum = np.sum(batch_frame)
			if frame_sum < silho.size(1):
				silho = silho[:, :frame_sum, :, :]
			self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
		n = silho.size(0)
		x = silho.unsqueeze(2)
		del silho

		g_x = self.base(x)
		g_x = self.frame_max(g_x)[0]
		
		print(px.size())
		batch, t, rank, xy = px.size()
		px = px.view(batch*t, rank, xy)
		RoIs = hard_RFbox(px)
		#RoIs = RFbox(g_x, 0)
		
		B, rank, n = RoIs.size()
		l_x = ROI(x,RoIs,14,14)
		#print(RoIs[0:100]) #[34,34]
		#l_x = ROI(x,RoIs)
		Br, S, c, h, w=l_x.size()
		l_x = self.base(l_x)
		l_x = l_x.view(Br*S, -1)
		l_fc = self.fc_l(l_x)
		l_fc = l_fc.view(B, S*rank, -1)
		
		g_x = g_x.view(g_x.size(0), -1)
		g_fc = self.fc_g(g_x)

		gat_fc = torch.zeros(B,S*rank,16).cuda()
		gat_fc = Variable(gat_fc)
		adj = create_adj2(S, rank)
		for i in range(B):
			gat_fc[i] = self.gat(l_fc[i], adj)


		l_fc = torch.cat([l_fc,gat_fc],2) 
		l_fc = torch.max(l_fc, 1)[0]		
		feature = torch.cat([g_fc, l_fc],1)	
		#feature = g_fc
		#print(feature.size())
		feature = feature.unsqueeze(1)
		#print(feature.size())
		
		return feature, None
		
class SF1Net(nn.Module):
	"""
	input_size(3,128,128) 
	"""
	def __init__(self, hidden_dim):
		super(SF1Net, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_frame = None
		
		self.slow = Base1(hidden_dim,_set_channels = [64, 128, 256]);
		self.slow1 = Base1(hidden_dim,_set_channels = [128, 256, 512]);
		self.base = Base1(hidden_dim,_set_channels = [32, 64, 128]);
		self.gat = GAT(nfeat=256, nhid=8, nclass=16, dropout=0.6, nheads=1,alpha = 0.2)
		self.drop = nn.Dropout(p=0.5) 
		self.fc_g = nn.Linear(128 * 16 * 11, 256)  
		self.fc_l = nn.Linear(256 * 16 * 11, 256)  
		self.fc_l1 = nn.Linear(512 * 16 * 11, 256)  
		self.gamma = nn.Parameter(torch.zeros(1))
		self.gamma1 = nn.Parameter(torch.zeros(1))
		self.bin_num = [1, 2, 4, 8, 16]
		self.fc_bin = nn.ParameterList([
			nn.Parameter(
				nn.init.xavier_uniform_(
					torch.zeros(sum(self.bin_num)*2 , 128, hidden_dim)))])

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Conv1d)):
				nn.init.xavier_uniform_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant(m.bias.data, 0.0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.normal(m.weight.data, 1.0, 0.02)
				nn.init.constant(m.bias.data, 0.0)

	def frame_max(self, x):
		if self.batch_frame is None:
			a = torch.max(x, 1)
			return a
		else:
			_tmp = [
				torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
				for i in range(len(self.batch_frame) - 1)
				]
			max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
			arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
			return max_list, arg_max_list
			
	def forward(self, silho, batch_frame=None, clip=2):
		# n: batch_size, s: frame_num, k: keypoints_num, c: channel
		if batch_frame is not None:
			batch_frame = batch_frame[0].data.cpu().numpy().tolist()
			_ = len(batch_frame)
			for i in range(len(batch_frame)):
				if batch_frame[-(i + 1)] != 0:
					break
				else:
					_ -= 1
			batch_frame = batch_frame[:_]
			frame_sum = np.sum(batch_frame)
			if frame_sum < silho.size(1):
				silho = silho[:, :frame_sum, :, :]
			self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
		n = silho.size(0)
		x = silho.unsqueeze(2)
		del silho

		#l_x=slow_sample(x,clip=2)
		l_x = x[:,0:x.size(1):clip]	
		l1_x= l_x[:,0:l_x.size(1):clip]	
		x2,x4,g_x = self.base(x)
		g_x = self.frame_max(g_x)[0]    
		x2,x4,l_x=self.slow(l_x,x2,x4)
		l_x = self.frame_max(l_x)[0]    
		x2,x4,l1_x=self.slow1(l1_x,x2,x4)
		l1_x = self.frame_max(l1_x)[0]    		


		g_x = g_x.view(g_x.size(0), -1)
		g_fc = self.fc_g(g_x)
		#print(l_x.size())		
		l_x = l_x.view(l_x.size(0), -1)
		l_fc = self.fc_l(l_x)
		l1_x = l1_x.view(l1_x.size(0), -1)
		l1_fc = self.fc_l1(l1_x)
		feature=g_fc + self.gamma*l_fc + self.gamma1*l1_fc
		#feature = torch.cat([g_fc, l_fc, l1_fc],1)
		feature = feature.unsqueeze(1)		
		#print(feature.size())

		return feature, None
