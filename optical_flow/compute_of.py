import glob
import cv2
import torch
import math
import numpy
import numpy as np
import os
import re
from warnings import warn
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
######### Parameters #########
VIDEOS_DIR = '/home2/gait-datasets/CASIA-B/videos/'
SILH_DIR = '/home2/gait-datasets/CASIA-B/silhouettes/'
VIDEOS_SUFFIX = '*.avi'
OUTPUT_DIR = '/home1/wxh/dataset/CASIAB_OF_silh_normalization_128/'
GPU_ID = 0
MODEL_PATH = '/home1/wxh/Gaitset-code/optical_flow/opt_model' + '.pytorch'
T_H = 128
T_W = 128
##############################

def Backward(tensorInput, tensorFlow):
	Backward_tensorGrid = {}
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, tensorInput):
				tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
				tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
				tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleBasic(tensorInput)
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.load_state_dict(torch.load(MODEL_PATH))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFlow = []

		tensorFirst = [ self.modulePreprocess(tensorFirst) ]
		tensorSecond = [ self.modulePreprocess(tensorSecond) ]

		for intLevel in range(5):
			if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
				tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
				tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))
			# end
		# end

		tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

		for intLevel in range(len(tensorFirst)):
			tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
		# end

		return tensorFlow
	# end
# end

##########################################################

def estimate(tensorFirst, tensorSecond):
	moduleNetwork = Network().cuda().eval()
	# assert(tensorFirst.size(1) == tensorSecond.size(1))
	# assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tensorFlow[0, :, :, :].cpu()
# end

##########################################################

def compute_of(frame, previous_frame):
	torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
	#torch.cuda.device(5)  # change this if you have a multiple graphics cards and you want to utilize them
	torch.cuda.set_device(GPU_ID)
	torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

	tensorFirst = torch.FloatTensor(
		previous_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
					1.0 / 255.0))
	tensorSecond = torch.FloatTensor(
		frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
					1.0 / 255.0))

	optical_flow = estimate(tensorFirst, tensorSecond)
	optical_flow = optical_flow.numpy()

	return optical_flow

def cut_img(frame, img, seq_info, framenum):
	# A silhouette contains too little white pixels
	# might be not valid for identification.
	if img.sum() <= 10000:
		message = 'seq:%s, frame:%s, no data.' % (
			seq_info, framenum)
		warn(message)
		# log_print(pid, WARNING, message)
		return None,None
	# Get the top and bottom point
	y = img.sum(axis=1)
	y_top = (y != 0).argmax(axis=0)
	y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
	img = img[y_top:y_btm + 1, :]
	frame = frame[y_top:y_btm + 1, :, :]
	# As the height of a person is larger than the width,
	# use the height to calculate resize ratio.
	_r = img.shape[1] / img.shape[0]
	_t_w = int(T_H * _r)
	img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
	frame = cv2.resize(frame, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
	# Get the median of x axis and regard it as the x center of the person.
	sum_point = img.sum()
	sum_column = img.sum(axis=0).cumsum()
	x_center = -1
	for i in range(sum_column.size):
		if sum_column[i] > sum_point / 2:
			x_center = i
			break
	if x_center < 0:
		message = 'seq:%s, frame:%s, no center.' % (
			seq_info, framenum)
		warn(message)
		# log_print(pid, WARNING, message)
		return None, None
	h_T_W = int(T_W / 2)
	left = x_center - h_T_W
	right = x_center + h_T_W
	if left <= 0 or right >= img.shape[1]:
		left += h_T_W
		right += h_T_W
		_ = np.zeros((img.shape[0], h_T_W))
		img = np.concatenate([_, img, _], axis=1)
		_ = np.zeros((img.shape[0], h_T_W, frame.shape[2]))
		frame = np.concatenate([_, frame, _], axis=1)
	img = img[:, left:right]
	frame = frame[:, left:right]
	return frame.astype('uint8'),img


if __name__ == '__main__':
	# Loading data.
	# TODO: Adapt this loading part to the corresponding file hierarchy.
	files = glob.glob(VIDEOS_DIR + VIDEOS_SUFFIX)
	files = sorted(files)
	for i in range(0,len(files)):
		#005-cl-02-108
		print(i)
		file = files[i]
		seq_info = re.split(r'[/.]', file)[-2]
		strs = re.split(r'[-]', seq_info)
		if strs[1] == 'bkgrd':
			continue
		print(seq_info)
		silh_subdir = os.path.join(SILH_DIR, strs[0], strs[1] + '-' + strs[2], strs[3])
		silh_list = sorted(os.listdir(silh_subdir))

		previous_frame = None
		rgb_frame = None
		of_video = []
		framenum = 1
		cap = cv2.VideoCapture(file)
		while cap.isOpened():
			# print(framenum)
			# Capture frame-by-frame
			ret, frame = cap.read()
			if ret:
				silh_path = os.path.join(silh_subdir, seq_info + '-' + str(framenum).zfill(3) + '.png')
				silh_img = None
				if not os.path.exists(silh_path):
					framenum += 1
					previous_frame = None
					continue

				silh_img = cv2.imread(silh_path)[:, :, 0]
				rgb_frame,silh_img = cut_img(frame, silh_img, file, framenum)
				framenum += 1

				if previous_frame is not None and rgb_frame is not None:
					of = compute_of(rgb_frame, previous_frame)
					of = of.transpose(1, 2, 0)
					of[silh_img == 0] = 0
					of = of.transpose(2, 0, 1)
					# of_video.append(of.astype(numpy.int16))
					of_video.append(of)

				if rgb_frame is not None:
					previous_frame = rgb_frame.copy()
				else:
					previous_frame = None

			# Break the loop
			else:
				break
				
		of_video = np.array(of_video)
		cap.release()
		outname = os.path.splitext(os.path.split(file)[1])[0]
		os.makedirs(OUTPUT_DIR, exist_ok=True)
		numpy.savez_compressed(OUTPUT_DIR + outname + '.npz', of=of_video)
		# numpy.savez_compressed('001-bg-01-090.npz', of=of_video)
		

	# previous_frame = None
	# rgb_frame = None
	# of_video = []
	# framenum = 1
	# file = "/home1/mzh/057_scene1_nm_H_105_2.mp4"
	# cap = cv2.VideoCapture(file)
	# while cap.isOpened():
		# # Capture frame-by-frame
		# ret, frame = cap.read()
		
		# if ret:
			# # cv2.imwrite('057_scene1_nm_H_105_2-rgb/' + str(framenum) + '.png', frame)
			# if previous_frame is not None:
				# of = compute_of(frame, previous_frame)
				# # of = of[1]
				# # of1 = of
				# # of = (of - np.min(of)) / (np.max(of)-np.min(of)) 
				# # of[of1==0] = 0
				# # of = of * 255
				# if framenum == 29:
					# numpy.savez_compressed('29.npz', of=of)
				# # cv2.imwrite('057_scene1_nm_H_105_2-npz/' + str(framenum) + '.png', of[1])
				# # of_video.append(of.astype(numpy.int16))

			# previous_frame = frame.copy()
			# framenum += 1
		# # Break the loop
		# else:
			# break
	# cap.release()
	# numpy.savez_compressed('057_scene1_nm_H_105_2.npz', of=of_video)

		# For reading.
		# of = numpy.load(OUTPUT_DIR + outname + '.npz')['of']
		# of = of / 100.0

	# frame = cv2.imread("/home1/wxh/Gaitset-code/optical_flow/001-bg-01-090-rgb/71.png")
	# previous_frame = cv2.imread("/home1/wxh/Gaitset-code/optical_flow/001-bg-01-090-rgb/70.png")
	# of = compute_of(frame, previous_frame)
	# numpy.savez_compressed('1.npz', of=of)
	# of = np.load('1.npz')['of']
	# of = of * 100
	# of1 = numpy.abs(of[1])
	# # of0 = numpy.abs(of[0])
	# cv2.imwrite("2.png",of1)
