import numpy as np
import cv2
import os
import os.path as osp
import re

# flow_dir = "/home1/wxh/dataset/CASIAB_OF_normalization/"
# silh_dir = "/home1/mzh/casia_B_of/unzip_contour/"

# flow_silh_dir = "/home/wxh/dataset/CASIAB_OF_silh_normalization/"
# max_y = []
# max_x = []
# flows = sorted(list(os.listdir(flow_silh_dir)))
# for i in range(0, len(flows)):
	# _flow = flows[i]

	# file_path = osp.join(flow_silh_dir, _flow)
	# try:
		# of_video = np.load(file_path)['of']
		# of_video_y = of_video[:,1]		
		# my = np.max(np.abs(of_video_y))
		# max_y.append(my)

		# of_video_x = of_video[:,0]
		# mx = np.max(np.abs(of_video_x))
		# max_x.append(mx)
		
	# except:
		# print("error")
		
# np.savez_compressed('max.npz', max_y=max_y, max_x=max_x)
		
	# for i in range(len(of_video)):
		# of_y = of_video[i,1]
		# of_x = of_video[i,0]
#
#     strs = re.split(r'[-.]', _flow)
#     silh_subdir = osp.join(silh_dir, strs[0], strs[1] + '-' + strs[2], strs[3])
#
#     for _silh in sorted(list(os.listdir(silh_subdir))):
#         # print(_silh)
#         index = int(re.split(r'[-.]', _silh)[-2])
#
#         silh_img = cv2.imread(osp.join(silh_subdir, _silh))[:, :, 0] / 255.
#         try:
#             flow_img_y = flow_img[index-1, 1]
#         except:
#             break
#
#         # print(flow_img_y.shape)
#         try:
#             flow_img_y[silh_img == 0] = 0
#             x, y = np.nonzero(silh_img)
#             flow_img_y = flow_img_y[min(x) - 10:max(x) + 10, max(min(y) - 10, 0):max(y) + 10]
#             flow_img_y = np.abs(flow_img_y)
#             flow_img_y = cv2.resize(flow_img_y, (88, 128))
#         except:
#             break
#
#         img_out = osp.join(flow_silh_dir, _flow[:-4])
#         os.makedirs(img_out, exist_ok=True)
#         cv2.imwrite(osp.join(img_out, _silh), flow_img_y)

# file_path = "/home/wxh/dataset/CASIAB_OF_silh_normalization/005-bg-01-000.npz"
file_path = "001-bg-01-090.npz"
of_video = np.load(file_path)['of']
# print(of_video.shape)
of_video_y = of_video[:,1]
of_video_x = of_video[:,0]
# # of_video_y1 = of_video_y
# # n = np.mean(of_video_y)
# # of_video_y[of_video_y > 2] = 0
# # of_video_y[of_video_y < -2] = 0
# # of_video_y = (of_video_y - np.min(of_video_y)) / (np.max(of_video_y)-np.min(of_video_y)) 
# # of_video_y = (of_video_y - np.mean(of_video_y)) / np.std(of_video_y)  

for i in range(len(of_video)):
	of = of_video_x[i]
	# of = np.abs(of)
	# of1 = of_video_y1[i]
	# of[of1==0] = 0
	of1 = of
	# of[of > 2] = 0
	# of[of < -2] = 0
	of = (of - np.min(of)) / (np.max(of)-np.min(of)) 
	of[of1==0] = 0
	of = of * 255
	# of = cv2.resize(of, (64, 64), interpolation=cv2.INTER_NEAREST)
	# of = cv2.resize(of, (128, 128), interpolation=cv2.INTER_NEAREST)
	# print(of.shape)
	cv2.imwrite('001-bg-01-090-npz/' + str(i) + '.png', of)
	
# file_path = "057_scene1_nm_H_105_2/40.png"
# img = cv2.imread(file_path)
# print(img.shape)
# print(img[10,10,0])
# print(img[762,415,0])
