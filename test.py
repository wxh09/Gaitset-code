from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf,cbconf,cbofconf,poseconf
import scipy.io as sio

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test') 
parser.add_argument('--iter', default='12000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
parser.add_argument('--test', default=True, type=boolean_string,help='test:true. Default: TRUE')
parser.add_argument('--type', default='ousilh', type=str,help='train:true. Default: TRUE')
parser.add_argument('--display', default=False, type=str,help='display the whole accuracy matrix. Default: False')
parser.add_argument('--angle_num', default='10.0', type=float, help='angle_num: the number of angles')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, angle_num, each_angle=False): 
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / angle_num
    if not each_angle:
        result = np.mean(result)
    return result

if opt.type == 'ousilh':
    conf = conf
elif opt.type == 'cbsilh':
    conf = cbconf
elif opt.type == 'cbof':
	conf = cbofconf
elif opt.type == 'pose':
    conf = poseconf

if conf['data']['dataset'] == 'CASIA':
	opt.angle_num = 10.0
elif conf['data']['dataset'] == 'OUMVLP':
	opt.angle_num = 14.0
	
m = initialization(conf, test=opt.test)[0]

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Evaluating...')
acc = evaluation(test, conf['data'])
print('Evaluation complete. Cost:', datetime.now() - time)

# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
# for i in range(1):
#     print('===Rank-%d (Include identical-view cases)===' % (i + 1))
#     print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
#         np.mean(acc[0, :, :, i]),
#         np.mean(acc[1, :, :, i]),
#         np.mean(acc[2, :, :, i])))



for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    print('NM: %.3f' % (
        np.mean(acc[0, :, :, i])))
    if conf['data']['dataset'] == 'CASIA':
        print('BG: %.3f,\tCL: %.3f' % (
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model, excluding identical-view cases
# e.g.
# ===Rank-1 (Exclude identical-view cases)===
# NM: 94.964,     BG: 87.239,     CL: 70.355
# for i in range(1):
#     print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
#     print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
#         de_diag(acc[0, :, :, i]),
#         de_diag(acc[1, :, :, i]),
#         de_diag(acc[2, :, :, i])))

for i in range(1):
    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
    print('NM: %.3f' % (
        de_diag(acc[0, :, :, i],opt.angle_num)))
    if conf['data']['dataset'] == 'CASIA':
        print('BG: %.3f,\tCL: %.3f' % (
            de_diag(acc[1, :, :, i],opt.angle_num),
            de_diag(acc[2, :, :, i],opt.angle_num)))

# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]

np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
	print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
	print('NM:', np.mean(acc[0, :, :, i], axis =1)) 
	if conf['data']['dataset'] == 'CASIA':
		print('BG:', np.mean(acc[1, :, :, i], axis =1)) 
		print('CL:', np.mean(acc[2, :, :, i], axis =1)) 
    # print('BG:', de_diag(acc[1, :, :, i], True))
    # print('CL:', de_diag(acc[2, :, :, i], True))

if opt.display:
    # sio.savemat('/home1/wxh/Gaitset-code/CCR.mat', {'CCR': acc})
    for i in range(1):
        print('===Rank-%d of each angle ===' % (i + 1))
        print('NM:', acc[0, :, :, i])
        if conf['data']['dataset'] == 'CASIA':
            print('BG:', acc[1, :, :, i])
            print('CL:', acc[2, :, :, i])
