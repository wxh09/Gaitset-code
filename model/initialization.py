# -*- coding: utf-8 -*-
# @Author  : admin
# @Time	: 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data,load_OU_data
from .model import Model
from .utils.Loader import BPEI_Dataset


def initialize_data(config, train=False, test=False):
	print("Initializing data source...")
	# train_source, test_source = load_OU_data(**config['data'], cache=(train or test))
	# print(train_source)
	data_config = config['data']
	if train:
		train_source = BPEI_Dataset(txtlist=data_config['train_list'], 
			posedata_dir=data_config['posedata_dir'], silhdata_dir=data_config['silhdata_dir'], 
			spid=0, pid_num=data_config['train_pid'], 
			frame_num=data_config['frame_num'])		
		test_source = None
	# 	print("Loading training data...")
	# 	train_source.load_all_data()
	if test:
		test_source = BPEI_Dataset(txtlist=data_config['test_list'], 
			posedata_dir=data_config['posedata_dir'], silhdata_dir=data_config['silhdata_dir'], 
			spid=data_config['train_pid'], pid_num=data_config['train_pid']+data_config['test_pid'], 
			frame_num=data_config['frame_num'])	
		train_source = None
	# 	print("Loading test data...")
	# 	test_source.load_all_data()
	print("Data initialization complete.")
	return train_source, test_source


def initialize_model(config, train_source, test_source):
	print("Initializing model...")
	data_config = config['data']
	model_config = config['model']
	model_param = deepcopy(model_config)
	model_param['train_source'] = train_source
	model_param['test_source'] = test_source
	model_param['train_pid_num'] = data_config['train_pid']
	batch_size = int(np.prod(model_config['batch_size']))
	model_param['save_name'] = '_'.join(map(str,[
		model_config['model_name'],
		data_config['dataset'],
		data_config['train_pid'],
		# data_config['pid_shuffle'],
		model_config['hidden_dim'],
		model_config['margin'],
		batch_size,
		model_config['hard_or_full_trip'],
		model_config['frame_num'],
	]))

	m = Model(**model_param)
	print("Model initialization complete.")
	return m, model_param['save_name']


def initialization(config, train=False, test=False):
	print("Initialzing...")
	WORK_PATH = config['WORK_PATH']
	os.chdir(WORK_PATH)
	os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
	train_source, test_source = initialize_data(config, train, test)
	return initialize_model(config, train_source, test_source)