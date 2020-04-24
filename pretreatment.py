# -*- coding: utf-8 -*-
# @Author  : Abner
# @Time    : 2018/12/19

import os
from scipy import misc as scisc
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse
import json

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='/home1/wxh/dataset/OUMVLP_silhouette/', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--pose_path', default='/home1/anweizhi/data/openpose_raw/OUMVLP4/', type=str,
                    help='pose dataset.')
# parser.add_argument('--output_path', default='/home1/wxh/dataset/OUMVLP_silhouette_normalization/', type=str,
#                     help='Root path for output.')
parser.add_argument('--output_path', default='/home1/wxh/dataset/OUMVLP_pose_normalization/', type=str,
                    help='Root path for output.')
parser.add_argument('--dataset', default='OUPOSE', type=str,
                    help='dataset.')
parser.add_argument('--log_file', default='pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=32, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
POSE_PATH = opt.pose_path
OUTPUT_PATH = opt.output_path
DATASET = opt.dataset
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

T_H = 64
T_W = 64


def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return 
    # print(str_log, end='')

def normalize_pose(img, pose, seq_info, frame_name, pid):
    # normalize pose keypoint coordinate
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]

    pose[:,1] = pose[:,1] - y_top
    r = img.shape[0] / 64.
    pose /= r


    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
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
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        # img = np.concatenate([_, img, _], axis=1)
    # img = img[:, left:right]

    pose[:,0] = pose[:,0] - left

    return pose.astype('float')


def cut_img(img, seq_info, frame_name, pid):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
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
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def cut_pickle(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, seq_name)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]
        img = cut_img(img, seq_info, _frame_name, pid)
        if img is not None:
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            scisc.imsave(save_path, img)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

def cut_pickle_ou(seq_info, pid):
    _id, _seq_type, _view = seq_info
    seq_name = os.path.join('Silhouette_' + _view + '-' + _seq_type,_id)
    print(seq_name)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, seq_name)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]
        img = cut_img(img, seq_info, _frame_name, pid)
        if img is not None:
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            # scisc.imsave(save_path, img)
            cv2.imwrite(save_path, img)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

def cut_pickle_oupose(seq_info, pid):
    _id, _seq_type, _view = seq_info
    seq_name = os.path.join('Silhouette_' + _view + '-' + _seq_type,_id)
    pose_name = os.path.join(_id, 'RGB', _view + '_' + _seq_type)
    print(pose_name)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, seq_name)
    pose_path = os.path.join(POSE_PATH, pose_name)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)

    frame_list = os.listdir(pose_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        _frame_num = _frame_name.split('_')[0]
        frame_path = os.path.join(seq_path, _frame_num + '.png')
        img = cv2.imread(frame_path)[:, :, 0]
        _filepath = os.path.join(pose_path, _frame_name)
        with open(_filepath, 'r') as f:
            json_dict = json.load(f)
            keypoints = json_dict['people'][0]['pose_keypoints_2d']
            pose = np.reshape(np.array(keypoints), [18, 3])[:, :2]
        pose = normalize_pose(img, pose, seq_info, _frame_name, pid)

        if pose is not None:
            # Save the pose
            save_path = os.path.join(out_dir, _frame_num + '.txt')
            np.savetxt(save_path, pose)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

pool = Pool(WORKERS)
results = list()
pid = 0

print('Pretreatment Start.\n'
      'Input path: %s\n'
      'Output path: %s\n'
      'Log file: %s\n'
      'Worker num: %d' % (
          INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

if DATASET == 'CASIA':
    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    # Walk the input path
    # for _id in id_list:
    #     seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
    #     seq_type.sort()
    #     for _seq_type in seq_type:
    #         view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))
    #         view.sort()
    #         for _view in view:
    #             seq_info = [_id, _seq_type, _view]
    #             out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    #             os.makedirs(out_dir)
    #             results.append(
    #                 pool.apply_async(
    #                     cut_pickle,
    #                     args=(seq_info, pid)))
    #             sleep(0.02)
    #             pid += 1
    # changed
    for id_path in id_list:
        _seq_type = id_path[4:9]
        _view = id_path[-3:]
        _id = id_path[:3]

        seq_info = [_id, _seq_type, _view]
        out_dir = os.path.join(OUTPUT_PATH, *seq_info)
        os.makedirs(out_dir)
        results.append(
            pool.apply_async(
                cut_pickle,
                args=(seq_info, pid)))
        sleep(0.02)
        pid += 1

elif DATASET == 'OU':
    view_list = os.listdir(INPUT_PATH)
    view_list.sort()
    l = len(view_list)
    for i in range(17,l):
        view_path = view_list[i]
        _view = view_path[-6:-3]
        _seq_type = view_path[-2:]
        view_path = os.path.join(INPUT_PATH, view_path)
        if os.path.isdir(view_path):
            id_list = os.listdir(view_path)
            id_list.sort()
            for id_path in id_list:
                _id = id_path
                seq_info = [_id, _seq_type, _view]
                out_dir = os.path.join(OUTPUT_PATH, *seq_info)
                os.makedirs(out_dir,exist_ok=True)
                results.append(
                    pool.apply_async(
                        cut_pickle_ou,
                        args=(seq_info, pid)))
                sleep(0.02)
                pid += 1

elif DATASET == 'OUPOSE':
    id_list = os.listdir(POSE_PATH)
    id_list.sort()
    # id=5657-1
    for i in range(9598,len(id_list)):
        id_path = id_list[i]
        view_list = sorted(os.listdir(os.path.join(POSE_PATH, id_path, 'RGB')))
        for view_path in view_list:
            _seq_type = view_path[-2:]
            _view = view_path[0:3]
            _id = id_path

            seq_info = [_id, _seq_type, _view]
            out_dir = os.path.join(OUTPUT_PATH, *seq_info)
            os.makedirs(out_dir, exist_ok=True)
            results.append(
                pool.apply_async(
                    cut_pickle_oupose,
                    args=(seq_info, pid)))
            sleep(0.02)
            pid += 1

pool.close()
unfinish = 1
while unfinish > 0:
    unfinish = 0
    for i, res in enumerate(results):
        try:
            res.get(timeout=0.1)
        except Exception as e:
            if type(e) == MP_TimeoutError:
                unfinish += 1
                continue
            else:
                print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                      i, type(e))
                raise e
pool.join()
