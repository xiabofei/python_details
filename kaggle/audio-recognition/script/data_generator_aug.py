# encoding=utf8

import librosa
import numpy as np
import random

from keras.utils import to_categorical
from keras.preprocessing import image

from data_split import SPLIT_SEP
from fe_and_augmentation import Augmentataion, conduct_fe, LABEL_INDEX, LEGAL_LABELS
import pickle
import os

n_classes = len(LEGAL_LABELS)

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000

TEST_LENGTH = 100

EPS = 1e-8

from fe_and_augmentation import NOISE, STRETCH, PITCH, SHIFT_TIME

ORIGINAL = 'original'

from ipdb import set_trace as st


class AudioGenerator(object):
    def __init__(self, root_dir, k, batch_size, train_or_valid):
        self.root_dir = root_dir
        self.k = k
        self.batch_size = batch_size
        self.train_or_valid = train_or_valid
        self.ori_data = self.get_ori_data()
        self.aug_data = self.get_aug_data()
        self.steps_per_epoch = self.ori_data['data'].shape[0] // self.batch_size
        self.aug_hits = 0

    def check_aug_hits(self):
        print('aug hits : {0}'.format(self.aug_hits))

    def load_pkl_data(self, path):
        return pickle.load(open(path, 'rb'))

    def get_ori_data(self):
        print('...Load original data begin')
        path = self.root_dir + 'fold{0}/'.format(self.k) + 'original_{0}.pkl'.format(self.train_or_valid)
        data = self.load_pkl_data(path)
        data['data'] = data['data'].astype('float32')
        data['label'] = to_categorical(data['label'], n_classes)
        print('......original data shape : {0}'.format(data['data'].shape))
        print('...Load original data done')
        return data

    def get_aug_data(self):
        if self.train_or_valid == 'valid':
            return {}
        print('...Load augmentation data begin')
        data = {}
        aug_dir = self.root_dir + 'fold{0}/'.format(self.k)
        aug_file_list = sorted(filter(lambda s: s.startswith('aug'), os.listdir(aug_dir)))[:-2]
        self.aug_file_list = aug_file_list

        for aug in aug_file_list:
            data[aug] = self.load_pkl_data(aug_dir + aug)
            data[aug]['data'] = data[aug]['data'].astype('float32')
            data[aug]['label'] = to_categorical(data[aug]['label'], n_classes)
            print('......aug file : {0}'.format(aug))
            print('......data shape : {0}'.format(data[aug]['data'].shape))
        print('...Load augmentation data done')
        return data

    def decide_aug_file(self, aug_prob):
        if self.train_or_valid == 'valid':
            return ORIGINAL
        return random.choice(self.aug_file_list)

    def generator(self):
        idx_range = list(range(len(self.ori_data['data'])))
        idx_max = len(self.ori_data['data']) - 1
        while 1:
            random.shuffle(idx_range)
            for offset in range(0, idx_max, self.batch_size):

                # Current batch range
                begin = offset
                end = offset + self.batch_size if (offset + self.batch_size) <= idx_max else idx_max

                # Choose original file or augmentation file for current batch
                aug_file = self.decide_aug_file(50)

                # Prepare batch data
                if aug_file == ORIGINAL:
                    batch_data = self.ori_data['data'][idx_range[begin:end]]
                    batch_label = self.ori_data['label'][idx_range[begin:end]]
                else:
                    batch_data = self.aug_data[aug_file]['data'][idx_range[begin:end]]
                    batch_label = self.aug_data[aug_file]['label'][idx_range[begin:end]]

                # Reshape batch data and yield
                yield batch_data.reshape(tuple(list(batch_data.shape) + [1])), batch_label
