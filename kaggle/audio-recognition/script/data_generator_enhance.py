# encoding=utf8

import librosa
import numpy as np
import random

from keras.utils import to_categorical
from keras.preprocessing import image

from data_split import SPLIT_SEP
from fe_and_augmentation import Augmentataion, LABEL_INDEX, LEGAL_LABELS
import pickle
import os

n_classes = len(LEGAL_LABELS)

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000

TEST_LENGTH = 100

EPS = 1e-8

from fe_and_augmentation import NOISE, STRETCH, PITCH, SHIFT_TIME
from fe_and_augmentation import SPEC
from fe_and_augmentation import conduct_fe, conduct_augmentation

ORIGINAL = 'original'


from ipdb import set_trace as st


class AudioGenerator(object):
    def __init__(self, root_dir, k, batch_size, train_or_valid, enhance):
        self.root_dir = root_dir
        self.k = k
        self.batch_size = batch_size
        self.train_or_valid = train_or_valid
        self.enhance = enhance
        self.ori_data = self.get_ori_data()
        self.aug_data = {}
        self.steps_per_epoch = len(self.ori_data['data']) // self.batch_size

    def load_pkl_data(self, path):
        return pickle.load(open(path, 'rb'))

    def get_ori_data(self):
        print('...Load original data begin')
        if self.train_or_valid=='train':
            path = self.root_dir + 'fold{0}/enhance{1}_train.pkl'.format(self.k, self.enhance)
        if self.train_or_valid=='valid':
            path = self.root_dir + 'fold{0}/valid.pkl'.format(self.k)
        data = self.load_pkl_data(path)
        if self.train_or_valid=='train':
            data['data'] = np.array(data['data'])
        if self.train_or_valid=='valid':
            data['data'] = conduct_fe(data['data'], SPEC)
        data['label'] = to_categorical(data['label'], n_classes)
        print('...Load original data done')
        return data

    def conduct_augmentation_each_epoch(self, data):
        return conduct_augmentation(data)

    def conduct_fe_each_epoch(self, data):
        return conduct_fe(data, SPEC)

    def generator(self):
        idx_range = list(range(len(self.ori_data['data'])))
        idx_max = len(self.ori_data['data']) - 1
        while 1:
            # Do shuffle
            random.shuffle(idx_range)

            # Do train epochs
            for offset in range(0, idx_max, self.batch_size):

                # batch index
                begin = offset
                end = offset + self.batch_size if (offset + self.batch_size) <= idx_max else idx_max

                # batch data and label
                batch_data = self.ori_data['data'][idx_range[begin:end]]
                batch_data = self.conduct_augmentation_each_epoch(batch_data)
                batch_data = self.conduct_fe_each_epoch(batch_data)
                batch_label = self.ori_data['label'][idx_range[begin:end]]

                # Reshape batch data and yield
                yield batch_data.reshape(tuple(list(batch_data.shape) + [1])).astype('float32'), batch_label
