# encoding=utf8

import librosa
import numpy as np
import random

from keras.utils import to_categorical

from data_split import SPLIT_SEP
from fe_and_augmentation import Augmentataion, conduct_fe, LABEL_INDEX, LEGAL_LABELS
import pickle

n_classes = len(LEGAL_LABELS)

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000

TEST_LENGTH = 100

EPS = 1e-8

from fe_and_augmentation import WHITE_NOISE, RUNNING_TAP, PINK_NOISE, EXERCISE_BIKE, DUDE_MIAOWING, DOING_THE_DISHES
from fe_and_augmentation import STRETCH_SLOW, STRETCH_FAST
from fe_and_augmentation import PITCH_UP, PITCH_DOWN
from fe_and_augmentation import SHIFT_TIME_FORWARD, SHIFT_TIME_BACKWARD

ORIGINAL = 'original'

# aug_noise = [WHITE_NOISE, RUNNING_TAP, PINK_NOISE, EXERCISE_BIKE, DUDE_MIAOWING]
aug_noise = [WHITE_NOISE, PINK_NOISE]
aug_stretch = [STRETCH_SLOW, STRETCH_FAST]
aug_pitch = [PITCH_UP, PITCH_DOWN]
aug_time_shift = [SHIFT_TIME_FORWARD, SHIFT_TIME_BACKWARD]

aug_type_list = aug_noise + aug_stretch + aug_pitch + aug_time_shift

from ipdb import set_trace as st


class AudioGenerator(object):
    def __init__(self, root_dir, k, batch_size, train_or_valid):
        self.root_dir = root_dir
        self.k = k
        self.batch_size = batch_size
        self.train_or_valid = train_or_valid
        self.augmentations = aug_type_list
        self.ori_data = self.get_ori_data()
        self.aug_data = self.get_aug_data()
        self.steps_per_epoch = self.ori_data['data'].shape[0] // self.batch_size

        self.ori_percentage = 20
        self.ori_hits = 0

        self.noise_percentage = 15
        self.noise_hits = {noise: 0 for noise in aug_noise}

        self.stretch_percentage = 10
        self.stretch_hits = {stretch: 0 for stretch in aug_stretch}

        self.pitch_percentage = 5
        self.pitch_hits = {pitch: 0 for pitch in aug_pitch}

        self.shift_percentage = 0
        self.shift_hits = {shift: 0 for shift in aug_time_shift}

    def check_ori_and_aug_hits(self):
        print('ori hits : {0}'.format(self.ori_hits))
        print('noise hits : {0}'.format(self.noise_hits))
        print('stretch hits : {0}'.format(self.stretch_hits))
        print('pitch hits : {0}'.format(self.pitch_hits))
        print('shift hits : {0}'.format(self.shift_hits))

    def load_pkl_data(self, augmentation):
        path = self.root_dir + 'fold{0}/'.format(self.k) + augmentation + '_{0}.pkl'.format(self.train_or_valid)
        data = pickle.load(open(path, 'rb'))
        return data

    def get_ori_data(self):
        print('...Load original data begin')
        data = self.load_pkl_data(ORIGINAL)
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
        for aug in self.augmentations:
            data[aug] = self.load_pkl_data(aug)
            data[aug]['data'] = data[aug]['data'].astype('float32')
            data[aug]['label'] = to_categorical(data[aug]['label'], n_classes)
            print('......aug type : {0}'.format(aug))
            print('......data shape : {0}'.format(data[aug]['data'].shape))
        print('...Load augmentation data done')
        return data

    def decide_aug_type(self):
        if self.train_or_valid == 'valid':
            return ORIGINAL
        lucky_num = np.random.randint(100)
        if lucky_num >= self.ori_percentage:
            aug_type = ORIGINAL
            self.ori_hits += 1
        elif lucky_num >= self.noise_percentage:
            aug_type = random.choice(aug_noise)
            self.noise_hits[aug_type] += 1
        elif lucky_num >= self.stretch_percentage:
            aug_type = random.choice(aug_stretch)
            self.stretch_hits[aug_type] += 1
        elif lucky_num >= self.pitch_percentage:
            aug_type = random.choice(aug_pitch)
            self.pitch_hits[aug_type] += 1
        else:
            aug_type = random.choice(aug_time_shift)
            self.shift_hits[aug_type] += 1
        return aug_type

    def generator(self):
        idx_range = list(range(len(self.ori_data['data'])))
        idx_max = len(self.ori_data['data']) - 1
        while 1:
            random.shuffle(idx_range)
            for offset in range(0, idx_max, self.batch_size):

                # Current batch range
                begin = offset
                end = offset + self.batch_size if (offset + self.batch_size) <= idx_max else idx_max

                # Choose original or augmentation data for current batch
                aug_type = self.decide_aug_type()

                # Prepare batch data
                '''
                if aug_type==ORIGINAL:
                    batch_data = [self.ori_data['data'][idx] for idx in idx_range[begin:end]]
                    batch_label = [self.ori_data['label'][idx] for idx in idx_range[begin:end]]
                else:
                    batch_data = [self.aug_data[aug_type]['data'][idx] for idx in idx_range[begin:end]]
                    batch_label = [self.aug_data[aug_type]['label'][idx] for idx in idx_range[begin:end]]
                batch_data = np.array(batch_data)
                batch_label = np.array(batch_label)
                '''
                '''
                '''
                if aug_type == ORIGINAL:
                    batch_data = self.ori_data['data'][idx_range[begin:end]]
                    batch_label = self.ori_data['label'][idx_range[begin:end]]
                else:
                    batch_data = self.aug_data[aug_type]['data'][idx_range[begin:end]]
                    batch_label = self.aug_data[aug_type]['label'][idx_range[begin:end]]

                # Reshape batch data and yield
                yield batch_data.reshape(tuple(list(batch_data.shape) + [1])), batch_label
