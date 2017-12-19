# encoding=utf8

import librosa
import numpy as np
import random
import sys

from keras.utils import to_categorical

from data_split import K
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from data_split import wanted_words, SILENCE_LABEL, UNKNOWN_WORD_LABEL
from fe_and_augmentation import Augmentataion, FE

LEGAL_LABELS = wanted_words + [SILENCE_LABEL, UNKNOWN_WORD_LABEL]
LABEL_INDEX = {label: index for index, label in enumerate(LEGAL_LABELS)}
n_classes = len(LEGAL_LABELS)

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000

TEST_LENGTH = 500

from ipdb import set_trace as st


class AudioGenerator(object):
    def __init__(self, root_dir, k, file_temp, ori_batch_size, train_or_valid, augmentation_prob=50):
        self.root_dir = root_dir
        self.k = k
        self.file_temp = file_temp
        self.ori_batch_size = ori_batch_size
        self.train_or_valid =  train_or_valid
        self.augmentation_prob = augmentation_prob
        self.in_fold_data = self.conduct_fe_and_augmentation()

    def read_raw_wav(self, path):
        data = librosa.core.load(path=path, sr=SAMPLE_RATE)[0]
        if len(data) > SAMPLE_LENGTH:
            data = data[:SAMPLE_LENGTH]
        else:
            data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
        return data

    def conduct_fe_and_augmentation(self):
        # read original data in once
        in_fold_data = {'label': [], 'data': []}
        with open(''.join([self.root_dir, str(self.k), self.file_temp]), 'r') as f:
            for index, l in enumerate(f.readlines()):
                # if index>=TEST_LENGTH:
                #     break
                label, file_path = l.strip().split(SPLIT_SEP)
                assert label in LEGAL_LABELS, 'illegal label {0}'.format(label)
                if label not in ['silence']:
                    data = self.read_raw_wav(file_path)
                else:
                    data = np.zeros(SAMPLE_LENGTH)
                in_fold_data['label'].append(LABEL_INDEX[label])
                in_fold_data['data'].append(data)
        label_len = len(in_fold_data['label'])
        data_len = len(in_fold_data['data'])
        assert label_len == data_len, 'label len {0} and data len {1} not match'.format(label_len, data_len)
        in_fold_data['label'] = to_categorical(in_fold_data['label'], n_classes)
        print('{0} data length {1}'.format(self.train_or_valid, len(in_fold_data['data'])))
        self.steps_per_epoch = len(in_fold_data['data']) // self.ori_batch_size
        return in_fold_data

    def generator(self):
        idx_range = list(range(len(self.in_fold_data['data'])))
        idx_max = len(self.in_fold_data['data']) - 1
        while 1:
            random.shuffle(idx_range)
            for offset in range(0, idx_max, self.ori_batch_size):
                # select batch data
                begin = offset
                end = offset + self.ori_batch_size if (offset + self.ori_batch_size) <= idx_max else idx_max
                batch_data = [self.in_fold_data['data'][idx] for idx in idx_range[begin:end]]
                batch_label = [self.in_fold_data['label'][idx] for idx in idx_range[begin:end]]
                # conduct data augmentation only in 'train' mode
                if self.train_or_valid == 'train':
                    if np.random.randint(100) < self.augmentation_prob:
                        batch_data = list(map(Augmentataion.shifts_in_time, batch_data))
                    # if np.random.randint(100) < self.augmentation_prob:
                    #     batch_data = list(map(Augmentataion.shifts_in_pitch, batch_data))
                    # if np.random.randint(100) < self.augmentation_prob:
                    #     batch_data = list(map(Augmentataion.stretch, batch_data))
                    if np.random.randint(100) < self.augmentation_prob:
                        batch_data = list(map(Augmentataion.adds_background_noise, batch_data))
                # transform to spectrogram
                batch_data = np.array(list(map(FE.calculates_spectrogram, batch_data)))
                batch_label = np.array(batch_label)
                # reshape batch data and yield
                yield batch_data.reshape(tuple(list(batch_data.shape) + [1])).astype('float32'), batch_label

