# encoding=utf8

import numpy as np
import pandas as pd
import pickle
import librosa
import random
import gc
import os
import re
from scipy import signal
from python_speech_features import mfcc, fbank, logfbank, ssc
from data_split import K
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from data_split import wanted_words, SILENCE_LABEL, UNKNOWN_WORD_LABEL


from ipdb import set_trace as st

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000
WINDOW_SIZE_MS = 20.0
WINDOW_STRIDE_MS = 10.0
EPS = 1e-10

LEGAL_LABELS = wanted_words + [SILENCE_LABEL, UNKNOWN_WORD_LABEL]
LABEL_INDEX = {label: index for index, label in enumerate(LEGAL_LABELS)}

TEST_LENGTH = 100

##################################################################################################
# various background noise
##################################################################################################
BG_NOISE_PATH = '../data/input/train/audio/_background_noise_/'

WHITE_NOISE = 'white_noise'
RUNNING_TAP = 'running_tap'
PINK_NOISE = 'pink_noise'
EXERCISE_BIKE = 'exercise_bike'
DUDE_MIAOWING = 'dude_miaowing'
DOING_THE_DISHES = 'doing_the_dishes'

wanted_bg_noise = [WHITE_NOISE, RUNNING_TAP, PINK_NOISE, EXERCISE_BIKE, DUDE_MIAOWING, DOING_THE_DISHES]
BG_NOISE_DATA = {
    noise: librosa.core.load(path=''.join([BG_NOISE_PATH, noise, '.wav']), sr=SAMPLE_RATE)[0]
    for noise in wanted_bg_noise
    }


def read_raw_wav(path):
    data = librosa.core.load(path=path, sr=SAMPLE_RATE)[0]
    if len(data) > SAMPLE_LENGTH:
        data = data[:SAMPLE_LENGTH]
    else:
        data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
    return data


##################################################################################################
# Feature extracting methods
##################################################################################################
class FE(object):
    @staticmethod
    def calculates_spectrogram(data):
        nperseg = int(round(WINDOW_SIZE_MS * SAMPLE_RATE / 1e3))
        noverlap = int(round(WINDOW_STRIDE_MS * SAMPLE_RATE / 1e3))
        freqs, times, spec = signal.spectrogram(
            data, fs=SAMPLE_RATE, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
        return np.log(spec.T.astype(np.float32) + EPS)

    @staticmethod
    def calculates_log_mel(data):
        S = librosa.feature.melspectrogram(data, sr=SAMPLE_RATE, n_mels=64)
        return librosa.power_to_db(S, ref=np.max)

    @staticmethod
    def calculates_mfcc(data):
        return mfcc(data, samplerate=SAMPLE_RATE, winlen=0.02, winstep=0.01)

    @staticmethod
    def calculates_log_fbank(data):
        return logfbank(data, samplerate=SAMPLE_RATE, winlen=0.02, winstep=0.01, nfilt=40)





##################################################################################################
# Data augmentation methods
##################################################################################################
class Augmentataion(object):
    @staticmethod
    def shifts_in_time(data):
        roll_length = np.random.randint(-1200, 1200)
        return np.roll(data, roll_length)

    @staticmethod
    def shifts_in_pitch(data):
        n_steps = np.random.randint(-5, 5)
        return librosa.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=n_steps)

    @staticmethod
    def stretch(data):
        stretch_rate = np.random.uniform(0.8, 1.2)
        data = librosa.effects.time_stretch(data, stretch_rate)
        if len(data) > SAMPLE_LENGTH:  # sped up
            data = data[:SAMPLE_LENGTH]
        else:  # slow down
            data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
        return data

    @staticmethod
    def adds_background_noise(data):
        noise_type = random.choice(wanted_bg_noise)
        noise_weight = np.random.uniform(0.001, 0.03)
        offset = random.randint(0, 30) * SAMPLE_LENGTH
        data = data + noise_weight * BG_NOISE_DATA[noise_type][offset:offset + len(data)]
        return data


##################################################################################################
# Data augmentation pipeline
##################################################################################################
def conduct_augmentation(label, data):
    for l,d in zip(label, data):
        pass
    return label, data


##################################################################################################
# Feature extracting pipeline
##################################################################################################
def conduct_fe(data):
    # data = np.array(list(map(FE.calculates_spectrogram, data)))
    # data = np.array(list(map(FE.calculates_mfcc, data)))
    # data = np.array(list(map(FE.calculates_log_fbank, data)))
    data = np.array(list(map(FE.calculates_log_mel, data)))
    return data


##################################################################################################
# Test a known clip augmentation result
##################################################################################################
def test_clip_augment(path):
    base_name = os.path.basename(path)
    usr_id = re.sub(r'_nohash_.*$', '', base_name)

    data = read_raw_wav(path)

    shift_size_ms = 100
    data_shifted = Augmentataion.shifts_in_time(data, shift_size_ms)
    pickle.dump(data_shifted, open('../data/input/tmp/' + usr_id + '_shifted.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    stretch_rate = 0.8
    data_stretch = Augmentataion.stretch(data, stretch_rate)
    pickle.dump(data_stretch, open('../data/input/tmp/' + usr_id + '_stretch.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    noise_weight = 0.05  # 0.01 - 0.05
    noise_type = RUNNING_TAP
    data_noising = Augmentataion.adds_background_noise(data, noise_type, noise_weight)
    pickle.dump(data_noising, open('../data/input/tmp/' + usr_id + '_noising.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    n_steps = 6
    data_pitched = Augmentataion.shifts_in_pitch(data, n_steps)
    pickle.dump(data_pitched, open('../data/input/tmp/' + usr_id + '_pitched.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    _, _, spec = FE.calculates_spectrogram(data)


##################################################################################################
# Produce train data
##################################################################################################
def produce_train_data():
    def _conduct_fe_and_augmentation(root_dir, k, file_temp, train_or_valid):
        in_fold_data = {'label': [], 'data': []}

        # step1. read raw wav file
        with open(''.join([root_dir, str(k), file_temp]), 'r') as f:
            for index, l in enumerate(f.readlines()):
                # if index >= TEST_LENGTH:
                #     break
                label, file_path = l.strip().split(SPLIT_SEP)
                assert label in LEGAL_LABELS, 'illegal label {0}'.format(label)
                if label not in ['silence']:
                    data = read_raw_wav(file_path)
                else:
                    # like test/audio/clip_00293950f.wav all silence are zeros
                    data = np.zeros(SAMPLE_LENGTH)
                in_fold_data['label'].append(LABEL_INDEX[label])
                in_fold_data['data'].append(data)

        # step2. check label and data dimension
        label_len = len(in_fold_data['label'])
        data_len = len(in_fold_data['data'])
        assert label_len == data_len, 'label len {0} and data len {1} not match'.format(label_len, data_len)

        # step3. conduct data augment
        in_fold_data['label'], in_fold_data['data'] = \
            conduct_augmentation(in_fold_data['label'], in_fold_data['data'])

        # step4. conduct feature extracting
        in_fold_data['data'] = conduct_fe(in_fold_data['data'])
        st(context=21)

        # step5. write to local disk
        pickle.dump(
            obj=in_fold_data,
            file=open('../data/input/processed_train/{0}_fold_'.format(k) + train_or_valid + '.pkl', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )

    root_dir = '../data/input/train/audio/'
    for k in range(K):
        print('####################')
        # in-fold train
        print('fold {0} train augment begin'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, TRAIN_SPLIT_FILE_TEMP, 'train')
        print('fold {0} train augment done'.format(k))
        gc.collect()
        # in-fold valid
        print('fold {0} valid augment begin'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, VALID_SPLIT_FILE_TEMP, 'valid')
        print('fold {0} valid augment done'.format(k))
        gc.collect()

##################################################################################################
# Produce test data
##################################################################################################
def produce_test_data():
    root_dir = '../data/input/test/audio/'
    fname_data = {'fname': [], 'data': []}
    print('####################')
    print('read test audio data begin...')
    for index, fname in enumerate(os.listdir(root_dir)):
        if index >= TEST_LENGTH:
            break
        if os.path.isdir(fname):
            continue
        data = read_raw_wav(root_dir + fname)
        fname_data['fname'].append(fname)
        fname_data['data'].append(data)
    assert len(fname_data['fname']) == len(fname_data['data']), 'test fname and data size not match'
    print('read test audio data done')
    print('conduct test audio data FE begin...')
    fname_data['data'] = conduct_fe(fname_data['data'])
    print('conduct test audio data FE done')
    print('record test audio data begin...')
    pickle.dump(
        obj=fname_data,
        file=open('../data/input/processed_test/test.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    print('record test audio data done')


if __name__ == '__main__':
    produce_train_data()
    gc.collect()
    produce_test_data()
