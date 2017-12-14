# encoding=utf8

import numpy as np
import pandas as pd
import pickle
import sys
import librosa
import random
import gc
import os
import re
from scipy.io import wavfile
from scipy import signal
from data_split import K
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from data_split import wanted_words, SILENCE_LABEL, UNKNOWN_WORD_LABEL

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from ipdb import set_trace as st

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000
LEGAL_LABELS = wanted_words + [SILENCE_LABEL, UNKNOWN_WORD_LABEL]
LABEL_INDEX = {label: index for index, label in enumerate(LEGAL_LABELS)}
TEST_LENGTH = 1000

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
    noise:librosa.core.load(path=''.join([BG_NOISE_PATH, noise, '.wav']), sr=SAMPLE_RATE)[0]
    for noise in wanted_bg_noise
    }

def read_raw_wav(path):
    data = librosa.core.load(path=path, sr=SAMPLE_RATE)[0]
    if len(data) > SAMPLE_LENGTH:
        data = data[:SAMPLE_LENGTH]
    else:
        data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
    return data


def conduct_augmentation(data):
    pass


class Augmentataion(object):
    @staticmethod
    def shifts_in_time(data, roll_length):
        return np.roll(data, roll_length)

    @staticmethod
    def shifts_in_pitch(data, shift_size_ms=100):
        n_steps = int(round(shift_size_ms * SAMPLE_RATE / 1e3))
        return  librosa.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=n_steps)

    @staticmethod
    def stretch(data, stretch_rate):
        data = librosa.effects.time_stretch(data, stretch_rate)
        if len(data) > SAMPLE_LENGTH:  # sped up
            data = data[:SAMPLE_LENGTH]
        else:  # slow down
            data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
        return data

    @staticmethod
    def adds_background_noise(data, noise_type, noise_weight):
        offset = random.randint(0, 30) * SAMPLE_LENGTH
        data = data + noise_weight * BG_NOISE_DATA[noise_type][offset:offset+len(data)]
        return data

    @staticmethod
    def calculates_spectrogram(data, window_size_ms=20.0, window_stride_ms=10.0, eps=1e-10):
        nperseg = int(round(window_size_ms * SAMPLE_RATE / 1e3))
        noverlap = int(round(window_stride_ms * SAMPLE_RATE / 1e3))
        freqs, times, spec = signal.spectrogram(
            data, fs=SAMPLE_RATE, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    @staticmethod
    def build_mfcc_fingerprint(data, n_mels):
        data = librosa.feature.melspectrogram(data, sr=SAMPLE_RATE, n_mels=40)
        data = librosa.power_to_db(data, ref=np.max)
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

    noise_weight = 0.05 # 0.01 - 0.05
    noise_type = RUNNING_TAP
    data_noising = Augmentataion.adds_background_noise(data, noise_type, noise_weight)
    pickle.dump(data_noising, open('../data/input/tmp/' + usr_id + '_noising.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    n_steps = 6
    data_pitched = Augmentataion.shifts_in_pitch(data, n_steps)
    pickle.dump(data_pitched, open('../data/input/tmp/' + usr_id + '_pitched.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    _,  _, spec = Augmentataion.calculates_spectrogram(data)

    st(context=21)

a_sample_clip_path = '../data/input/train/audio/bed/0a7c2a8d_nohash_0.wav'
test_clip_augment(a_sample_clip_path)

# sys.exit(0)

if __name__ == '__main__':
    root_dir = '../data/input/train/audio/'


    def _conduct_augment_and_save_to_disk(k, file_temp, train_or_valid):
        in_fold_data = []
        # read raw wav file
        with open(''.join([root_dir, str(k), file_temp]), 'r') as f:
            for index, l in enumerate(f.readlines()):
                if index > TEST_LENGTH:
                    break
                label, file_path = l.strip().split(SPLIT_SEP)
                assert label in LEGAL_LABELS, 'illegal label {0}'.format(label)
                if label not in ['silence']:
                    data = read_raw_wav(file_path)
                    in_fold_data.append({'label': LABEL_INDEX[label], 'data': data})
                else:
                    # like test/audio/clip_00293950f.wav all silence are zeros
                    in_fold_data.append({'label': LABEL_INDEX[label], 'data': np.zeros(SAMPLE_LENGTH)})
        # data augment
        in_fold_data = conduct_augmentation(in_fold_data)
        pickle.dump(
            obj=in_fold_data,
            file=open('{0}_fold_'.format(k) + train_or_valid + '.format(k)', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )


    for k in range(K):
        print('####################')
        # in-fold train
        print('fold {0} train augment begin'.format(k))
        _conduct_augment_and_save_to_disk(k, TRAIN_SPLIT_FILE_TEMP, 'train')
        print('fold {0} train augment done'.format(k))
        gc.collect()
        # in-fold valid
        print('fold {0} valid augment begin'.format(k))
        _conduct_augment_and_save_to_disk(k, VALID_SPLIT_FILE_TEMP, 'valid')
        print('fold {0} valid augment done'.format(k))
        gc.collect()
