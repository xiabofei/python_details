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
from keras.preprocessing import image
from time import time
from multiprocessing.dummy import Pool
import random

from ipdb import set_trace as st

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000
WINDOW_SIZE_MS = 20.0
WINDOW_STRIDE_MS = 10.0
EPS = 1e-10

LEGAL_LABELS = wanted_words + [SILENCE_LABEL, UNKNOWN_WORD_LABEL]
LABEL_INDEX = {label: index for index, label in enumerate(LEGAL_LABELS)}

##################################################################################################
# various background noise
##################################################################################################
BG_NOISE_PATH = '../data/input/train/audio/_background_noise_/'

WHITE_NOISE = 'white_noise'
RUNNING_TAP = 'running_tap'
PINK_NOISE = 'pink_noise'
EXERCISE_BIKE = 'exercise_bike'
DOING_THE_DISHES = 'doing_the_dishes'
DUDE_MIAOWING = 'dude_miaowing'

NOISE = 'noise'
STRETCH = 'stretch'
PITCH = 'pitch'
SHIFT_TIME = 'shift_time'

wanted_bg_noise = [WHITE_NOISE, RUNNING_TAP, PINK_NOISE, EXERCISE_BIKE, DOING_THE_DISHES, DUDE_MIAOWING]
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


# feature extracting type
SPEC = 'spectrogram'
MFCC = 'mfcc'
LFBANK = 'log_fbank'
LMEL = 'log_mel'


class FE(object):
    @staticmethod
    def calculates_spectrogram(data):
        nperseg = int(round(WINDOW_SIZE_MS * SAMPLE_RATE / 1e3))
        noverlap = int(round(WINDOW_STRIDE_MS * SAMPLE_RATE / 1e3))
        freqs, times, spec = signal.spectrogram(
            data, fs=SAMPLE_RATE, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
        spec = np.log(spec.T.astype(np.float32) + EPS)
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + EPS)
        spec = (spec - 0.5) * 2
        return spec

    @staticmethod
    def calculates_log_mel(data):
        S = librosa.feature.melspectrogram(data, sr=SAMPLE_RATE, n_mels=128)
        return librosa.power_to_db(S, ref=np.max)

    @staticmethod
    def calculates_mfcc(data):
        return mfcc(data, samplerate=SAMPLE_RATE, winlen=0.02, winstep=0.01)

    @staticmethod
    def calculates_log_fbank(data):
        return logfbank(data, samplerate=SAMPLE_RATE, winlen=0.02, winstep=0.01, nfilt=40)


def conduct_fe(data, fe_type):
    if fe_type == SPEC:
        data = np.array(list(map(FE.calculates_spectrogram, data)))
    elif fe_type == MFCC:
        data = np.array(list(map(FE.calculates_mfcc, data)))
    elif fe_type == LFBANK:
        data = np.array(list(map(FE.calculates_log_fbank, data)))
    elif fe_type == LMEL:
        data = np.array(list(map(FE.calculates_log_mel, data)))
    return data


##################################################################################################
# Data augmentation online
##################################################################################################
class Augmentataion(object):
    @staticmethod
    def shifts_in_time(data):
        roll_length = np.random.randint(-3200, 3200)
        return np.roll(data, roll_length)

    @staticmethod
    def shifts_in_pitch(data):
        n_steps = np.random.randint(-5, 5)
        data = librosa.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=n_steps)
        return data

    @staticmethod
    def stretch(data):
        stretch_rate = np.random.uniform(0.7, 0.9)
        data = librosa.effects.time_stretch(data, stretch_rate)
        # speed up
        if len(data) > SAMPLE_LENGTH:
            data = data[:SAMPLE_LENGTH]
        # slow down
        else:
            data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
        return data

    @staticmethod
    def adds_background_noise(data):
        noise_type = random.choice(wanted_bg_noise)
        noise_weight = np.random.uniform(0.001, 0.05)
        offset = random.randint(0, 30) * SAMPLE_LENGTH
        data = data + noise_weight * BG_NOISE_DATA[noise_type][offset:offset + len(data)]
        return data


##################################################################################################
# Data augmentation offline
##################################################################################################
class SilenceEnhance(object):
    @classmethod
    def _adds_noise(cls, data, noise_type, noise_weight, offset):
        if noise_weight>0:
            data = data + noise_weight * BG_NOISE_DATA[noise_type][offset:offset + SAMPLE_LENGTH]
        return data

    @classmethod
    def _adds_doing_the_dishes_noise(cls, data):
        # 90s
        weight = np.random.beta(1,3)*1
        offset = int(np.random.uniform(0.0, 85.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, DOING_THE_DISHES, weight, offset)

    @classmethod
    def _adds_dude_miaowing_noise(cls, data):
        # 60s
        weight = np.random.beta(1, 3) * 1
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, DUDE_MIAOWING, weight, offset)

    @classmethod
    def _adds_running_tap_noise(cls, data):
        # 60s
        weight = np.random.beta(1,3)* 1
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, RUNNING_TAP, weight, offset)

    @classmethod
    def _adds_exercise_bike_noise(cls, data):
        # 60s
        weight = np.random.beta(1,3) * 1
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, EXERCISE_BIKE, weight, offset)

    @classmethod
    def _adds_white_noise(cls, data):
        # 60s
        weight = np.random.beta(1, 3) * 1
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, WHITE_NOISE, weight, offset)

    @classmethod
    def _adds_pink_noise(cls, data):
        # 60s
        weight = np.random.beta(1, 3) * 1
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, PINK_NOISE, weight, offset)

    @classmethod
    def adds_noise(cls, data):
        noise_adder = random.choice(
            [
                cls._adds_doing_the_dishes_noise,
                cls._adds_dude_miaowing_noise,
                cls._adds_white_noise,
                cls._adds_pink_noise,
                cls._adds_exercise_bike_noise,
                cls._adds_running_tap_noise
            ]
        )
        return noise_adder(data)

class AugmentationOffline(object):
    #################################################
    # Add background noise
    #################################################
    @classmethod
    def _adds_noise(cls, data, noise_type, noise_weight, offset):
        if noise_weight>0:
            data = data + noise_weight * BG_NOISE_DATA[noise_type][offset:offset + SAMPLE_LENGTH]
        return data

    @classmethod
    def _adds_doing_the_dishes_noise(cls, data):
        # 90s
        weight = np.random.beta(1,3)*0.5
        offset = int(np.random.uniform(0.0, 85.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, DOING_THE_DISHES, weight, offset)

    @classmethod
    def _adds_dude_miaowing_noise(cls, data):
        # 60s
        weight = np.random.beta(1, 3) * 0.9
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, DUDE_MIAOWING, weight, offset)

    @classmethod
    def _adds_running_tap_noise(cls, data):
        # 60s
        weight = np.random.beta(1,3)*0.5
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, RUNNING_TAP, weight, offset)

    @classmethod
    def _adds_exercise_bike_noise(cls, data):
        # 60s
        weight = np.random.beta(1,3)*0.4
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, EXERCISE_BIKE, weight, offset)

    @classmethod
    def _adds_white_noise(cls, data):
        # 60s
        weight = np.random.beta(1, 3) * 0.05
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, WHITE_NOISE, weight, offset)

    @classmethod
    def _adds_pink_noise(cls, data):
        # 60s
        weight = np.random.beta(1, 3) * 0.05
        offset = int(np.random.uniform(0.0, 55.0)*SAMPLE_LENGTH)
        return cls._adds_noise(data, PINK_NOISE, weight, offset)

    @classmethod
    def adds_noise(cls, data):
        noise_adder = random.choice(
            [
                cls._adds_doing_the_dishes_noise,
                cls._adds_dude_miaowing_noise,
                cls._adds_white_noise,
                cls._adds_pink_noise,
                cls._adds_exercise_bike_noise,
                cls._adds_running_tap_noise
            ]
        )
        return noise_adder(data)

    #################################################
    # Stretch
    #################################################
    @classmethod
    def _stretch(cls, data, stretch_rate):
        data = librosa.effects.time_stretch(data, stretch_rate)
        if len(data) > SAMPLE_LENGTH:
            data = data[:SAMPLE_LENGTH]
        else:
            data = np.pad(data, (0, max(0, SAMPLE_LENGTH - len(data))), 'constant')
        return data

    @classmethod
    def stretch(cls, data):
        rate = np.random.uniform(0.95, 1.05)
        return cls._stretch(data, rate)

    #################################################
    # Pitch
    #################################################
    @classmethod
    def _pitch(cls, data, n_steps):
        data = librosa.effects.pitch_shift(data, sr=SAMPLE_RATE, n_steps=n_steps)
        return data

    @classmethod
    def pitch(cls, data):
        n_steps = np.random.randint(-2, 2)
        return cls._pitch(data, n_steps)

    #################################################
    # Shift time
    #################################################
    @classmethod
    def _time(cls, data, roll_length):
        return np.roll(data, roll_length)

    @classmethod
    def shift_time(cls, data):
        roll_length = np.random.randint(-2000, 2000)
        return cls._time(data, roll_length)


def conduct_augmentation(data):
    augmentation_list = [
        (AugmentationOffline.adds_noise, NOISE),
        (AugmentationOffline.shift_time, SHIFT_TIME),
    ]
    # for aug in augmentation_list:
    #     if np.random.uniform(0.0, 1.0)<0.1:
    #         data = list(map(aug[0], data))
    for aug in augmentation_list:
        data = list(map(aug[0], data))
    return data
