# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe
from fe_and_augmentation import SPEC, LFBANK
from copy import deepcopy
import pickle
import os
import gc
import numpy as np

from ipdb import set_trace as st

TEST_LENGTH = 10

fe_type = SPEC

def produce_test_silence():
    root_dir = '../data/input/test/audio/'
    confirm_silence = []

    # step1. read test audio wav file
    print('Read test audio data begin...')
    for index, fname in enumerate(os.listdir(root_dir)):
        # if index >= TEST_LENGTH:
        #     break
        if index % 1000 == 0:
            print('index {0} : silence {1}'.format(index, len(confirm_silence)))
        data = read_raw_wav(root_dir + fname)
        if abs(np.sum(data))<=1e-5:
            confirm_silence.append(fname)
    return confirm_silence

silence = produce_test_silence()
