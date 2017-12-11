# encoding=utf8

import re
import os
import hashlib
import random
import math
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from ipdb import set_trace as st

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
SILENCE_FILE = 'silence_file'
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 2017

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')
data_dir = '../data/input/train/audio'
search_path = os.path.join(data_dir, '*', '*.wav')


def _trans_wav_path_to_percentage_hash(wav_path):
    base_name = os.path.basename(wav_path)
    user_id = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(user_id)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    return percentage_hash


##################################################################################################
# split by validation percentage and testing percentage
##################################################################################################

def which_set(wav_path, validation_percentage, testing_percentage):
    """distribute wav file to train / valid / test by preset percentage
    """
    percentage_hash = _trans_wav_path_to_percentage_hash(wav_path)
    if percentage_hash < validation_percentage:
        result = VALID
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = TEST
    else:
        result = TRAIN
    return result


def split_data_by_percentage(valid_percentage, test_percentage, silence_percentage, unknown_percentage):
    """
    The following split guarantee :
        audios contributed by same contributor will only occurs in
        one of train / valid / test
        therefore no data leak
    """
    random.seed(RANDOM_SEED)
    wanted_data = {TRAIN: [], VALID: [], TEST: []}
    unknown_data = {TRAIN: [], VALID: [], TEST: []}
    # split train/valid/test
    for wav_path in gfile.Glob(search_path):
        _, word = os.path.split(os.path.dirname(wav_path))
        word = word.lower()
        if word == BACKGROUND_NOISE_DIR_NAME:
            continue
        set_index = which_set(
            wav_path=wav_path,
            validation_percentage=valid_percentage,
            testing_percentage=test_percentage
        )
        if word in wanted_words:
            wanted_data[set_index].append({'label': word, 'file': wav_path})
        else:
            unknown_data[set_index].append({'label': word, 'file': wav_path})
    # add silence and unknown
    for set_index in [VALID, TEST, TRAIN]:
        set_size = len(wanted_data[set_index])
        # add silence data
        silence_size = int(math.ceil(set_size * silence_percentage / 100))
        for _ in range(silence_size):
            wanted_data[set_index].append({'label': SILENCE_LABEL, 'file': SILENCE_FILE})
        # add unknown data
        random.shuffle(unknown_data[set_index])
        unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
        wanted_data[set_index].extend(unknown_data[set_index][:unknown_size])
    # shuffle ordering
    for set_index in [VALID, TEST, TRAIN]:
        random.shuffle(wanted_data[set_index])
    return wanted_data




##################################################################################################
# split K folds
##################################################################################################

def distribute_fold(wav_path, fold, K):
    """distribute wav file to train or valid in a certain fold
    """
    assert fold >= 0 and fold < K, "invalid fold {0}".format(fold)
    percentage_hash = _trans_wav_path_to_percentage_hash(wav_path)
    # make cross-valid split here
    valid_percentage_min = (fold / K) * 100
    valid_percentage_max = ((fold + 1) / K) * 100
    if percentage_hash >= valid_percentage_min and percentage_hash < valid_percentage_max:
        result = VALID
    else:
        result = TRAIN
    return result


def split_data_by_Kfold(K, silence_percentage, unknown_percentage):
    # split train/valid/test
    random.seed(RANDOM_SEED)
    ret = []
    for k in range(K):
        wanted_data = {TRAIN: [], VALID: []}
        unknown_data = {TRAIN: [], VALID: []}
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            set_index = distribute_fold(wav_path=wav_path, fold=k, K=K)
            if word in wanted_words:
                wanted_data[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_data[set_index].append({'label': word, 'file': wav_path})
        # add silence and unknown
        for set_index in [VALID, TRAIN]:
            set_size = len(wanted_data[set_index])
            # add silence data
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                wanted_data[set_index].append({'label': SILENCE_LABEL, 'file': SILENCE_FILE})
            # add unknown data
            random.shuffle(unknown_data[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            wanted_data[set_index].extend(unknown_data[set_index][:unknown_size])
        # shuffle ordering
        for set_index in [VALID, TRAIN]:
            random.shuffle(wanted_data[set_index])
        ret.append(wanted_data)
    return ret


##################################################################################################
# test split result
##################################################################################################

data = split_data_by_percentage(10, 10, 10, 10)
# data = split_data_by_Kfold(5, 10, 10)

st(context=21)
