# encoding=utf8

import re
import os
import hashlib
import random
import math
from collections import defaultdict

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from ipdb import set_trace as st

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = 'silence'
SILENCE_INDEX = 0
SILENCE_FILE = 'silence_file'
UNKNOWN_WORD_LABEL = 'unknown'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 2017

CLRF = '\n'

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
TRAIN_SPLIT_FILE_TEMP = '_fold_train.dat'
VALID_SPLIT_FILE_TEMP = '_fold_valid.dat'
SPLIT_SEP = '\t'
K = 5

wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')
data_dir = '../data/input/train/audio'
search_path = os.path.join(data_dir, '*', '*.wav')


def _trans_wav_path_to_percentage_hash(wav_path):
    base_name = os.path.basename(wav_path)
    usr_id = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(usr_id)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    return percentage_hash, usr_id


##################################################################################################
# split by validation percentage and testing percentage
##################################################################################################

def which_set(wav_path, validation_percentage, testing_percentage):
    """distribute wav file to train / valid / test by preset percentage
    """
    percentage_hash, usr_id = _trans_wav_path_to_percentage_hash(wav_path)
    if percentage_hash < validation_percentage:
        result = VALID
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = TEST
    else:
        result = TRAIN
    return result, usr_id

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
        set_index, usr_id = which_set(
            wav_path=wav_path,
            validation_percentage=valid_percentage,
            testing_percentage=test_percentage
        )
        if word in wanted_words:
            wanted_data[set_index].append({'label': word, 'file': wav_path})
        else:
            unknown_data[set_index].append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path})
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
    percentage_hash, usr_id = _trans_wav_path_to_percentage_hash(wav_path)
    # make cross-valid split here
    valid_percentage_min = (fold / K) * 100
    valid_percentage_max = ((fold + 1) / K) * 100
    if percentage_hash >= valid_percentage_min and percentage_hash < valid_percentage_max:
        result = VALID
    else:
        result = TRAIN
    return result, usr_id

def split_data_by_Kfold(K, silence_percentage, unknown_percentage):
    # split train/valid/test
    random.seed(RANDOM_SEED)
    ret = []
    uid_list = []
    for k in range(K):
        wanted_data = {TRAIN: [], VALID: []}
        unknown_data = {TRAIN: [], VALID: []}
        uid = {TRAIN: [], VALID: []}
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            set_index, usr_id = distribute_fold(wav_path=wav_path, fold=k, K=K)
            uid[set_index].append(usr_id)
            if word in wanted_words:
                wanted_data[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_data[set_index].append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path})
        # add silence and unknown
        addition_silence_unknown_count = 0
        for set_index in [VALID, TRAIN]:
            set_size = len(wanted_data[set_index])
            # add silence data
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            addition_silence_unknown_count += silence_size
            for _ in range(silence_size):
                wanted_data[set_index].append({'label': SILENCE_LABEL, 'file': SILENCE_FILE})
            # add unknown data
            random.shuffle(unknown_data[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            addition_silence_unknown_count += unknown_size
            wanted_data[set_index].extend(unknown_data[set_index][:unknown_size])
        print('addition silence unknown count : {0}'.format(addition_silence_unknown_count))
        # shuffle ordering
        for set_index in [VALID, TRAIN]:
            random.shuffle(wanted_data[set_index])
            uid[set_index] = list(set(uid[set_index]))
        ret.append(wanted_data)
        uid_list.append(uid)
    return ret, uid_list


##################################################################################################
# evaluate Kfold split data leak
##################################################################################################

def evaluate_Kfold_split_correction(uid_list, K):
    # uid count
    uid_in_Kfold = []
    for k in range(K):
        uid_all = len(uid_list[k][TRAIN]) + len(uid_list[k][VALID])
        uid_in_Kfold.append(uid_all)
    assert len(list(set(uid_in_Kfold))) == 1, \
        "different audio contributor number in different folds"
    # in fold split evaluation
    for k in range(K):
        uid_intersection = list(set(uid_list[k][TRAIN]).intersection(set(uid_list[k][VALID])))
        assert len(uid_intersection) == 0, \
            "same audio contributor in both train set and valid set"
    # between fold evaluation
    last_uid_valid = []
    last_uid_train = []
    uid_num = uid_in_Kfold[0]
    for k in range(K):
        current_uid_valid = uid_list[k][VALID]
        current_uid_train = uid_list[k][TRAIN]

        print('train intersection : {0}'.format(
            len(list(set(last_uid_train).intersection(set(current_uid_train)))) / uid_num))

        assert len(list(set(last_uid_valid).intersection(set(current_uid_valid)))) == 0, \
            "same audio contributor in different fold validation"
        last_uid_valid = current_uid_valid
        last_uid_train = current_uid_train

##################################################################################################
# record Kfold split result in local file
##################################################################################################

def record_Kfold_result(data):
    labels_all = wanted_words + [SILENCE_LABEL, UNKNOWN_WORD_LABEL]
    label_count_all = []
    for k in range(K):
        label_count = {TRAIN: defaultdict(int), VALID: defaultdict(int)}
        # record train data in fold k
        fold_data_trn = data[k][TRAIN]
        with open('../data/input/train/audio/{0}'.format(k)+TRAIN_SPLIT_FILE_TEMP, 'w') as f:
            for d in fold_data_trn:
                assert d['label'] in labels_all, 'unwanted label {0} occurs'.format(d['label'])
                label_count[TRAIN][d['label']] += 1
                f.write(SPLIT_SEP.join([d['label'], d['file']]) + CLRF)
        # record valid data in fold k
        fold_data_vld = data[k][VALID]
        with open('../data/input/train/audio/{0}'.format(k)+VALID_SPLIT_FILE_TEMP, 'w') as f:
            for d in fold_data_vld:
                assert d['label'] in labels_all, 'unwanted label {0} occurs'.format(d['label'])
                label_count[VALID][d['label']] += 1
                f.write(SPLIT_SEP.join([d['label'], d['file']]) + CLRF)
        label_count_all.append(label_count)




# data = split_data_by_percentage(10, 10, 10, 10)
def run(K):
    data, uid_list = split_data_by_Kfold(K, silence_percentage=5, unknown_percentage=20)
    evaluate_Kfold_split_correction(uid_list, K)
    record_Kfold_result(data)
    print('data split done')

if __name__ == '__main__':
    run(K)
