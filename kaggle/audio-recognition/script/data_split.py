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


################################################################
# Train data dir
################################################################
data_dir = '../data/input/train/audio'
search_path = os.path.join(data_dir, '*', '*.wav')


################################################################
# Global macro variables
################################################################

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

SILENCE_LABEL = 'silence'
SILENCE_INDEX = 0
SILENCE_FILE = 'silence_file'

UNKNOWN_WORD_LABEL = 'unknown'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 2017

CLRF = '\n'
SPLIT_SEP = '\t'

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
TRAIN_SPLIT_FILE_TEMP = '_fold_train.dat'
VALID_SPLIT_FILE_TEMP = '_fold_valid.dat'

# K fold
K = 5

# 10 wanted words and others unknown words
wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')
unknown_words = 'bed,bird,cat,dog,eight,five,four,happy,' \
                'house,marvin,nine,one,seven,' \
                'sheila,six,three,tree,two,wow,zero'.split(',')
assert len(list(set(wanted_words).intersection(set(unknown_words))))==0, \
    'same word among wanted_words and unknown words'

################################################################
# Split K folds
################################################################
def _trans_wav_path_to_percentage_hash(wav_path):
    base_name = os.path.basename(wav_path)
    usr_id = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(usr_id)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    return percentage_hash, usr_id

def distribute_fold(wav_path, fold, K):
    """distribute wav file to train or valid in a certain fold
    """
    assert fold >= 0 and fold < K, "invalid fold {0}".format(fold)
    # transform from wav file name to percentage hash (as tf audio recognition tutorial does)
    percentage_hash, usr_id = _trans_wav_path_to_percentage_hash(wav_path)
    # decide valid percentage range
    valid_percentage_min = (fold / K) * 100
    valid_percentage_max = ((fold + 1) / K) * 100
    # decide train or valid
    if percentage_hash >= valid_percentage_min and percentage_hash < valid_percentage_max:
        result = VALID
    else:
        result = TRAIN
    return result, usr_id

def split_data_by_Kfold(K, silence_percentage, unknown_percentage):
    random.seed(RANDOM_SEED)
    ret = []
    uid_list = []
    for k in range(K):
        print('{0} fold'.format(k))
        wanted_data = {TRAIN: [], VALID: []}

        ## Step1
        ## ----Guarantee same 'unknown words' doesn't exist in both train and valid set
        random.shuffle(unknown_words)
        _unknown_words_train = unknown_words[:10]
        _unknown_words_valid = unknown_words[10:]
        unknown_data_train = []
        unknown_data_valid = []
        uid = {TRAIN: [], VALID: []}
        missing_unknown_counts = 0
        train_unknown_counts = 0
        valid_unknown_counts = 0

        ## Step2
        ## ----Split 10 known words and other unknown words
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
                if set_index==TRAIN and word in _unknown_words_train:
                    train_unknown_counts += 1
                    unknown_data_train.append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path})
                elif set_index==VALID and word in _unknown_words_valid:
                    valid_unknown_counts += 1
                    unknown_data_valid.append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path})
                else:
                    missing_unknown_counts += 1
                    pass
        print('valid unknown counts : {0}'.format(valid_unknown_counts))
        print('train unknown counts : {0}'.format(train_unknown_counts))

        ## Step3
        ## ----Add 'silence' and 'unknown' according to preset 'silence_percentage' and 'unknown_percentage'
        addition_silence_unknown_count = 0
        for set_index in [VALID, TRAIN]:
            set_size = len(wanted_data[set_index])
            # add silence data
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            print('silence size : {0}'.format(silence_size))
            addition_silence_unknown_count += silence_size
            for _ in range(silence_size):
                wanted_data[set_index].append({'label': SILENCE_LABEL, 'file': SILENCE_FILE})
            # add unknown data
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            print('unknown size : {0}'.format(unknown_size))
            addition_silence_unknown_count += unknown_size
            if set_index==TRAIN:
                random.shuffle(unknown_data_train)
                wanted_data[set_index].extend(unknown_data_train[:unknown_size])
            else:
                random.shuffle(unknown_data_valid)
                wanted_data[set_index].extend(unknown_data_valid[:unknown_size])
        print('addition silence unknown count : {0}'.format(addition_silence_unknown_count))

        ## Step4
        ## ----Shuffle ordering
        for set_index in [VALID, TRAIN]:
            random.shuffle(wanted_data[set_index])
            uid[set_index] = list(set(uid[set_index]))
        ret.append(wanted_data)
        uid_list.append(uid)
        print('')
    return ret, uid_list


################################################################
# Evaluate k fold split correctness
################################################################
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

################################################################
# record Kfold split result in local file
################################################################
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


def run(K):
    ## step1
    ## ----split data by k folds and preset each fold's silence percentage / unknown percentage
    ## ----since silence data is relative easy to predict so reduce the percentage to 5
    ## ----unknown percentage is preset as 10, since public lb's unknown is 10%
    data, uid_list = split_data_by_Kfold(K, silence_percentage=5, unknown_percentage=10)
    ## step2
    ## ----guarantee same audio contributor not in train or valid same fold
    ## ----guarantee same 'unknown' words not in train or valid same fold
    evaluate_Kfold_split_correction(uid_list, K)
    ## step3
    ## ----produce 10 (5×2) files in 'data_dir' (preset in very beginning)
    ## ----<0_fold_train.dat, 0_fold_valid.dat>,...,<4_fold_train.dat, 4_fold_valid.dat>
    ## ----each file contain two columns, as follows:
    ## --------'on	../data/input/train/audio/on/9f869f70_nohash_1.wav'
    ## ----column1 as label, column2 as its wav file path
    record_Kfold_result(data)
    print('Data split done')

if __name__ == '__main__':
    run(K)
