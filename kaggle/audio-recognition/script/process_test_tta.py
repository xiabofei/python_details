# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe, conduct_augmentation
from fe_and_augmentation import SPEC
from copy import deepcopy
import pickle
import os
import gc

from ipdb import set_trace as st

TEST_LENGTH = 10

TTA_RATE = 3

fe_type = SPEC

def produce_test_time_augmentation_data():
    root_dir = '../data/input/test/audio/'
    fname_data = {'fname': [], 'data': []}

    # step1. read test audio wav file
    print('Read test audio data begin...')
    for index, fname in enumerate(os.listdir(root_dir)):
        if index >= TEST_LENGTH:
            break
        if os.path.isdir(fname):
            continue
        data = read_raw_wav(root_dir + fname)
        fname_data['fname'].append(fname)
        fname_data['data'].append(data)

    # step2. check label and data dimension
    assert len(fname_data['fname']) == len(fname_data['data']), 'test fname and data size not match'
    print('Read test audio data done')

    # step3. dump original test data
    raw_label = deepcopy(fname_data['fname'])
    raw_data = deepcopy(fname_data['data'])
    fname_data['data'] = conduct_fe(fname_data['data'], fe_type)
    pickle.dump(
        obj=fname_data,
        file=open('../data/input_backup/test_augmentation/test_original.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    del fname_data
    gc.collect()

    # step4. TTA
    tta = {'fname':[], 'data':[]}
    tta['fname'] = raw_label
    for t in range(TTA_RATE):
        tta['data'] = conduct_augmentation(raw_data)
        tta['data'] = conduct_fe(tta['data'], fe_type)
        pickle.dump(
            obj=tta,
            file=open('../data/input_backup/test_augmentation/test_{0}.pkl'.format(t), 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )

produce_test_time_augmentation_data()
