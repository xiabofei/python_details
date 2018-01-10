# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe
from fe_and_augmentation import SPEC, LFBANK
from fe_and_augmentation import conduct_augmentation_offline
from copy import deepcopy
import pickle
import os
import gc
import numpy as np

from ipdb import set_trace as st

TEST_LENGTH = 10

fe_type = SPEC

def produce_test_data():
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
    print('conduct test audio data FE begin...')

    # step3. dump original test data
    raw_test_fname = deepcopy(fname_data['fname'])
    raw_test_data = deepcopy(fname_data['data'])
    fname_data['data'] = conduct_fe(fname_data['data'], fe_type)
    pickle.dump(
        obj=fname_data,
        file=open('../data/input_backup/test_augmentation/test_original.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

    for aug_data, aug_name in conduct_augmentation_offline(raw_test_data):
        print('  Augmentation : {0} begin'.format(aug_name))
        fname_data = {'fname': raw_test_fname, 'data': []}
        fname_data['data'] = aug_data
        # st(context=21)
        fname_data['data'] = conduct_fe(fname_data['data'], fe_type)
        pickle.dump(
            obj=fname_data,
            file=open('../data/input_backup/test_augmentation/test_{0}'.format(aug_name) + '.pkl', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
        print('  Augmentation : {0} done'.format(aug_name))
        del fname_data
        gc.collect()

produce_test_data()
