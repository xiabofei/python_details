# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe
from fe_and_augmentation import SPEC
import pickle
import os
import gc

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

    # step3. dump original test data
    fname_data['data'] = conduct_fe(fname_data['data'], fe_type)
    pickle.dump(
        obj=fname_data,
        file=open('../data/input_backup/test_augmentation/test_original.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
produce_test_data()
