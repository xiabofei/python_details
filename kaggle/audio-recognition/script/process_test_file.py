# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe
from model_fit_generator import FE_TYPE
import pickle
import os

TEST_LENGTH = 500


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
    fname_data['data'] = conduct_fe(fname_data['data'], FE_TYPE)
    print('test audio data FE shape {0}'.format(fname_data['data'].shape))
    print('conduct test audio data FE done')
    print('record test audio data begin...')
    pickle.dump(
        obj=fname_data,
        file=open('../data/input/processed_test/test_{0}.pkl'.format(FE_TYPE), 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    print('record test audio data done')

produce_test_data()
