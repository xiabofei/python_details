# encoding=utf8

from fe_and_augmentation import read_raw_wav, SPLIT_SEP, LEGAL_LABELS, SAMPLE_LENGTH, LABEL_INDEX
from fe_and_augmentation import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP
from fe_and_augmentation import K
from fe_and_augmentation import SPEC
import pickle
import numpy as np
import gc

from ipdb import set_trace as st

TEST_LENGTH = 10

fe_type = SPEC

def produce_train_data():
    def _conduct_fe_and_augmentation(root_dir, k, file_temp, train_or_valid):
        in_fold_data = {'label': [], 'data': []}

        # step1. read raw wav file
        with open(''.join([root_dir, str(k), file_temp]), 'r') as f:
            for index, l in enumerate(f.readlines()):
                # if index >= TEST_LENGTH:
                #     break
                label, file_path = l.strip().split(SPLIT_SEP)
                assert label in LEGAL_LABELS, 'illegal label {0}'.format(label)
                if label not in ['silence']:
                    data = read_raw_wav(file_path)
                else:
                    data = np.zeros(SAMPLE_LENGTH)
                in_fold_data['label'].append(LABEL_INDEX[label])
                in_fold_data['data'].append(data)

        # step2. check label and data dimension
        label_len = len(in_fold_data['label'])
        data_len = len(in_fold_data['data'])
        assert label_len == data_len, 'label len {0} and data len {1} not match'.format(label_len, data_len)

        # step3. dump to local disk
        in_fold_data['data'] = in_fold_data['data']
        in_fold_data['label'] = in_fold_data['label']
        pickle.dump(
            obj=in_fold_data,
            file=open('../data/input_backup/train_augmentation/fold{0}/original_'.format(k) + train_or_valid + '.pkl', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
        del in_fold_data
        gc.collect()

    root_dir = '../data/input/train/audio/'
    for k in range(K):
        print('##################################################')
        # in-fold train
        print('Fold {0} train data for online augmentation'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, TRAIN_SPLIT_FILE_TEMP, 'train')
        gc.collect()
        # keep fold valid unchanged
        '''
        # in-fold valid
        print('Fold {0} valid data for online augmentation'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, VALID_SPLIT_FILE_TEMP, 'valid')
        gc.collect()
        '''

produce_train_data()