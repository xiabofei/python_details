# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe, SPLIT_SEP, LEGAL_LABELS, SAMPLE_LENGTH, LABEL_INDEX
from fe_and_augmentation import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP
from fe_and_augmentation import K
import pickle
import numpy as np
import gc

TEST_LENGTH = 100


def produce_train_data():
    def _conduct_fe_and_augmentation(root_dir, k, file_temp, train_or_valid):
        in_fold_data = {'label': [], 'data': []}
        # step1. read raw wav file
        with open(''.join([root_dir, str(k), file_temp]), 'r') as f:
            for index, l in enumerate(f.readlines()):
                if index >= TEST_LENGTH:
                    break
                label, file_path = l.strip().split(SPLIT_SEP)
                assert label in LEGAL_LABELS, 'illegal label {0}'.format(label)
                if label not in ['silence']:
                    data = read_raw_wav(file_path)
                else:
                    # like test/audio/clip_00293950f.wav all silence are zeros
                    data = np.zeros(SAMPLE_LENGTH)
                in_fold_data['label'].append(LABEL_INDEX[label])
                in_fold_data['data'].append(data)

        # step2. check label and data dimension
        label_len = len(in_fold_data['label'])
        data_len = len(in_fold_data['data'])
        assert label_len == data_len, 'label len {0} and data len {1} not match'.format(label_len, data_len)
        # step3. conduct feature extracting
        in_fold_data['data'] = conduct_fe(in_fold_data['data'])
        # step4. write to local disk
        pickle.dump(
            obj=in_fold_data,
            file=open('../data/input/processed_train/{0}_fold_'.format(k) + train_or_valid + '.pkl', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )

    root_dir = '../data/input/train/audio/'
    for k in range(K):
        print('####################')
        # in-fold train
        print('fold {0} train augment begin'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, TRAIN_SPLIT_FILE_TEMP, 'train')
        print('fold {0} train augment done'.format(k))
        gc.collect()
        # in-fold valid
        print('fold {0} valid augment begin'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, VALID_SPLIT_FILE_TEMP, 'valid')
        print('fold {0} valid augment done'.format(k))
        gc.collect()

produce_train_data()