# encoding=utf8

from fe_and_augmentation import read_raw_wav, conduct_fe, SPLIT_SEP, LEGAL_LABELS, SAMPLE_LENGTH, LABEL_INDEX
from fe_and_augmentation import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP
from fe_and_augmentation import K
from fe_and_augmentation import SPEC, LFBANK
from fe_and_augmentation import conduct_augmentation_offline
import pickle
import numpy as np
import gc

TEST_LENGTH = 100

fe_type = SPEC

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

        # step3. dump original spectrogram data
        print('process fold {0} original train data begin'.format(k))
        in_fold_data['data'] = conduct_fe(in_fold_data['data'], fe_type)
        pickle.dump(
            obj=in_fold_data,
            file=open('../data/input_backup/fold{0}/original_'.format(k) + train_or_valid + '.pkl', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
        print('process fold {0} original train data done'.format(k))

        # step4. in 'train' mode, conduct offline augmentation and dump data
        if train_or_valid=='train':
            for aug_data, aug_name in conduct_augmentation_offline(in_fold_data['data']):
                print('process fold {0} train data augmentation {1} begin'.format(k, aug_name))
                label_data = {'label': [], 'data': []}
                label_data['label'] = in_fold_data['label']
                label_data['data'] = aug_data
                label_data['data'] = conduct_fe(in_fold_data['data'], fe_type)
                pickle.dump(
                    obj=label_data,
                    file=open('../data/input_backup/fold{0}/{1}_'.format(k, aug_name) + train_or_valid + '.pkl', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL
                )
                print('process fold {0} train data augmentation {1} done'.format(k, aug_name))
                del label_data
                gc.collect()

    root_dir = '../data/input/train/audio/'
    for k in range(K):
        print('####################')
        # in-fold train
        print('fold {0} train data fe and augmentation begin'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, TRAIN_SPLIT_FILE_TEMP, 'train')
        print('fold {0} train data fe and augmentation done'.format(k))
        gc.collect()
        # in-fold valid
        print('fold {0} valid data fe and augmentation begin'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, VALID_SPLIT_FILE_TEMP, 'valid')
        print('fold {0} valid data fe and augmentation done'.format(k))
        gc.collect()

produce_train_data()