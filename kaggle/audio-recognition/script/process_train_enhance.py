# encoding=utf8

from fe_and_augmentation import read_raw_wav, SPLIT_SEP, LEGAL_LABELS, SAMPLE_LENGTH, LABEL_INDEX
from fe_and_augmentation import K
from fe_and_augmentation import SPEC
from data_split_enhance import UNKNOWN_ENHANCE_RATE
import pickle
import numpy as np
import gc

from ipdb import set_trace as st

TEST_LENGTH = 10

fe_type = SPEC

def produce_train_data():
    def _conduct_fe_and_augmentation(root_dir, k, train_or_valid):
        if train_or_valid=='train':
            for enhance_idx in range(UNKNOWN_ENHANCE_RATE):
                print('  create enhance {0} data begin'.format(enhance_idx))
                in_fold_data = {'label': [], 'data': []}
                with open(root_dir+'{0}fold_enhance{1}_train.dat'.format(k, enhance_idx), 'r') as f:
                    for index, l in enumerate(f.readlines()):
                        if index >= TEST_LENGTH:
                            break
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
                    print('    len of enhance data {0}'.format(data_len))

                    # step3. dump to local disk
                    in_fold_data['data'] = in_fold_data['data']
                    in_fold_data['label'] = in_fold_data['label']
                    pickle.dump(
                        obj=in_fold_data,
                        file=open('../data/input_backup/train_augmentation/fold{0}/enhance{1}_train.pkl'.format(k, enhance_idx), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
                del in_fold_data
                gc.collect()
                print('  create enhance {0} data done\n'.format(enhance_idx))
        else:
            print('  create valid data begin')
            in_fold_data = {'label': [], 'data': []}
            with open(root_dir+'{0}fold_valid.dat'.format(k), 'r') as f:
                for index, l in enumerate(f.readlines()):
                    if index >= TEST_LENGTH:
                        break
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
                print('    len of valid data {0}'.format(data_len))
                # step3. dump to local disk
                in_fold_data['data'] = in_fold_data['data']
                in_fold_data['label'] = in_fold_data['label']
                pickle.dump(
                    obj=in_fold_data,
                    file=open('../data/input_backup/train_augmentation/fold{0}/valid.pkl'.format(k), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            del in_fold_data
            gc.collect()
            print('  create valid data done\n')

    root_dir = '../data/input/train/audio/'
    for k in range(K):
        print('##################################################')
        # in-fold train
        print('Fold {0} train data for unknown enhance'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, 'train')
        gc.collect()
        # in-fold valid
        print('Fold {0} valid data for unknown enhance'.format(k))
        _conduct_fe_and_augmentation(root_dir, k, 'valid')
        gc.collect()

produce_train_data()