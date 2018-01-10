# encoding=utf8

from model_fit import get_model, RUNS_IN_FOLD, AudioGenerator, VALID_SPLIT_FILE_TEMP, batch_size, FE_TYPE
from model_fit import n_classes, LEGAL_LABELS
from fe_and_augmentation import conduct_fe
from data_split import K
from calculate_cv import get_cv_acc_score
from collections import Counter
import numpy as np
import pandas as pd
import pickle
import gc

from ipdb import set_trace as st

CV_DIR = '../data/output/'
MODEL_SUBDIR = 'model/'
VALID_SUBDIR = 'valid/'
SUBMIT_SUBDIR = 'submit/'


def get_valid_data(fold):
    print('\nPrepare valid data in fold {0}\n'.format(fold))
    valid_generator = AudioGenerator(
        root_dir='../data/input/train/audio/',
        k=fold,
        file_temp=VALID_SPLIT_FILE_TEMP,
        ori_batch_size=batch_size,
        train_or_valid='valid',
    )
    fname_valid = valid_generator.in_fold_data['fname']
    truth_valid = valid_generator.in_fold_data['truth']
    X_valid, y_valid = valid_generator.in_fold_data['data'], valid_generator.in_fold_data['label']
    X_valid, y_valid = np.array(conduct_fe(X_valid, fe_type=FE_TYPE)), np.array(y_valid)
    X_valid = X_valid.reshape(tuple(list(X_valid.shape) + [1])).astype('float32')
    del valid_generator
    gc.collect()
    return fname_valid, truth_valid, X_valid


from fe_and_augmentation import WHITE_NOISE, RUNNING_TAP, PINK_NOISE, EXERCISE_BIKE, DUDE_MIAOWING, DOING_THE_DISHES
from fe_and_augmentation import STRETCH_SLOW, STRETCH_FAST
from fe_and_augmentation import PITCH_UP, PITCH_DOWN
from fe_and_augmentation import SHIFT_TIME_FORWARD, SHIFT_TIME_BACKWARD

ORIGINAL = 'original'

aug_type_list = [
    WHITE_NOISE, RUNNING_TAP, PINK_NOISE, EXERCISE_BIKE, DUDE_MIAOWING, DOING_THE_DISHES,
    STRETCH_SLOW, STRETCH_FAST,
    PITCH_UP, PITCH_DOWN,
    SHIFT_TIME_FORWARD, SHIFT_TIME_BACKWARD,
]


def get_test_data(aug_type):
    print('prepare test data aug type : {0}'.format(aug_type))
    test_dir = '../data/input_backup/test_augmentation/'
    d_test = pickle.load(open(test_dir + 'test_{0}.pkl'.format(aug_type), 'rb'))
    fname_test, X_test = d_test['fname'], d_test['data']
    X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
    del d_test
    gc.collect()
    return fname_test, X_test


##################################################
# Test time augmentation
##################################################
## Get TTA result
for aug_type in aug_type_list:
    print('Augmentation type : {0}'.format(aug_type))
    fname_test, X_test = get_test_data(aug_type)
    for fold in range(K):
        preds_test = np.zeros((X_test.shape[0], n_classes))
        for run in range(RUNS_IN_FOLD):
            model = get_model()
            print('Predict using trained model weights fold {0} run {1}'.format(fold, run))
            bst_model_path = CV_DIR + MODEL_SUBDIR + 'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
            model.load_weights(bst_model_path)
            preds_test += model.predict(X_test, batch_size=batch_size) / RUNS_IN_FOLD
            del model
            gc.collect()
        labels_index_test = np.argmax(preds_test, axis=1)
        submit_in_fold = pd.DataFrame()
        submit_in_fold['fname'] = fname_test
        submit_in_fold['label'] = [LEGAL_LABELS[index] for index in labels_index_test]
        file_path = CV_DIR + SUBMIT_SUBDIR + 'submit_fold{0}_aug_{1}.csv'.format(fold, aug_type)
        submit_in_fold.to_csv(file_path, index=False)
    del fname_test
    del X_test
    gc.collect()
gc.collect()

## Voting by TTA result
def voting(votes):
    # if Counter(votes)['unknown']>=2:
    #     return 'unknown'
    return Counter(votes).most_common(1)[0][0]
# Boost original test result weight
ORIGINAL_WEIGHT = 1
for fold in range(K):
    fname_list = []
    vote_candidates = []
    for aug_type in aug_type_list:
        df = pd.read_csv(CV_DIR+SUBMIT_SUBDIR+'submit_fold{0}_aug_{1}.csv'.format(fold, aug_type), sep=',', index_col=False)
        fname_list = df['fname'].values
        if aug_type==ORIGINAL:
            for _ in range(ORIGINAL_WEIGHT):
                vote_candidates.append(list(df['label'].values))
        vote_candidates.append(list(df['label'].values))
    vote_candidates = np.array(vote_candidates).T
    st(context=21)
    voted_label = list(map(voting, vote_candidates))
    submit = pd.DataFrame()
    submit['fname'] = fname_list
    submit['label'] = voted_label
    path = CV_DIR + SUBMIT_SUBDIR + 'submit_fold{0}.csv'.format(fold)
    submit.to_csv(path, index=False)



##################################################
# Reproduce valid file and calculate CV acc score
##################################################
for fold in range(K):
    ## Prepare valid data in fold
    fname_valid, truth_valid, X_valid = get_valid_data(fold)
    preds_valid = np.zeros((X_valid.shape[0], n_classes))

    ## use trained model predict in fold
    for run in range(RUNS_IN_FOLD):
        model = get_model()
        print('Predict using trained model weights fold {0} run {1}'.format(fold, run))
        bst_model_path = CV_DIR + MODEL_SUBDIR + 'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
        model.load_weights(bst_model_path)

        preds_valid += model.predict(X_valid, batch_size=batch_size) / RUNS_IN_FOLD
        del model

    # produce valid file in fold
    labels_index_valid = np.argmax(preds_valid, axis=1)
    in_fold = pd.DataFrame()
    in_fold['fname'] = fname_valid
    in_fold['truth'] = truth_valid
    in_fold['preds'] = [LEGAL_LABELS[index] for index in labels_index_valid]
    acc_counts = 0
    for truth, preds in zip(in_fold['truth'], in_fold['preds']):
        acc_counts += 1 if truth == preds else 0
    acc_in_fold = acc_counts / len(in_fold['preds'])
    print('Fold %s valid accuracy : %.5f\n' % (fold, acc_in_fold))
    in_fold_valid_path = CV_DIR + VALID_SUBDIR + 'valid_by_{0}fold_acc{1}.csv'.format(fold, acc_in_fold)
    in_fold.to_csv(in_fold_valid_path, index=False)

    # release memory
    del X_valid
    gc.collect()

get_cv_acc_score(CV_DIR + VALID_SUBDIR)
