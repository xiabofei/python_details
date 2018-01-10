# encoding=utf8
from model_fit_enhance import AudioGenerator, n_classes, LEGAL_LABELS, get_model, RUNS_IN_FOLD, batch_size
from data_split_enhance import UNKNOWN_ENHANCE_RATE
from data_split import K
import pickle
import numpy as np
import pandas as pd
import gc

from ipdb import set_trace as st

CV_DIR = '../data/output/'
MODEL_SUBDIR = 'model/'
VALID_SUBDIR = 'valid/'
SUBMIT_SUBDIR = 'submit/'

USED_ENHANCE_RATE = 2
assert USED_ENHANCE_RATE<=UNKNOWN_ENHANCE_RATE, 'enhance rate setting wrong'

def get_valid_data(fold):
    print('\nPrepare valid data in fold {0}\n'.format(fold))
    valid_generator = AudioGenerator(
        root_dir='../data/input_backup/train_augmentation/',
        k=fold,
        batch_size=batch_size,
        train_or_valid='valid',
        enhance='enhance'
    )
    # prepare valid data
    X_valid, y_valid = valid_generator.ori_data['data'], np.array(valid_generator.ori_data['label'])
    X_valid = X_valid.reshape(tuple(list(X_valid.shape) + [1])).astype('float32')
    y_truth = [LEGAL_LABELS[index] for index in np.argmax(y_valid, axis=1)]
    del valid_generator
    gc.collect()
    return X_valid, y_truth

def reproduce_cv():
    total_counts = 0
    correct_counts = 0

    for fold in range(K):

        # Read valid data in fold
        X_valid, y_truth = get_valid_data(fold)
        total_counts += X_valid.shape[0]
        preds_valid = np.zeros((X_valid.shape[0], n_classes))

        # Average nn blending enhances
        for enhance_idx in range(USED_ENHANCE_RATE):
            for run in range(RUNS_IN_FOLD):
                model = get_model()
                print('Predict using trained model weights fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                bst_model_path = CV_DIR + MODEL_SUBDIR + \
                                 'enhance{0}/'.format(enhance_idx) + \
                                 'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
                model.load_weights(bst_model_path)
                preds_valid += model.predict(X_valid, batch_size=1024) / RUNS_IN_FOLD / USED_ENHANCE_RATE
                del model

        # Count correct
        y_preds = [LEGAL_LABELS[index] for index in np.argmax(preds_valid, axis=1)]
        for truth, preds in zip(y_truth, y_preds):
            correct_counts += 1 if truth == preds else 0

        # Free memory
        del X_valid
        gc.collect()
    print('Local CV acc : {0}'.format(correct_counts / total_counts))

def get_test_data():
    print('Prepare test data')
    test_dir = '../data/input_backup/test_augmentation/'
    d_test = pickle.load(open(test_dir + 'test_original.pkl', 'rb'))
    fname_test, X_test = d_test['fname'], d_test['data']
    X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
    del d_test
    gc.collect()
    return X_test, fname_test

def reproduce_enhance_submit(mode):

    # Read test data
    X_test, fname_test = get_test_data()

    # average blending mode
    if mode=='average':
        preds_test = np.zeros((X_test.shape[0], n_classes))
        # Average nn blending enhances
        for fold in range(K):
            for enhance_idx in range(USED_ENHANCE_RATE):
                for run in range(RUNS_IN_FOLD):
                    model = get_model()
                    print('Predict using trained model weights fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                    bst_model_path = CV_DIR + MODEL_SUBDIR + \
                                     'enhance{0}/'.format(enhance_idx) + \
                                     'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
                    model.load_weights(bst_model_path)
                    preds_test += model.predict(X_test, batch_size=1024) / RUNS_IN_FOLD / USED_ENHANCE_RATE / K
                    del model
                    gc.collect()

        # Produce blending submit file
        submit = pd.DataFrame()
        submit['fname'] = fname_test
        submit['label'] = [LEGAL_LABELS[index] for index in np.argmax(preds_test, axis=1)]
        submit.to_csv('../data/output/submit/submit_by_blending.csv', index=False)

    # voting ensemble mode
    if mode=='vote':
        # Average nn blending enhances
        for fold in range(K):
            preds_test = np.zeros((X_test.shape[0], n_classes))
            for enhance_idx in range(USED_ENHANCE_RATE):
                for run in range(RUNS_IN_FOLD):
                    model = get_model()
                    print('Predict using trained model weights fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                    bst_model_path = CV_DIR + MODEL_SUBDIR + \
                                     'enhance{0}/'.format(enhance_idx) + \
                                     'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
                    model.load_weights(bst_model_path)
                    preds_test += model.predict(X_test, batch_size=2048) / RUNS_IN_FOLD / USED_ENHANCE_RATE
                    del model
                    gc.collect()
            # create in fold submit file
            submit = pd.DataFrame()
            submit['fname'] = fname_test
            submit['label'] = [LEGAL_LABELS[index] for index in np.argmax(preds_test, axis=1)]
            submit.to_csv('../data/output/submit_enhance/submit_by_{0}fold.csv'.format(fold), index=False)

if __name__ == '__main__':
    reproduce_cv()
    mode = 'vote'
    reproduce_enhance_submit(mode)

