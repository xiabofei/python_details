# encoding=utf8
from model_fit_enhance import AudioGenerator, n_classes, LEGAL_LABELS, get_model, RUNS_IN_FOLD, batch_size
from data_split_enhance import UNKNOWN_ENHANCE_RATE
from data_split import K
import pickle
import numpy as np
import pandas as pd
import gc
from process_test_tta import TTA_RATE
from ipdb import set_trace as st
import argparse

CV_DIR = '../data/output/'
MODEL_SUBDIR = 'model/'
VALID_SUBDIR = 'valid/'
SUBMIT_SUBDIR = 'submit/'

PREDS_TEST_DIR = '../data/output/preds_test/'
PREDS_VALID_DIR = '../data/output/preds_valid/'

use_enhance_rate = UNKNOWN_ENHANCE_RATE
assert use_enhance_rate<=UNKNOWN_ENHANCE_RATE, 'enhance rate setting wrong'

def record_preds(obj, path):
    pickle.dump(obj=obj, file=open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def get_test_data(test_file):
    print('Prepare test data')
    test_dir = '../data/input_backup/test_augmentation/'
    d_test = pickle.load(open(test_dir + test_file, 'rb'))
    fname_test, X_test = d_test['fname'], d_test['data']
    X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
    del d_test
    gc.collect()
    return X_test, fname_test

def reproduce_enhance_submit(mode, test_file, tta):
    # Read test data
    X_test, fname_test = get_test_data(test_file)
    # average blending mode
    if mode=='average':
        preds_test = np.zeros((X_test.shape[0], n_classes))
        # Average nn blending enhances
        for fold in range(K):
            for enhance_idx in range(use_enhance_rate):
                for run in range(RUNS_IN_FOLD):
                    model = get_model()
                    print('Predict using trained model weights fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                    bst_model_path = CV_DIR + MODEL_SUBDIR + \
                                     'enhance{0}/'.format(enhance_idx) + \
                                     'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
                    model.load_weights(bst_model_path)
                    tmp_preds = model.predict(X_test, batch_size=1024)

                    # record test preds numpy
                    tmp_path = PREDS_TEST_DIR + 'fold{0}/enhance{1}/tta{2}/runs{3}_test.pkl'.format(fold, enhance_idx, tta, run)
                    record_preds(tmp_preds, tmp_path)

                    preds_test += tmp_preds  / RUNS_IN_FOLD / use_enhance_rate / K

                    del tmp_preds
                    del model
                    gc.collect()

        # Produce blending submit file
        submit = pd.DataFrame()
        submit['fname'] = fname_test
        submit['label'] = [LEGAL_LABELS[index] for index in np.argmax(preds_test, axis=1)]
        submit.to_csv('../data/output/submit/submit_by_blending.csv', index=False)

    # voting ensemble mode
    if mode=='vote':
        for fold in range(K):
            preds_test = np.zeros((X_test.shape[0], n_classes))
            for enhance_idx in range(use_enhance_rate):
                for run in range(RUNS_IN_FOLD):
                    model = get_model()
                    print('Predict using trained model weights fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                    bst_model_path = CV_DIR + MODEL_SUBDIR + \
                                     'enhance{0}/'.format(enhance_idx) + \
                                     'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
                    model.load_weights(bst_model_path)
                    tmp_preds = model.predict(X_test, batch_size=1024)

                    # record test preds numpy
                    tmp_path = PREDS_TEST_DIR + 'fold{0}/enhance{1}/tta{2}/runs{3}_test.pkl'.format(fold, enhance_idx, tta, run)
                    record_preds(tmp_preds, tmp_path)

                    preds_test += tmp_preds / RUNS_IN_FOLD / use_enhance_rate
                    del model
                    gc.collect()
            # create in fold submit file
            submit = pd.DataFrame()
            submit['fname'] = fname_test
            submit['label'] = [LEGAL_LABELS[index] for index in np.argmax(preds_test, axis=1)]
            submit.to_csv('../data/output/submit_enhance/tta{0}/submit_by_{1}fold.csv'.format(tta, fold), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tta', type=str, default='0', help='which tta')
    FLAGS, _ = parser.parse_known_args()
    mode = 'vote'
    t = FLAGS.tta
    reproduce_enhance_submit(mode, 'test_{0}.pkl'.format(t), t)

