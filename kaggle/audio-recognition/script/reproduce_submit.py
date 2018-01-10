# encoding=utf8

from model_fit import get_model, RUNS_IN_FOLD, batch_size
from model_fit import n_classes, LEGAL_LABELS
from data_split import K
import numpy as np
import pandas as pd
import pickle
import gc


CV_DIR = '../data/output/'
MODEL_SUBDIR = 'model/'
SUBMIT_SUBDIR = 'submit/'



def get_test_data():
    test_dir = '../data/input_backup/test_augmentation/'
    d_test = pickle.load(open(test_dir + 'test_original.pkl', 'rb'))
    fname_test, X_test = d_test['fname'], d_test['data']
    X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
    del d_test
    gc.collect()
    return fname_test, X_test


##################################################
# predict by average
##################################################
fname_test, X_test = get_test_data()
preds_test = np.zeros((X_test.shape[0], n_classes))
for fold in range(K):
    for run in range(RUNS_IN_FOLD):
        model = get_model()
        print('Predict using trained model weights fold {0} run {1}'.format(fold, run))
        bst_model_path = CV_DIR + MODEL_SUBDIR + 'nn_fold{0}_run{1}_spectrogram.h5'.format(fold, run)
        model.load_weights(bst_model_path)
        preds_test += model.predict(X_test, batch_size=512) / RUNS_IN_FOLD / K
        del model
        gc.collect()
labels_index_test = np.argmax(preds_test, axis=1)
submit_in_fold = pd.DataFrame()
submit_in_fold['fname'] = fname_test
submit_in_fold['label'] = [LEGAL_LABELS[index] for index in labels_index_test]
file_path = CV_DIR + SUBMIT_SUBDIR + 'submit_by_avg.csv'
submit_in_fold.to_csv(file_path, index=False)
del fname_test
del X_test
gc.collect()

