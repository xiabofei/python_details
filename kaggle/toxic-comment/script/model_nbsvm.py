# encoding=utf8
import argparse

import pandas as pd
import numpy as np

from data_split import label_candidates
from data_split import K
from comm_preprocessing import COMMENT_COL, ID_COL

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self

from ipdb import set_trace as st
import gc
data_split_dir = '../data/input/data_split/'
NUM_OF_LABEL = 6


def read_data_in_fold(k):
    df_trn = pd.read_csv(data_split_dir + '{0}_train.csv'.format(k)).fillna('unknown')
    df_val = pd.read_csv(data_split_dir + '{0}_valid.csv'.format(k)).fillna('unknown')
    df_trn[COMMENT_COL] = df_trn[COMMENT_COL].astype('str')
    df_val[COMMENT_COL] = df_val[COMMENT_COL].astype('str')
    print('train data in fold {0} : {1}'.format(k, len(df_trn.index)))
    print('valid data in fold {0} : {1}'.format(k, len(df_val.index)))
    return df_trn, df_val


def read_test_data():
    df_test = pd.read_csv('../data/input/test.csv').fillna('unknown')
    df_test[COMMENT_COL] = df_test[COMMENT_COL].astype('str')
    print('test data {0}'.format(len(df_test.index)))
    return df_test


def read_train_data():
    df_train = pd.read_csv('../data/input/train.csv').fillna('unknown')
    df_train[COMMENT_COL] = df_train[COMMENT_COL].astype('str')
    print('train data {0}'.format(len(df_train.index)))
    return df_train


def get_extractor(mode):
    if mode == 'word':
        extractor = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            strip_accents='unicode',
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1,
            analyzer='word_char',
        )
    return extractor

def conduct_transform(extractor, data):
    return extractor.transform(data)

def get_model():
    return NbSvmClassifier(C=4, dual=True, n_jobs=12)


def run_cv():
    # read whole train / test data for extractor
    df_train = read_train_data()
    df_test = read_test_data()
    X_test = df_test[COMMENT_COL].values
    id_test = df_test[ID_COL].values.tolist()

    print('Fitting word level features ... ')
    extractor_word = get_extractor('word')
    extractor_word.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))

    X_test_word = conduct_transform(extractor_word, X_test)

    for fold in range(K):
        # read in fold data
        df_trn, df_val = read_data_in_fold(fold)

        X_trn = df_trn[COMMENT_COL].values
        X_trn_word = conduct_transform(extractor_word, X_trn)
        y_trn = df_trn[label_candidates].values
        print('\nFold {0} train data shape {1} '.format(fold, X_trn.shape))

        X_val = df_val[COMMENT_COL].values
        X_val_word = conduct_transform(extractor_word, X_val)
        y_val = df_val[label_candidates].values
        id_val = df_val[ID_COL].values.tolist()
        print('Fold {0} valid data shape {1} '.format(fold, X_val.shape))

        # preds result array
        preds_test = np.zeros((X_test.shape[0], NUM_OF_LABEL))
        preds_valid = np.zeros((X_val.shape[0], NUM_OF_LABEL))

        models = []
        for idx, label in enumerate(label_candidates):
            print('\nFold {0} label {1}'.format(fold, label))
            model = get_model()
            print('   train')
            model.fit(x=X_trn_word, y=y_trn[:,idx])
            models.append(model)
            print('   predict valid')
            preds_valid[:,idx] = model.predict_proba(x=X_val_word)[:,1]

        # predict in fold
        print('Fold {0} predict test'.format(fold))
        for idx, model in enumerate(models):
            preds_test[:,idx] = model.predict_proba(x=X_test_word)[:,1]

        # record preds result
        preds_test = preds_test.T
        df_preds_test = pd.DataFrame()
        df_preds_test[ID_COL] = id_test
        for idx, label in enumerate(label_candidates):
            df_preds_test[label] = preds_test[idx]
        df_preds_test.to_csv('../data/output/preds/nbsvm/{0}fold_test.csv'.format(fold), index=False)

        preds_valid = preds_valid.T
        df_preds_val = pd.DataFrame()
        df_preds_val[ID_COL] = id_val
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid[idx]
        df_preds_val.to_csv('../data/output/preds/nbsvm/{0}fold_valid.csv'.format(fold), index=False)


if __name__ == '__main__':
    run_cv()
