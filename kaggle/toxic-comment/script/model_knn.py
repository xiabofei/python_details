# encoding=utf8
import argparse

import pandas as pd
import numpy as np

from data_split import label_candidates
from data_split import K
from comm_preprocessing import COMMENT_COL, ID_COL

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import SnowballStemmer

from scipy.sparse import hstack

from ipdb import set_trace as st
import gc
data_split_dir = '../data/input/data_split/'
NUM_OF_LABEL = 6


def read_data_in_fold(k):
    df_trn = pd.read_csv(data_split_dir + '{0}_train.csv'.format(k)).fillna(' ')
    df_val = pd.read_csv(data_split_dir + '{0}_valid.csv'.format(k)).fillna(' ')
    df_trn[COMMENT_COL] = df_trn[COMMENT_COL].astype('str')
    df_val[COMMENT_COL] = df_val[COMMENT_COL].astype('str')
    print('train data in fold {0} : {1}'.format(k, len(df_trn.index)))
    print('valid data in fold {0} : {1}'.format(k, len(df_val.index)))
    return df_trn, df_val


def read_test_data():
    df_test = pd.read_csv('../data/input/test.csv').fillna(' ')
    df_test[COMMENT_COL] = df_test[COMMENT_COL].astype('str')
    print('test data {0}'.format(len(df_test.index)))
    return df_test


def read_train_data():
    df_train = pd.read_csv('../data/input/train.csv').fillna(' ')
    df_train[COMMENT_COL] = df_train[COMMENT_COL].astype('str')
    print('train data {0}'.format(len(df_train.index)))
    return df_train


english_stemmer = SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def get_extractor(mode):
    if mode == 'word':
        extractor = StemmedTfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            max_features=10000)
    if mode == 'char':
        extractor = StemmedTfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(1, 5),
            max_features=30000)
    return extractor


def conduct_transform(extractor, data):
    return extractor.transform(data)

def get_model():
    return KNeighborsClassifier(n_jobs=8)


def run_cv():
    # read whole train / test data for extractor
    df_train = read_train_data()
    df_test = read_test_data()
    X_test = df_test[COMMENT_COL].values
    id_test = df_test[ID_COL].values.tolist()

    print('Fitting word level features ... ')
    extractor_word = get_extractor('word')
    extractor_word.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))

    print('Fitting char level features ... ')
    extractor_char = get_extractor('char')
    extractor_char.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))

    X_test_word = conduct_transform(extractor_word, X_test)
    X_test_char = conduct_transform(extractor_char, X_test)
    X_test_all = hstack([X_test_word, X_test_char])

    for fold in range(K):
        # read in fold data
        df_trn, df_val = read_data_in_fold(fold)

        X_trn = df_trn[COMMENT_COL].values
        X_trn_word = conduct_transform(extractor_word, X_trn)
        X_trn_char = conduct_transform(extractor_char, X_trn)
        X_trn_all = hstack([X_trn_word, X_trn_char])
        y_trn = df_trn[label_candidates].values
        print('\nFold {0} train data shape {1} '.format(fold, X_trn.shape))

        X_val = df_val[COMMENT_COL].values
        X_val_word = conduct_transform(extractor_word, X_val)
        X_val_char = conduct_transform(extractor_char, X_val)
        X_val_all = hstack([X_val_word, X_val_char])
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
            model.fit(X=X_trn_all, y=y_trn[:,idx])
            models.append(model)
            print('   predict valid')
            preds_valid[:,idx] = model.predict_proba(X=X_val_all)[:,1]

        # predict in fold
        print('Fold {0} predict test'.format(fold))
        for idx, model in enumerate(models):
            preds_test[:,idx] = model.predict_proba(X=X_test_all)[:,1]

        # record preds result
        preds_test = preds_test.T
        df_preds_test = pd.DataFrame()
        df_preds_test[ID_COL] = id_test
        for idx, label in enumerate(label_candidates):
            df_preds_test[label] = preds_test[idx]
        df_preds_test.to_csv('../data/output/preds/knn/{0}fold_test.csv'.format(fold), index=False)

        preds_valid = preds_valid.T
        df_preds_val = pd.DataFrame()
        df_preds_val[ID_COL] = id_val
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid[idx]
        df_preds_val.to_csv('../data/output/preds/knn/{0}fold_valid.csv'.format(fold), index=False)


if __name__ == '__main__':
    run_cv()
