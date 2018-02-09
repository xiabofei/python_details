# encoding=utf8
import argparse

import pandas as pd
import numpy as np

from data_split import label_candidates
from data_split import K
from comm_preprocessing import data_comm_preprocessed_dir
from comm_preprocessing import COMMENT_COL, ID_COL
from comm_preprocessing import toxicIndicator_transformers

from sklearn.feature_extraction.text import CountVectorizer

from ipdb import set_trace as st
import gc

NUM_OF_LABEL = 6

max_features = 80000
min_df = 10
ngram_range = (1, 2)
C = 0.05 + 0.1 * np.random.random()


def read_data_in_fold(k):
    df_trn = pd.read_csv(data_comm_preprocessed_dir + '{0}_train.csv'.format(k))
    df_val = pd.read_csv(data_comm_preprocessed_dir + '{0}_valid.csv'.format(k))
    print('train data in fold {0} : {1}'.format(k, len(df_trn.index)))
    print('valid data in fold {0} : {1}'.format(k, len(df_val.index)))
    return df_trn, df_val


def read_test_data():
    df_test = pd.read_csv(data_comm_preprocessed_dir + 'test.csv')
    df_test[COMMENT_COL] = df_test[COMMENT_COL].astype('str')
    print('test data {0}'.format(len(df_test.index)))
    return df_test


def read_train_data():
    df_train = pd.read_csv(data_comm_preprocessed_dir + 'train.csv')
    df_train[COMMENT_COL] = df_train[COMMENT_COL].astype('str')
    print('train data {0}'.format(len(df_train.index)))
    return df_train


def get_extractor(mode):
    assert mode in ['word', 'char']
    if mode == 'word':
        extractor = CountVectorizer(
            max_df=0.999, min_df=min_df,
            max_features=max_features, ngram_range=ngram_range,
            analyzer='word', binary=True, lowercase=True
        )
    if mode == 'char':
        extractor = CountVectorizer(
            max_df=0.999, min_df=min_df,
            max_features=max_features, ngram_range=ngram_range,
            analyzer='char', binary=True, lowercase=True
        )
    return extractor


def run_cv():
    # read whole train / test data for tokenizer
    df_train = read_train_data()
    df_test = read_test_data()
    id_test = df_test[ID_COL].values.tolist()

    extractor = get_extractor('word')
    extractor.fit(pd.concat((df_train.ix[:, COMMENT_COL], df_test.ix[:, COMMENT_COL])))

    st(context=21)
    '''
    for fold in range(K):
        # read in fold data
        df_trn, df_val = read_data_in_fold(fold)

        X_trn = df_trn[COMMENT_COL].values.tolist()
        y_trn = df_trn[label_candidates].values
        print('Fold {0} train data shape {1} '.format(fold, X_trn.shape))

        X_val = df_val[COMMENT_COL].values.tolist()
        y_val = df_val[label_candidates].values
        id_val = df_val[ID_COL].values.tolist()
        print('Fold {0} valid data shape {1} '.format(fold, X_val.shape))

        # preds result array
        preds_test = np.zeros((X_test.shape[0], NUM_OF_LABEL))
        preds_valid = np.zeros((X_val.shape[0], NUM_OF_LABEL))

        # record preds result
        preds_test = preds_test.T
        df_preds_test = pd.DataFrame()
        df_preds_test[ID_COL] = id_test
        for idx, label in enumerate(label_candidates):
            df_preds_test[label] = preds_test[idx]
        df_preds_test.to_csv('../data/output/preds/lr/{0}fold_test.csv'.format(fold), index=False)

        preds_valid = preds_valid.T
        df_preds_val = pd.DataFrame()
        df_preds_val[ID_COL] = id_val
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid[idx]
        df_preds_val.to_csv('../data/output/preds/lr/{0}fold_valid.csv'.format(fold), index=False)
    '''


if __name__ == '__main__':
    run_cv()
