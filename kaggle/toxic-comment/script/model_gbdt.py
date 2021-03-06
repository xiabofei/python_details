# encoding=utf8
import argparse

import pandas as pd
import numpy as np

from data_split import label_candidates
from data_split import K
# from comm_preprocessing import data_comm_preprocessed_dir
from comm_preprocessing import COMMENT_COL, ID_COL
from comm_preprocessing import toxicIndicator_transformers
from comm_preprocessing_heavy import data_comm_preprocessed_heavy_dir

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack

from ipdb import set_trace as st
import gc

NAN_BLANK = 'nanblank'
NUM_OF_LABEL = 6

max_features_word = 50000
max_features_char = 50000
min_df = 10
ngram_range_word = (1, 3)
ngram_range_char = (2, 5)


def read_data_in_fold(k):
    # df_trn = pd.read_csv(data_comm_preprocessed_dir + '{0}_train.csv'.format(k))
    # df_val = pd.read_csv(data_comm_preprocessed_dir + '{0}_valid.csv'.format(k))
    df_trn = pd.read_csv(data_comm_preprocessed_heavy_dir + '{0}_train.csv'.format(k))
    df_val = pd.read_csv(data_comm_preprocessed_heavy_dir + '{0}_valid.csv'.format(k))
    df_trn[COMMENT_COL] = df_trn[COMMENT_COL].astype('str')
    df_val[COMMENT_COL] = df_val[COMMENT_COL].astype('str')
    print('train data in fold {0} : {1}'.format(k, len(df_trn.index)))
    print('valid data in fold {0} : {1}'.format(k, len(df_val.index)))
    return df_trn, df_val


def read_test_data():
    # df_test = pd.read_csv(data_comm_preprocessed_dir + 'test.csv')
    df_test = pd.read_csv(data_comm_preprocessed_heavy_dir + 'test.csv')
    df_test[COMMENT_COL] = df_test[COMMENT_COL].astype('str')
    print('test data {0}'.format(len(df_test.index)))
    return df_test


def read_train_data():
    # df_train = pd.read_csv(data_comm_preprocessed_dir + 'train.csv')
    df_train = pd.read_csv(data_comm_preprocessed_heavy_dir + 'train.csv')
    df_train[COMMENT_COL] = df_train[COMMENT_COL].astype('str')
    print('train data {0}'.format(len(df_train.index)))
    return df_train


def get_extractor(mode):
    assert mode in ['word', 'char']
    if mode == 'word':
        extractor = TfidfVectorizer(
            max_df=0.995, min_df=10,
            max_features=max_features_word, ngram_range=ngram_range_word,
            analyzer='word', binary=True, lowercase=True
        )
    if mode == 'char':
        extractor = TfidfVectorizer(
            max_df=0.995, min_df=10,
            max_features=max_features_char, ngram_range=ngram_range_char,
            analyzer='char', binary=True, lowercase=True
        )
    return extractor


def conduct_transform(extractor, data):
    return extractor.transform(data)

def get_model():
    return LogisticRegression(solver='sag', n_jobs=12, verbose=1, tol=5e-6)


def run_cv():
    # read whole train / test data for extractor
    df_train = read_train_data()
    df_test = read_test_data()
    X_test = df_test[COMMENT_COL].values
    id_test = df_test[ID_COL].values.tolist()

    extractor_word = get_extractor('word')
    extractor_word.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))

    st(context=21)

    extractor_char = get_extractor('char')
    extractor_char.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))

    st(context=21)

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
        df_preds_test.to_csv('../data/output/preds/lr/{0}fold_test.csv'.format(fold), index=False)

        preds_valid = preds_valid.T
        df_preds_val = pd.DataFrame()
        df_preds_val[ID_COL] = id_val
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid[idx]
        df_preds_val.to_csv('../data/output/preds/lr/{0}fold_valid.csv'.format(fold), index=False)


if __name__ == '__main__':
    run_cv()
