# encoding=utf8
import argparse

import pandas as pd
import numpy as np

import lightgbm as lgbm

from data_split import label_candidates
from data_split import K
# from comm_preprocessing import data_comm_preprocessed_dir
from comm_preprocessing_gbdt import data_comm_preprocessed_dir
from comm_preprocessing import COMMENT_COL, ID_COL

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from lgbm_params import params_groups
from scipy.sparse import hstack

from ipdb import set_trace as st
import gc

n_classes = len(label_candidates)

NAN_BLANK = 'nanblank'
NUM_OF_LABEL = 6

# word level params
max_features_word = 10000
ngram_range_word = (1, 2)

# char level params
max_features_char = 20000
ngram_range_char = (2, 4)

# lgbm params
num_boost_round=2000
early_stopping_rounds=100
verbose_eval=30

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


def get_extractor_word():
    extractor = TfidfVectorizer(
        max_df=0.995, min_df=10,
        max_features=max_features_word, ngram_range=ngram_range_word,
        analyzer='word', binary=True, lowercase=True
    )
    return extractor

def get_extractor_char():
    extractor = TfidfVectorizer(
        max_df=0.995, min_df=10,
        max_features=max_features_word, ngram_range=ngram_range_word,
        analyzer='char', binary=True, lowercase=True
    )
    return extractor

def conduct_transform(extractor, data):
    return extractor.transform(data)

def get_gbdt_params(label_name):
    assert label_name in label_candidates, 'wrong label name'
    return params_groups[label_name]

def get_idx_trn_val(ids):
    # get each fold trn/val data ids [(trn, val),...]
    ids_trn_val = []
    for k in range(K):
        df_trn = pd.read_csv(data_comm_preprocessed_dir + '{0}_train.csv'.format(k))
        df_val = pd.read_csv(data_comm_preprocessed_dir + '{0}_valid.csv'.format(k))
        ids_trn = df_trn[ID_COL].values.tolist()
        ids_val = df_val[ID_COL].values.tolist()
        ids_trn_val.append((ids_trn, ids_val))

    # get each fold trn/val data index of array
    idx_trn_val = []
    for ids_trn, ids_val in ids_trn_val:
        ids_trn = set(ids_trn)
        ids_val = set(ids_val)
        idx_trn = [idx for idx, id in enumerate(ids) if id in ids_trn]
        idx_val = [idx for idx, id in enumerate(ids) if id in ids_val]
        idx_trn_val.append((idx_trn, idx_val))

    return idx_trn_val

def run_cv():
    # read train/test data
    df_train = read_train_data()
    X_train = df_train[COMMENT_COL].values
    id_train = df_train[ID_COL].values.tolist()
    y_true = df_train[label_candidates].values

    df_test = read_test_data()
    X_test = df_test[COMMENT_COL].values
    id_test = df_test[ID_COL].values.tolist()

    # extract n-gram word level feature
    print('Extracting n-gram features')
    extractor_word = get_extractor_word()
    extractor_word.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))
    X_train_word = conduct_transform(extractor_word, X_train)
    X_test_word = conduct_transform(extractor_word, X_test)
    print('Train word data shape : {0}'.format(X_train_word.shape))
    print('Test word data shape : {0}'.format(X_test_word.shape))

    # extract n-gram char level feature
    extractor_char = get_extractor_char()
    extractor_char.fit(pd.concat((df_train.loc[:, COMMENT_COL], df_test.loc[:, COMMENT_COL])))
    X_train_char = conduct_transform(extractor_char, X_train)
    X_test_char = conduct_transform(extractor_char, X_test)
    print('Train char data shape : {0}'.format(X_train_char.shape))
    print('Test char data shape : {0}'.format(X_test_char.shape))

    # combine word and char
    X_train_word_char = hstack([X_train_word, X_train_char])
    X_test_word_char = hstack([X_test_word, X_test_char])
    X_train_word_char = X_train_word_char.tocsr()
    X_test_word_char = X_test_word_char.tocsr()
    print('Train word char data shape : {0}'.format(X_train_word_char.shape))
    print('Test word char data shape : {0}'.format(X_test_word_char.shape))


    # get idx of trn/val for each fold
    print('Getting array index of train/valid for each fold')
    idx_trn_val = get_idx_trn_val(id_train)

    # preds on test/valid
    preds_test = np.zeros((X_test_word_char.shape[0], n_classes))
    preds_valid = np.zeros((X_train_word_char.shape[0], n_classes))

    # cv and train/predict
    for label_col, label_name in enumerate(label_candidates):
        print('\nlabel column : {0}'.format(label_col))
        print('label name : {0}'.format(label_name))
        # cv best boost rounds
        train_set = lgbm.Dataset(data=X_train_word_char, label=y_true[:, label_col])
        params = get_gbdt_params(label_name)
        print('lgbm params : {0}'.format(params))
        hist = lgbm.cv(
            params=params,
            train_set=train_set,
            folds=idx_trn_val,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            metrics=['auc'],
            verbose_eval=verbose_eval,
        )
        bst_boost_rounds = np.argmax(hist['auc-mean']) + 1
        bst_acc_score = np.max(hist['auc-mean'])
        print('label column : {0}'.format(label_col))
        print('label name : {0}'.format(label_name))
        print('best boost rounds {0}, best auc score {1}'.format(bst_boost_rounds, bst_acc_score))

        # oof train and predict
        for fold, (idx_trn, idx_val) in enumerate(idx_trn_val):
            print('fold {0}'.format(fold))
            print('    train')
            dat_trn = lgbm.Dataset(data=X_train_word_char[idx_trn, :], label=y_true[idx_trn, label_col])
            model = lgbm.train(
                params=params,
                train_set=dat_trn,
                num_boost_round=bst_boost_rounds,
                verbose_eval=verbose_eval,
            )
            print('    predict')
            preds_valid[idx_val, label_col] = model.predict(
                data=X_train_word_char[idx_val,:], num_iteration=bst_boost_rounds)
            preds_test[:,label_col] = model.predict(data=X_test_word_char, num_iteration=bst_boost_rounds) / K
            del model

    # ensemble cv score
    score = roc_auc_score(y_true=y_true, y_score=preds_valid, average='macro')
    print('\ncv score : {0}'.format(score))
    # divide data for ensemble
    for fold, (_, idx_val) in enumerate(idx_trn_val):
        preds_valid_fold = preds_valid[idx_val,:].T
        df_preds_val = pd.DataFrame()
        idx_val_set = set(idx_val)
        df_preds_val[ID_COL] = [ id for idx, id in enumerate(id_train) if idx in idx_val_set ]
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid_fold[idx]
        df_preds_val.to_csv('../data/output/preds/gbdt/{0}fold_valid.csv'.format(fold), index=False)

    # record ensemble result
    preds_test = preds_test.T
    df_preds_test = pd.DataFrame()
    df_preds_test[ID_COL] = id_test
    for idx, label in enumerate(label_candidates):
        df_preds_test[label] = preds_test[idx]
    df_preds_test.to_csv('../data/output/preds/gbdt/avg_submit.csv', index=False)

if __name__ == '__main__':
    run_cv()
