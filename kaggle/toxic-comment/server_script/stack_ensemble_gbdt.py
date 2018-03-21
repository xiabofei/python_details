# encoding=utf8
from data_split import K
from data_split import label_candidates
from comm_preprocessing import ID_COL
import pandas as pd
import numpy as np
from numpy import hstack
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
import gc

from ipdb import set_trace as st

n_classes = len(label_candidates)


stack_ensemble_materials = [
    '../data/output/stack_ensemble/rnn/',
    #'../data/output/stack_ensemble/fasttext_deep_gru/',
    # '../data/output/stack_ensemble/skip_gru/',
    # '../data/output/stack_ensemble/maxpool_gru/',
    # '../data/output/stack_ensemble/gru_conv1d/',
    '../data/output/stack_ensemble/cnn/',
    '../data/output/stack_ensemble/nbsvm/',
    '../data/output/stack_ensemble/lr/',
    # '../data/output/stack_ensemble/gbdt/',
]

EPS = 1e-8

def get_ensemble_inputShape():
    ret = (len(stack_ensemble_materials) * n_classes,)
    return ret

def get_gbdt_params():
    '''stack ensemble by gbdt model
    '''
    params = {
        'objective': 'binary',
        'learning_rate': 0.1,
        'max_depth': 3,
        'num_leaves' : 20,
        'feature_fraction': 0.71,
        'bagging_fraction': 0.78,
        'bagging_freq': 5,
        'min_child_weight' : 10,
        # 'min_split_gain' : 1,
        # 'reg_alpha' : 0,
        # 'reg_lambda' : 1,
        'max_bin': 255,
        'nthread': 12,
        'verbose': 0,
    }
    return params

def re_range(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + EPS)
    data = (data - 0.5) * 2
    return data

def get_data():
    # get test submit data
    test_submit = []
    id_test = []
    for solo_path in stack_ensemble_materials:
        df = pd.read_csv(solo_path + 'avg_submit.csv')
        preds = df[label_candidates].values
        id_test = df[ID_COL].values.tolist()
        test_submit.append(preds)
    test_submit = hstack(test_submit)
    # test_submit = re_range(test_submit)

    # get id and true label
    df_train = pd.read_csv('../data/input/train.csv')
    ids = df_train[ID_COL].values.tolist()
    y_true = df_train[label_candidates].values

    # get all model oof valid preds score
    preds_score = []
    fold_ids = dict()  # get ids in each fold
    for solo_path in stack_ensemble_materials:
        id_preds = {}
        for k in range(K):
            df = pd.read_csv(solo_path + '{0}fold_valid.csv'.format(k))
            id_list = df[ID_COL].values.tolist()
            fold_ids[k] = id_list
            preds_list = df[label_candidates].values.tolist()
            for id, preds in zip(id_list, preds_list):
                id_preds[id] = preds
        y_score = np.array([id_preds[id] for id in ids])
        preds_score.append(y_score)
    preds_score = hstack(np.array(preds_score))
    # preds_score = re_range(preds_score)

    idx_trn_val = []
    for k in range(K):
        ids_val = set(fold_ids[k])
        ids_trn = []
        for key in fold_ids.keys():
            if key == k:
                continue
            ids_trn += fold_ids[key]
        ids_trn = set(ids_trn)
        idx_trn = [idx for idx, id in enumerate(ids) if id in ids_trn]
        idx_val = [idx for idx, id in enumerate(ids) if id in ids_val]
        idx_trn_val.append((idx_trn, idx_val))
    return test_submit, id_test, preds_score, y_true, idx_trn_val, fold_ids,


if __name__ == '__main__':
    test_submit, id_test, preds_score, y_true, idx_trn_val, fold_ids = get_data()

    preds_test = np.zeros((test_submit.shape[0], n_classes))
    preds_valid = np.zeros((y_true.shape[0], n_classes))

    params = get_gbdt_params()

    for label_col, label_name in enumerate(label_candidates):
        print('label column : {0}'.format(label_col))
        print('label name : {0}'.format(label_name))
        # cv best boost rounds
        train_set = lgbm.Dataset(data=preds_score, label=y_true[:, label_col])
        hist = lgbm.cv(
            params=params,
            train_set=train_set,
            folds=idx_trn_val,
            num_boost_round=2000,
            early_stopping_rounds=50,
            metrics=['auc'],
            verbose_eval=10,
        )
        bst_boost_rounds = np.argmax(hist['auc-mean']) + 1
        bst_acc_score = np.max(hist['auc-mean'])
        print('label column : {0}'.format(label_col))
        print('label name : {0}'.format(label_name))
        print('best boost rounds {0}, best auc score {1}'.format(bst_boost_rounds, bst_acc_score))

        # oof train and predict
        for fold, (idx_trn, idx_val) in enumerate(idx_trn_val):
            print('oof train for label : {0}'.format(label_name))
            dat_trn = lgbm.Dataset(data=preds_score[idx_trn, :], label=y_true[idx_trn, label_col])
            model = lgbm.train(
                params=params,
                train_set=dat_trn,
                num_boost_round=bst_boost_rounds,
                verbose_eval=10,
            )
            print('oof predict for label : {0}'.format(label_name))
            preds_test[:,label_col] = model.predict(data=test_submit, num_iteration=bst_boost_rounds) / K
            preds_valid[idx_val, label_col] = model.predict(
                data=preds_score[idx_val,:], num_iteration=bst_boost_rounds)
            del model

    # ensemble cv score
    score = roc_auc_score(y_true=y_true, y_score=preds_valid, average='macro')
    print('\ncv score : {0}'.format(score))
    # record ensemble result
    preds_test = preds_test.T
    df_preds_test = pd.DataFrame()
    df_preds_test[ID_COL] = id_test
    for idx, label in enumerate(label_candidates):
        df_preds_test[label] = preds_test[idx]
    df_preds_test.to_csv('../data/output/stack_ensemble/gbdt_ensemble.csv', index=False)
