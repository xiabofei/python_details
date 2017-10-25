# encoding=utf8

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from evaluation import GiniEvaluation
from utils import ps_reg_03_recon
import gc
from collections import Counter
from ipdb import set_trace as st

import os

root_dir = '../../data/input/'

# load data from disk
train = pd.read_csv(os.path.join(root_dir, 'train.csv'), na_values=-1)
test = pd.read_csv(os.path.join(root_dir, 'test.csv'), na_values=-1)

# drop columns
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
print('After drop columns : train shape {0}, test shape {1}'.format(train.shape, test.shape))
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)

# drop outlier
train.drop(train.index[149161], axis=0, inplace=True)  # index
print('After drop rows : train shape {0}, test shape {1}'.format(train.shape, test.shape))

# save memory by change float64 to float32
for c in train.select_dtypes(include=['float64']).columns:
    train[c], test[c] = train[c].astype(np.float32), test[c].astype(np.float32)
# save memory by change int64 to int8
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c], test[c] = train[c].astype(np.int8), test[c].astype(np.int8)

# feature reconstruct
# train['ps_reg_F_cat'] = train['ps_reg_03'].apply(lambda x: ps_reg_03_recon(x)[0] if not np.isnan(x) else x)
# train['ps_reg_M_cat'] = train['ps_reg_03'].apply(lambda x: ps_reg_03_recon(x)[1] if not np.isnan(x) else x)
# test['ps_reg_F_cat'] = test['ps_reg_03'].apply(lambda x: ps_reg_03_recon(x)[0] if not np.isnan(x) else x)
# test['ps_reg_M_cat'] = test['ps_reg_03'].apply(lambda x: ps_reg_03_recon(x)[1] if not np.isnan(x) else x)

# prepare for train data
y = train['target']
train.drop(['id', 'target'], axis=1, inplace=True)
# prepare for submit data
sub = test['id'].to_frame()
sub['target'] = 0.0
test = test.drop(['id'], axis=1)

# Performing one hot encoding train and test together
combine = pd.concat([train, test], axis=0)
cat_features = [a for a in combine.columns if a.endswith('cat')]
for column in cat_features:
    temp = pd.get_dummies(pd.Series(combine[column]))
    combine = pd.concat([combine, temp], axis=1)
    combine = combine.drop([column], axis=1)
train = combine[:train.shape[0]]
test = combine[train.shape[0]:]
print('After one hot encoding : train shape {0}, test shape {1}'.format(train.shape, test.shape))

gc.collect()

X = train.values
y = y.values

gc.collect()


def cv_by_lgbm():
    params = {
        'metric': 'auc',
        'learning_rate': 0.02,
        'max_depth': 6,
        'num_leaves': 51,
        'min_data_in_leaf': 500,
        'max_bin': 10,
        'objective': 'binary',
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 10,
        'verbose': -1,
    }
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=np.random.randint(2017))
    folds = skf.split(X, y)
    dtrain = lgb.Dataset(data=X, label=y)
    num_boost_round = 2000
    bst = lgb.cv(
        params=params,
        train_set=dtrain,
        folds=folds,
        stratified=True,
        num_boost_round=num_boost_round,
        metrics=['auc'],
        feval=GiniEvaluation.gini_lgb,
        early_stopping_rounds=50,
        verbose_eval=20,
    )
    best_rounds = np.argmax(bst['gini-mean']) + 1
    best_val_score = np.max(bst['gini-mean'])
    print('Best gini_mean {0}, at round {1}'.format(best_val_score, best_rounds))
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' lgb kfold: {0}  of  {1} : '.format(i + 1, N))
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        model, eval_records = lgb.train(
            params=params,
            train_set=lgb.Dataset(data=X_train, label=y_train),
            valid_sets=lgb.Dataset(data=X_eval, label=y_eval),
            num_boost_round=best_rounds,
            feval=GiniEvaluation.gini_lgb,
            early_stopping_rounds=50,
            verbose_eval=20,
        )
        gini_scores = [item[1][2] for item in eval_records if item[1][1] == 'gini']
        num_iteration = np.argmax(gini_scores) + 1
        best_score = np.max(gini_scores)
        print('model best iteration {0} with gini score {1}'.format(num_iteration, best_score))
        sub['target'] += model.predict(test.values, num_iteration=num_iteration)
        model.save_model('../../data/model/lgbm_{0}'.format(i), num_iteration)
    print('{0} of models ensemble'.format(N))
    sub['target'] = sub['target'] / N
    sub.to_csv('../../data/output/sub_lgbm.csv', index=False, float_format='%.7f')
    print('LightGBM done')


def cv_by_xgb():
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'max_depth': 6,
        'min_child_weight': 8,
        'nthread': -1,
        'silent': 1,
        'alpha': 0.001,
        'gamma': 0.01,
        'seed': 2016
    }
    ## cross validation
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=np.random.randint(2017))
    folds = skf.split(X, y)
    dtrain = xgb.DMatrix(data=X, label=y)
    num_boost_round = 2000
    bst = xgb.cv(
        params,
        dtrain,
        num_boost_round,
        N,
        feval=GiniEvaluation.gini_xgb,
        maximize=True,
        metrics=['auc'],
        folds=folds,
        early_stopping_rounds=50,
        verbose_eval=10
    )
    best_rounds = np.argmax(bst['test-gini-mean'])
    train_score = bst['train-gini-mean'][best_rounds]
    best_val_score = bst['test-gini-mean'][best_rounds]
    print('Best train_gini_mean: %.5f, val_gini_mean: %.5f at round %d.' % \
          (train_score, best_val_score, best_rounds))
    ## out-of-fold prediction
    dtest = xgb.DMatrix(data=test.values)
    for trn_idx, val_idx in skf.split(X, y):
        trn_x, val_x = X[trn_idx], X[val_idx]
        trn_y, val_y = y[trn_idx], y[val_idx]
        dtrn = xgb.DMatrix(data=trn_x, label=trn_y)
        dval = xgb.DMatrix(data=val_x, label=val_y)
        # train model
        cv_model = xgb.train(
            params=params,
            dtrain=dtrn,
            evals=[(dval, 'val')],
            num_boost_round=best_rounds,
            feval=GiniEvaluation.gini_xgb,
            maximize=True,
            early_stopping_rounds=50,
            verbose_eval=10,
        )
        sub['target'] += cv_model.predict(dtest, ntree_limit=best_rounds)
    print('{0} of models ensemble'.format(N))
    sub['target'] = sub['target'] / N
    sub.to_csv('../../data/output/sub_xgb.csv', index=False, float_format='%.7f')
    print('XGBoost done')


cv_by_lgbm()
# cv_by_xgb()
