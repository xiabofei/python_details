# encoding=utf8

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from evaluation import GiniEvaluation
import gc
from ipdb import set_trace as st

import os

root_dir = '../../data/input/'

# load data from disk
train = pd.read_csv(os.path.join(root_dir, 'train.csv'), na_values=-1)
test = pd.read_csv(os.path.join(root_dir, 'test.csv'), na_values=-1)

# drop columns
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)
print('After drop columns : train shape {0}, test shape {1}'.format(train.shape, test.shape))

# save memory by change float64 to float32
for c in train.select_dtypes(include=['float64']).columns:
    train[c], test[c] = train[c].astype(np.float32), test[c].astype(np.float32)
# save memory by change int64 to int8
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c], test[c] = train[c].astype(np.int8), test[c].astype(np.int8)

# prepare cross validation data
X = train.drop(['id', 'target'], axis=1)
y = train['target']
# prepare submit data
sub = test['id'].to_frame()
sub['target'] = 0.0
feature_cols = X.columns

X = X.values
y = y.values

gc.collect()


def cv_by_lgbm():
    params = {
        'metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 60,
        'min_data_in_leaf': 500,
        'max_bin': 10,
        'objective': 'binary',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 10,
        # indicate logging level : -1 make CV process silence...
        'verbose': -1,
    }
    N = 10
    skf = StratifiedKFold(n_splits=N, shuffle=True)
    folds = skf.split(X, y)
    dtrain = lgb.Dataset(data=X, label=y)
    num_boost_round = 1500
    learning_rate = [0.03] * 100 + [0.02] * 200 + [0.005] * 1200
    max_depth = [7] * 100 + [6] * 200 + [5] * 1200
    # feature_fraction = [0.5]*200+[0.5]*600+[0.3]*1200
    bst = lgb.cv(
        params=params,
        train_set=dtrain,
        folds=folds,
        stratified=True,
        num_boost_round=num_boost_round,
        metrics=['auc'],
        feval=GiniEvaluation.gini_lgb,
        early_stopping_rounds=100,
        verbose_eval=10,
        seed=2017,
        callbacks=[
            lgb.reset_parameter(
                learning_rate=learning_rate,
                max_depth=max_depth,
                # feature_fraction=feature_fraction
            )]
    )
    return bst


def train_single_lgbm(kfolds):
    params = {
        'metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 60,
        'min_data_in_leaf': 500,
        'max_bin': 10,
        'objective': 'binary',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 10,
        'verbose': -1,
    }
    # params for test pipeline
    num_boost_round = 50
    learning_rate = [0.3] * 50
    max_depth = [7] * 50
    # params for submit
    # num_boost_round =2000
    # learning_rate = [0.03]*100+[0.02]*200+[0.005]*1700
    # max_depth = [7]*100 + [6]*200 + [5]*1700
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=2017)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' lgb kfold: {}  of  {} : '.format(i + 1, kfolds))
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        model = lgb.train(
            params=params,
            train_set=lgb.Dataset(data=X_train, label=y_train),
            valid_sets=lgb.Dataset(data=X_eval, label=y_eval),
            num_boost_round=num_boost_round,
            feval=GiniEvaluation.gini_lgb,
            early_stopping_rounds=100,
            verbose_eval=10,
            callbacks=[lgb.reset_parameter(learning_rate=learning_rate, max_depth=max_depth)]
        )
        yield model


def predict_by_ensemble_lgbm(number_of_lgbm):
    for i, model in enumerate(train_single_lgbm(number_of_lgbm)):
        print('model best iteration {0}'.format(model.best_iteration))
        if model.best_iteration==0:
            sub['target'] += model.predict(test[feature_cols].values, num_iteration=model.current_iteration())
        else:
            sub['target'] += model.predict(test[feature_cols].values, num_iteration=model.best_iteration)
        print('FINISH TRAINING {0} lgbm'.format(i+1))
    sub['target'] = sub['target'] / number_of_lgbm
    sub.to_csv('../../data/output/sub.csv', index=False, float_format='%.5f')


def cv_by_xgb():
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'max_depth': 6,
        'min_child_weight': 8,
        'nthread': 8,
        'silent': 1,
        'alpha': 0.001,
        'gamma': 0.01,
        'seed': 2017
    }
    ## cross validation
    folds = skf.split(X, y)
    dtrain = xgb.DMatrix(data=X, label=y)
    num_boost_round = 1000
    bst = xgb.cv(
        params,
        dtrain,
        num_boost_round,
        N,
        feval=GiniEvaluation.gini_xgb,
        # maximize=True,
        metrics=['logloss'],
        folds=folds,
        early_stopping_rounds=100,
        verbose_eval=10
    )
    best_rounds = np.argmax(bst['test-gini-mean'])
    train_score = bst['train-gini-mean'][best_rounds]
    best_val_score = bst['test-gini-mean'][best_rounds]

    print('Best train_loss: %.5f, val_loss: %.5f at round %d.' % \
          (train_score, best_val_score, best_rounds))
    ## out-of-fold prediction
    oof_preds = np.zeros(y.shape)
    for trn_idx, val_idx in skf.split(X, y):
        trn_x, val_x = X[trn_idx], X[val_idx]
        trn_y = y[trn_idx]

    dtrn = xgb.DMatrix(data=trn_x, label=trn_y)
    dval = xgb.DMatrix(data=val_x)

    cv_model = xgb.train(params, dtrn, best_rounds)
    oof_preds[val_idx] = cv_model.predict(dval)
    print('XGBoost done')


# bst = cv_by_lgbm()
predict_by_ensemble_lgbm(10)
