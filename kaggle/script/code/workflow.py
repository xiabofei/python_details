# encoding=utf8
import pandas as pd
import numpy as np
from fe import Processer, Compose

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score

from logging_manage import initialize_logger
import logging
import os
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')

## Data Loading
root_dir = '../../data/input/'
df_train = pd.read_csv(os.path.join(root_dir, 'train.csv'), na_values=-1)
df_train.drop([149161], axis=0, inplace=True)
df_test = pd.read_csv(os.path.join(root_dir, 'test.csv'), na_values=-1)

## Separate label & feature
df_y = df_train['target']
df_train.drop(['id', 'target'], axis=1, inplace=True)
df_sub = df_test['id'].to_frame()
df_sub['target'] = 0.0
df_test.drop(['id'], axis=1, inplace=True)

## Data Processing and Feature Engineering
# build train data process pipeline and test data process pipeline
common_transforms_params = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.dtype_transform, dict()),
]
train_specific = []
test_specific = []
# execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(common_transforms_params + train_specific)(df_train)
logging.info('Transform test data')
df_test = Compose(common_transforms_params + test_specific)(df_test)
# execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])
# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## kfolds
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2017)
folds = skf.split(X, y)

## Grid Search
# lightGBM
# step1. search for n_estimators
params_for_n_estimators = {
    'metric': 'auc',
    'objective': 'binary',
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'verbose': -1,
}
n_estimators = 2000
bst = lgbm.cv(
    params=params_for_n_estimators,
    train_set=lgbm.Dataset(data=X, label=y),
    folds=folds,
    num_boost_round=n_estimators,
    metrics=['auc'],
    feval=GiniEvaluation.gini_lgb,
    early_stopping_rounds=50,
    verbose_eval=20,
)
best_rounds = np.argmax(bst['gini-mean']) + 1
best_val_score = np.max(bst['gini-mean'])
st(context=21)
lgbm_param = dict(
    boosting_type='gbdt',
    num_leaves=61,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=10,
    max_bin=500,
    subsample_for_bin=50000,
    objective='binary',
    min_split_gain=0.1,
    min_child_weight=5,
    min_child_samples=10,
    subsample=0.6,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_alpha=0.01,
    reg_lambda=0.01,
    random_state=0,
    n_jobs=-1,
    silent=True,
    early_stopping_rounds=50,
    verbose_eval=10,
)
lgbm_param_grid = dict(
    num_leaves=[61],
    max_depth=[6],
    learning_rate=[0.01, 0.05],
)
lgbm_estimator = lgbm.LGBMClassifier(**lgbm_param)
lgbm_gs = GridSearchCV(
    estimator=lgbm_estimator,
    param_grid=param_grid,
    cv=skf,
    scoring=make_scorer(gini_normalized, greater_is_better=True, needs_proba=True),
    verbose=10,
    n_jobs=1,
)
lgbm_gs.fit(X, y)
# xgboost tuning
'''
params_for_n_estimators = {
    'objective': 'binary:logistic',
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 5,
    'nthread': 8,
    'silent': 1,
    'scale_pos_weight': 1,
}
n_estimators = 1000
bst = xgb.cv(
    params=params_for_n_estimators,
    dtrain=xgb.DMatrix(data=X, label=y),
    num_boost_round=n_estimators,
    folds=skf.split(X, y),
    feval=GiniEvaluation.gini_xgb,
    maximize=True,
    metrics=['auc'],
    early_stopping_rounds=50,
    verbose_eval=10
)
best_rounds = np.argmax(bst['test-gini-mean'])
logging.info('best n_estimator {0} at {1}'.format(best_rounds, params_for_n_estimators))
xgb_param = dict(
    # target
    objective='binary:logistic',
    # booster parameters
    booster='gbtree',
    n_estimators=312,
    # tree-based parameters
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=1,
    scale_pos_weight=1,
    max_delta_step=0,
    # regularization parameters
    reg_alpha=0,
    reg_lambda=1,
    # learning rate
    learning_rate=0.05,
    # others
    n_jobs=1,
    nthread=None,
    base_score=0.5,
    random_state=0,
    seed=2017,
    missing=None,
    early_stopping_rounds=50,
    verbose_eval=10,
)
xgb_param_grid = dict(
    max_depth=range(3,10,2),
    min_child_weight=range(1,6,2),
)
xgb_estimator = xgb.XGBClassifier(**xgb_param)
xgb_gs = GridSearchCV(
    estimator=xgb_estimator, param_grid=xgb_param_grid, cv=skf,
    scoring=make_scorer(gini_score, greater_is_better=True, needs_proba=True),
    verbose=2, n_jobs=10,
)
xgb_gs.fit(X, y)
'''
