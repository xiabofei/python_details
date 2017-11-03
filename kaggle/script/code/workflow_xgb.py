# encoding=utf8
import pandas as pd
import numpy as np
from fe import Processer, Compose, FeatureImportance

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from single_model import SingleXGB

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
transformer_one = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.negative_one_vals, dict()),
    (Processer.dtype_transform, dict()),
]
transformer_two = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.median_mean_range, dict(opt_median=True, opt_mean=True)),
    (Processer.convert_reg_03, dict()),
    (Processer.dtype_transform, dict()),
]
train_specific = []
test_specific = []
# execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(transformer_one + train_specific)(df_train)
logging.info('Transform test data')
df_test = Compose(transformer_one + test_specific)(df_test)
# execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])

# fast feature importance evaluation by xgboost
'''
params_for_fi = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'seed': 2017,
    'nthread': 8,
    'silent': 1,
}
df_feature_importance = FeatureImportance.xgb_fi(
    params=params_for_fi,
    data=df_train,
    label=df_y,
    feval=GiniEvaluation.gini_xgb,
    maximize=True,
    num_boost_round=100, cv=True,
)
st(context=21)
'''

# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## kfolds
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2017)

## Grid Search
params_for_n_round = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,
    'nthread': 6,
    'silent': 1,
    'seed': 2017,
}
xgb_param = dict(
    # target
    objective='binary:logistic',
    # booster parameters
    booster='gbtree',
    n_estimators=192,
    # tree-based parameters
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    # max_delta_step=0,
    # regularization parameters
    # reg_alpha=1e-4,
    # reg_lambda=20,
    # learning rate
    learning_rate=0.08,
    # others
    n_jobs=2,
    base_score=0.5,
    random_state=2017,
    missing=None,
    early_stopping_rounds=50,
    verbose_eval=10,
)
xgb_param_grid = dict(
    max_depth=[4, 5, 6],
    min_child_weight=[5, 7, 9],
)
single_xgb = SingleXGB(X=X, y=y, test=df_test, N=5, skf=skf)
'''
ret = single_xgb.cv(
    params=params_for_n_round,
    num_boost_round=1000,
    feval=GiniEvaluation.gini_xgb,
    feval_name='gini',
    maximize=True,
    metrics=['auc'],
)
st(context=21)
'''
ret = single_xgb.grid_search_tuning(
    xgb_param=xgb_param,
    xgb_param_grid=xgb_param_grid,
    f_score=gini_score,
    n_jobs=6
)
