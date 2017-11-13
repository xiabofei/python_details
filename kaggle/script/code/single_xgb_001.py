# encoding=utf8
#################################################################################
# FE:
#    1) add 'negative one vals' features
#    2) drop ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
# LB : 0.284
#################################################################################

import pandas as pd
from fe import Processer, Compose

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from model_utils import SingleXGB
from io_utils import read_data, comm_skf, Number_of_folds, write_data

from logging_manage import initialize_logger
import logging
import os
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')

## Data Loading
df_train, df_y, df_test, df_sub, train_id = read_data()
## Common skf
skf = comm_skf

## Data Processing and Feature Engineering
transformer_one = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.negative_one_vals, dict()),
    (Processer.dtype_transform, dict()),
]
# execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(transformer_one)(df_train)
logging.info('Transform test data')
df_test = Compose(transformer_one)(df_test)
# execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])

# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## Grid Search
'''
params_for_n_round = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.08,
    'max_depth': 5,
    'min_child_weight':9,
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
    # n_estimators=192,
    # tree-based parameters
    max_depth=5,
    min_child_weight=9.15,
    gamma=0.59,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    max_delta_step=1.9,
    # regularization parameters
    reg_alpha=10.4,
    reg_lambda=5,
    # learning rate
    learning_rate=0.04,
    # others
    n_jobs=8,
    # base_score=0.5,
    random_state=2017,
    missing=None,
    # early_stopping_rounds=50,
    # verbose_eval=10,
)
xgb_param_grid = dict(
)
ret = single_xgb.grid_search_tuning(
    xgb_param=xgb_param,
    xgb_param_grid=xgb_param_grid,
    f_score=gini_score,
    n_jobs=5
)
'''

## cv and oof
params_for_submit = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.04,
    'max_depth': 5,
    'min_child_weight': 9.15,
    'gamma': 0.59,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 10.4,
    'lambda': 5,
    'seed': 2017,
    'nthread': 10,
    'silent': 1,
}
single_xgb = SingleXGB(X=X, y=y, test=df_test, skf=skf, N=Number_of_folds)
best_rounds = 431
DO_CV = False
if DO_CV:
    best_rounds = single_xgb.cv(
        params=params_for_submit,
        num_boost_round=1000,
        feval=GiniEvaluation.gini_xgb,
        feval_name='gini',
        maximize=True,
        metrics=['auc'],
    )
# use oof for single submit file and train file
df_sub, stacker_train = single_xgb.oof(
    params=params_for_submit,
    best_rounds=best_rounds,
    sub=df_sub,
    do_logit=False
)
# write submit file and train file to local disk
write_data(
    df_sub=df_sub,
    train_id=train_id,
    stacker_train=stacker_train,
    sub_filename='sub_single_xgb_001_test.csv',
    train_filename='single_xgb_001_train.csv'
)
logging.info('Single XGBoost done')
