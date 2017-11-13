# encoding=utf8
#################################################################################
# FE:
#    1) add 'negative one vals' features
#    2) drop ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
# LB 0.281
#################################################################################

import pandas as pd
from fe import Processer, Compose

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from model_utils import SingleLGBM
from io_utils import read_data, comm_skf, write_data, Number_of_folds
from fe import FeatureImportance

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

# fast feature importance evaluation by xgboost
def fast_feature_importance(df_train, df_y, feval, feature_watch_list):
    params_for_fi = {
        'objective': 'binary',
        'learning_rate': 0.03,
        'num_leaves': 16,
        'min_data_in_leaf': 500,
        'max_bin': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': 0,
    }
    ret = FeatureImportance.lgbm_fi(
        params=params_for_fi,
        df_data=df_train,
        df_label=df_y,
        feval=feval,
        num_boost_round=500,
        feature_watch_list=feature_watch_list,
    )
    return ret
# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## define single lgbm model
single_lgbm = SingleLGBM(X=X, y=y, test=df_test, skf=skf, N=Number_of_folds)

'''
params_for_n_round = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 25,
    'min_child_weight' : 85,
    'max_bin' : 10,
    'max_depth' : 10,
    'verbosity': 0,
}
best_rounds, best_score = single_lgbm.cv(
    params=params_for_n_round,
    num_boost_round=500,
    feval=GiniEvaluation.gini_lgbm
)
st(context=21)
lgbm_param = dict(
    boosting_type='gbdt',
    num_leaves=31,
    # max_depth=10,
    learning_rate=0.05,
    n_estimators=280,
    max_bin=10,
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
    seed = 2017,
    nthread=2,
    silent=True,
    early_stopping_rounds=50,
)
lgbm_param_grid = dict(
)
ret = single_lgbm.grid_search_tuning(
    lgbm_param=lgbm_param,
    lgbm_param_grid=lgbm_param_grid,
    f_score=gini_score,
    n_jobs=5
)
'''
params_for_submit = {
    'objective': 'binary',
    'learning_rate': 0.01,
    'max_depth': 7,
    'num_leaves': 25,
    'min_child_weight' : 10,
    'min_split_gain' : 0.1,
    'min_data_in_leaf': 500,
    'max_bin': 10,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'reg_alpha' : 0.5,
    'reg_lambda' : 0.5,
    'verbose': -1,
    'seed' : 2017
}
do_cv = True
best_rounds = 20
if do_cv:
    best_rounds = single_lgbm.cv(
        params=params_for_submit,
        num_boost_round=2000,
        feval=GiniEvaluation.gini_lgbm,
    )
# record for stacker train
df_sub, stacker_train = single_lgbm.oof(
    params=params_for_submit,
    best_rounds=best_rounds,
    sub=df_sub,
    do_logit=False,
)

write_data(
    df_sub=df_sub,
    stacker_train=stacker_train,
    train_id=train_id,
    sub_filename='sub_single_lgbm_001_test.csv',
    train_filename='single_lgbm_001_train.csv'
)
logging.info('LightGBM done')
