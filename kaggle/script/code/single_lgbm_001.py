# encoding=utf8
#################################################################################
# FE:
#    1) add 'negative one vals' features
#    2) drop ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
# LB 0.284
#################################################################################

import pandas as pd
from fe import Processer, Compose
import numpy as np

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from model_utils import SingleLGBM
from io_utils import read_data, comm_skf, write_data, Number_of_folds
from fe import FeatureImportance
from scipy.stats import randint, uniform
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

## define single lgbm model
single_lgbm = SingleLGBM(X=X, y=y, test=df_test, skf=skf, N=Number_of_folds)

params_for_submit = {
    'objective': 'binary',
    'learning_rate': 0.01,
    'max_depth': 7,
    'num_leaves': 25,
    'min_child_weight' : 12.416566,
    'min_split_gain' : 0.1,
    'min_data_in_leaf': 10,
    'max_bin': 55,
    'feature_fraction': 0.7331594,
    'bagging_fraction': 0.75262525,
    'bagging_freq': 1,
    'reg_alpha' : 4.78529147,
    'reg_lambda' : 0.9993343,
    'verbose': 0,
    'seed' : 2017
}
do_cv = False
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
