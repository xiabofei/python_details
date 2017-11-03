# encoding=utf8
import pandas as pd
import numpy as np
from fe import Processer, Compose, FeatureImportance
from collections import OrderedDict

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from single_model import SingleXGB

from stack_ensemble import StackEnsemble
from sklearn.linear_model import LogisticRegression

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
df_sub = df_test['id'].to_frame()
df_sub['target'] = 0.0

## Separate label & feature
df_y = df_train['target']
df_train.drop(['id', 'target'], axis=1, inplace=True)
df_sub = df_test['id'].to_frame()
df_sub['target'] = 0.0
df_test.drop(['id'], axis=1, inplace=True)

## Data Processing and Feature Engineering
TRANSFORMER = 'transformer'
PARAMS = 'params'
transformer_parameter_candidates = [
    # transformer 0
    [
        (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
        (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
        (Processer.negative_one_vals, dict()),
        (Processer.dtype_transform, dict()),
    ],
    # transformer 1
    [
        (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
        (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
        (Processer.median_mean_range, dict(opt_median=True, opt_mean=True)),
        (Processer.convert_reg_03, dict()),
        (Processer.dtype_transform, dict()),
    ],
    # transformer 2
    [
        (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
        (Processer.descartes, dict(left_col_names=['ps_car_13'], right_col_names=['ps_reg_03'])),
        (Processer.negative_one_vals, dict()),
        (Processer.ohe_by_unique, dict()),
        (Processer.median_mean_range, dict(opt_median=True, opt_mean=True)),
        (Processer.dtype_transform, dict()),
    ],
]


## Execute transforms pipeline
def execute_transformer(df_train, df_test, transformer):
    logging.info('Transform train data by transformer : {0}'.format(transformer))
    df_train = Compose(transforms_params=transformer)(df_train)
    logging.info('Transform test data by transformer : {0}'.format(transformer))
    df_test = Compose(transforms_params=transformer)(df_test)
    # df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])
    return df_train, df_test


df_train, df_test = \
    execute_transformer(df_train=df_train, df_test=df_test, transformer=transformer_parameter_candidates[2])
logging.info('df_train shape {0}'.format(df_train.shape))
logging.info('df_test shape {0}'.format(df_test.shape))
st(context=21)

gc.collect()

## Fast feature importance evaluation by xgboost
def fast_feature_importance(df_train, df_y, feval, feature_watch_list):
    params_for_fi = {
        'objective': 'binary',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_data_in_leaf': 500,
        'max_bin': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': 1,
    }
    ret = FeatureImportance.lgbm_fi(
        params=params_for_fi,
        df_data=df_train,
        df_label=df_y,
        feval=feval,
        num_boost_round=100,
        feature_watch_list=feature_watch_list,
    )
    return ret


# fi = fast_feature_importance(
#     df_train=df_train, df_y=df_y, feval=GiniEvaluation.gini_lgbm, feature_watch_list=['negative_one_vals'] )


## Build lgbm classifier
# record model params
lgbm_params = [
    ## param0, param1, and param2 from ( https://www.kaggle.com/yekenot/simple-stacker-lb-0-284/code )
    # param 0
    dict(
        learning_rate=0.01,
        n_estimators=1250,
        max_bin=10,
        subsample=0.8,
        subsample_freq=10,
        colsample_bytree=0.8,
        min_child_weight=500,
        # random_state=99,
    ),
    # param 1
    dict(
        learning_rate=0.005,
        n_estimators=3700,
        subsample=0.7,
        subsample_freq=2,
        colsample_bytree=0.3,
        num_leaves=16,
        # random_state=99,
    ),
    # param 2
    dict(
        learning_rate=0.02,
        n_estimators=800,
        max_depth=4,
        # random_state=99,
    ),
]
# create materials for stack ensemble
lgbm_estimator1 = lgbm.LGBMClassifier(**lgbm_params[0])
lgbm_estimator2 = lgbm.LGBMClassifier(**lgbm_params[1])
lgbm_estimator3 = lgbm.LGBMClassifier(**lgbm_params[2])
base_models = (lgbm_estimator1, lgbm_estimator2, lgbm_estimator3,)
# create stacker
stack = StackEnsemble(n_splits=3, stacker=LogisticRegression(), base_models=base_models)
# fit and create and sub

##
'''
# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## Grid Search
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2017)
lgbm_param = dict(
    boosting_type='gbdt',
    num_leaves=61,
    # max_depth=6,
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
lgbm_param_grid = dict(num_leaves=[31, 63])
lgbm_estimator = lgbm.LGBMClassifier(**lgbm_param)
lgbm_gs = GridSearchCV(
    estimator=lgbm_estimator,
    param_grid=lgbm_param_grid,
    cv=skf,
    scoring=make_scorer(GiniEvaluation.gini_normalized, greater_is_better=True, needs_proba=True),
    verbose=10,
    n_jobs=1,
)
'''
