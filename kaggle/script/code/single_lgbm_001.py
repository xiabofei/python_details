# encoding=utf8
#################################################################################
# FE:
#    1) add 'negative one vals' features
#    2) drop ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
#################################################################################

import pandas as pd
from fe import Processer, Compose

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from single_model_utils import SingleLGBM
from fe import FeatureImportance

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
train_id = df_train['id']
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
        'num_leaves': 31,
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

# fi = fast_feature_importance(
#     df_train=df_train,
#     df_y=df_y,
#     feval=GiniEvaluation.gini_lgbm,
#     feature_watch_list=['negative_one_vals']
# )

# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## kfolds
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2017)

## define single lgbm model
single_lgbm = SingleLGBM(X=X, y=y, test=df_test, N=5, skf=skf)

params_for_n_round = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth' : 9,
    'min_data_in_leaf': 10000,
    'max_bin': 10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
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
