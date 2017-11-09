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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
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

'''
params_for_n_round = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth' : 9,
    'min_data_in_leaf': 5000,
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
    'learning_rate': 0.03,
    'max_depth': 7,
    'num_leaves': 25,
    'min_child_weight' : 85,
    'min_split_gain' : 0.1,
    'min_data_in_leaf': 500,
    'max_bin': 10,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'reg_alpha' : 0.5,
    'reg_lambda' : 5,
    'verbose': -1,
    'seed' : 2017
}
do_cv = True
best_rounds = 20
if do_cv:
    best_rounds = single_lgbm.cv(
        params=params_for_submit,
        num_boost_round=1000,
        feval=GiniEvaluation.gini_lgbm,
    )
# record for stacker train
df_sub, stacker_train = \
    single_lgbm.oof(params=params_for_submit, best_rounds=best_rounds, sub=df_sub, do_logit=True)
df_sub.to_csv('../../data/output/sub_single_lgbm_001.csv', index=False)
df_sub.to_csv('../../data/for_stacker/sub_single_lgbm_001_test.csv', index=False)
s_train = pd.DataFrame()
s_train['id'] = train_id
s_train['prob'] = stacker_train
s_train.to_csv('../../data/for_stacker/single_lgbm_001_train.csv', index=False)
print('LightGBM done')

