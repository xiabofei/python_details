# encoding=utf8
#################################################################################
# FE:
#    1) remain oliver features
#    2) add 'convert reg 3' features
#    3) add combine ('ps_reg_01', 'ps_car_02_cat')
#################################################################################

import pandas as pd
from fe import Processer, Compose

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score
from single_model_utils import SingleXGB

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

# from olivier
train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
    "ps_reg_03",  #            : 1408.42 / shadow  511.15
    "ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
    "ps_ind_03",  #            : 1219.47 / shadow  230.55
    "ps_ind_15",  #            :  922.18 / shadow  242.00
    "ps_reg_02",  #            :  920.65 / shadow  267.50
    "ps_car_14",  #            :  798.48 / shadow  549.58
    "ps_car_12",  #            :  731.93 / shadow  293.62
    "ps_car_01_cat",  #        :  698.07 / shadow  178.72
    "ps_car_07_cat",  #        :  694.53 / shadow   36.35
    "ps_ind_17_bin",  #        :  620.77 / shadow   23.15
    "ps_car_03_cat",  #        :  611.73 / shadow   50.67
    "ps_reg_01",  #            :  598.60 / shadow  178.57
    "ps_car_15",  #            :  593.35 / shadow  226.43
    "ps_ind_01",  #            :  547.32 / shadow  154.58
    "ps_ind_16_bin",  #        :  475.37 / shadow   34.17
    "ps_ind_07_bin",  #        :  435.28 / shadow   28.92
    "ps_car_06_cat",  #        :  398.02 / shadow  212.43
    "ps_car_04_cat",  #        :  376.87 / shadow   76.98
    "ps_ind_06_bin",  #        :  370.97 / shadow   36.13
    "ps_car_09_cat",  #        :  214.12 / shadow   81.38
    "ps_car_02_cat",  #        :  203.03 / shadow   26.67
    "ps_ind_02_cat",  #        :  189.47 / shadow   65.68
    "ps_car_11",  #            :  173.28 / shadow   76.45
    "ps_car_05_cat",  #        :  172.75 / shadow   62.92
    "ps_calc_09",  #           :  169.13 / shadow  129.72
    "ps_calc_05",  #           :  148.83 / shadow  120.68
    "ps_ind_08_bin",  #        :  140.73 / shadow   27.63
    "ps_car_08_cat",  #        :  120.87 / shadow   28.82
    "ps_ind_09_bin",  #        :  113.92 / shadow   27.05
    "ps_ind_04_cat",  #        :  107.27 / shadow   37.43
    "ps_ind_18_bin",  #        :   77.42 / shadow   25.97
    "ps_ind_12_bin",  #        :   39.67 / shadow   15.52
    "ps_ind_14",  #            :   37.37 / shadow   16.65
]

## Data Processing and Feature Engineering
transformer_two = [
    (Processer.remain_columns, dict(col_names=train_features)),
    # (Processer.median_mean_range, dict(opt_median=True, opt_mean=True)),
    (Processer.convert_reg_03, dict()),
    (Processer.dtype_transform, dict()),
]
# execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(transformer_two)(df_train)
logging.info('Transform test data')
df_test = Compose(transformer_two)(df_test)
# execute add combine
df_train, df_test = Processer.add_combine(df_train, df_test,'ps_reg_01', 'ps_car_02_cat')
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
'''
params_for_n_round = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.07,
    'max_depth': 5,
    'min_child_weight':1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,
    'nthread': 6,
    'silent': 1,
    'seed': 2017,
}
best_rounds = single_xgb.cv(
    params=params_for_n_round,
    num_boost_round=500,
    feval=GiniEvaluation.gini_xgb,
    feval_name='gini',
    maximize=True,
    metrics=['auc'],
)
'''
single_xgb = SingleXGB(X=X, y=y, test=df_test, N=5, skf=skf)
'''
xgb_param = dict(
    # target
    objective='binary:logistic',
    # booster parameters
    booster='gbtree',
    n_estimators=186,
    # tree-based parameters
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    # max_delta_step=0,
    # regularization parameters
    # reg_alpha=10.4,
    # reg_lambda=5,
    # learning rate
    learning_rate=0.07,
    # others
    n_jobs=2,
    # base_score=0.5,
    random_state=2017,
    # missing=None,
    early_stopping_rounds=40,
    verbose_eval=10,
)
xgb_param_grid = dict(
    max_depth=[4, 5, 6],
    min_child_weight=[1, 2, 3, 4, 5, 6, 7, 8, 9],
)
ret = single_xgb.grid_search_tuning(
    xgb_param=xgb_param,
    xgb_param_grid=xgb_param_grid,
    f_score=gini_score,
    n_jobs=5
)
'''
params_for_submit = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.03,
    'max_depth': 5,
    'min_child_weight': 8.49,
    'gamma': 0.93,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 9.9,
    'lambda': 0.4,
    'seed': 2017,
    'nthread': 10,
    'silent': 1,
}
do_cv = False
best_rounds = 569
if do_cv:
    best_rounds = single_xgb.cv(
        params=params_for_submit,
        num_boost_round=1000,
        feval=GiniEvaluation.gini_xgb,
        feval_name='gini',
        maximize=True,
        metrics=['auc'],
    )
# record for stacker train
df_sub, stacker_train = \
    single_xgb.oof(params=params_for_submit, best_rounds=best_rounds, sub=df_sub, do_logit=True)
df_sub.to_csv('../../data/output/sub_single_xgb_002.csv', index=False)
df_sub.to_csv('../../data/for_stacker/sub_single_xgb_002_test.csv', index=False)
s_train = pd.DataFrame()
s_train['id'] = train_id
s_train['prob'] = stacker_train
s_train.to_csv('../../data/for_stacker/single_xgb_002_train.csv', index=False)
print('XGBoost done')
