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
from model_utils import SingleXGB
from io_utils import read_data, write_data, Number_of_folds, comm_skf

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
    (Processer.median_mean_range, dict(opt_median=True, opt_mean=True)),
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


## Grid Search
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


