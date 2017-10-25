# encoding=utf8
import pandas as pd
from fe import Processer, Compose

import xgboost as xgb
import lightgbm as lgbm
import catboost as cbt

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
train_specific = [
    (Processer.drop_rows, dict(row_indexes=[149161])),
]
test_specific = []
# execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(common_transforms_params+train_specific)(df_train)
logging.info('Transform test data')
df_test = Compose(common_transforms_params+test_specific)(df_test)
# execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])
# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## Define classifiers for Grid search
# xgboost, lightGBM, and catboost support sklearn GirdSearchCV
# implement method 'fit', 'predict', and 'predict_proba'
clf_xgb = xgb.XGBClassifier()
clf_lgbm = lgbm.LGBMClassifier()
clf_cbt = cbt.CatBoostClassifier()

st(context=21)
