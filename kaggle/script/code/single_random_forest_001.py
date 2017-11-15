# encoding=utf8
#################################################################################
# FE:
#    1) add 'negative one vals' features
#    2) drop ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
#    3) rf must handle missing value
#################################################################################

import pandas as pd
from fe import Processer, Compose
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cPickle

from model_utils import SingleRF

from sklearn.preprocessing import Imputer

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

'''
## Data Processing and Feature Engineering
transformer = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.negative_one_vals, dict()),
    (Processer.median_mean_range, dict(opt_median=True, opt_mean=False)),
    (Processer.convert_reg_03, dict()),
    (Processer.dtype_transform, dict()),
]
# execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(transformer)(df_train)
logging.info('Transform test data')
df_test = Compose(transformer)(df_test)
# execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])
st(context=21)
# handle missing value
value_cols = [c for c in df_train.columns if '_cat' not in c and '_bin' not in c ]
cat_cols = [c for c in df_train.columns if c not in value_cols ]
logging.info('Handle category feature columns missing values')
imputer_for_cat = Imputer(strategy='most_frequent', axis=0)
df_train[cat_cols] = imputer_for_cat.fit_transform(X=df_train[cat_cols].values)
df_test[cat_cols] = imputer_for_cat.fit_transform(X=df_test[cat_cols].values)
logging.info('Handle value feature columns missing values')
imputer_for_value = Imputer(strategy='median', axis=0)
df_train[value_cols] = imputer_for_value.fit_transform(X=df_train[value_cols].values)
df_test[value_cols] = imputer_for_value.fit_transform(X=df_test[value_cols].values)
gc.collect()
df_train.to_csv('../../data/input/df_train_4_rf.csv', index=False)
df_test.to_csv('../../data/input/df_test_4_rf.csv', index=False)
st(context=21)
'''

logging.info('loading already processed train from disk file')
df_train = pd.read_csv('../../data/input/df_train_4_rf.csv', index_col=None)
logging.info('loading already processed test from disk file')
df_test = pd.read_csv('../../data/input/df_test_4_rf.csv', index_col=None)

feat_num = len(df_train.columns)

# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

single_rf = SingleRF(X=X, y=y, test=df_test, skf=skf, N=Number_of_folds)

rf_param = dict(
    n_estimators=500,
    criterion="gini",
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.,
    max_features=int(np.sqrt(feat_num)),
    max_leaf_nodes=None,
    min_impurity_decrease=0.,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=2,
    random_state=2017,
    verbose=0,
    warm_start=False,
    class_weight=None
)

rf_param_distribution = dict(
    criterion=['gini', 'entropy'],
    min_samples_split=list(set(np.random.randint(50,5000,100))),
    min_samples_leaf=list(set(np.random.randint(50,5000,100))),
    # max_features=np.multiply([int(np.sqrt(feat_num))], [1 ,2]) ,
)

best_params = single_rf.random_grid_search_tuning(
    rf_param=rf_param,
    rf_param_distribution=rf_param_distribution,
    n_iter=20,
    f_score=gini_score,
    n_jobs=5,
)

st(context=21)
logging.info('conduct random forest oof process')
df_sub, stack_train = single_rf.oof(
    params=rf_param,
    sub=df_sub,
)
st(context=21)
