# encoding=utf8
from __future__ import print_function

## logging setting
from logging_manage import initialize_logger
initialize_logger(output_dir='../../data/log/')
import logging
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from ipdb import set_trace as st


np.random.seed(17)

df_tn = pd.read_csv('../../data/input/train.csv')
df_tn.drop([149161], axis=0, inplace=True)
train_id = df_tn['id']
df_tt = pd.read_csv('../../data/input/test.csv')

# replace -1 for NaN

# train set
df_tn_z = df_tn.copy()
df_tn_z.replace(-1, np.NaN, inplace=True)

# test set
df_tt_z = df_tt.copy()
df_tt_z.replace(-1, np.NaN, inplace=True)

# -1 can be changed to 0 for features where there is no category "0",
# and features that have numerical values. Scrip below identifies such
# features as well as those where -1 shouldn't be changed.

# list with features
zero_list = ['ps_ind_02_cat', 'ps_reg_03', 'ps_car_12', 'ps_car_12',
             'ps_car_14', ]  # -1 can be changed for 0 in this features

minus_one = ['ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat',
             'ps_car_03_cat', 'ps_car_07_cat', 'ps_car_05_cat', 'ps_car_09_cat',
             'ps_car_11']  # these features already have 0 as value, thus -1 shouldn't be changed
# fill in missing values with 0 or -1

# train set
df_tn_z[minus_one] = df_tn_z[minus_one].fillna(-1)
df_tn_z[zero_list] = df_tn_z[zero_list].fillna(0)

# test set
df_tt_z[minus_one] = df_tt_z[minus_one].fillna(-1)
df_tt_z[zero_list] = df_tt_z[zero_list].fillna(0)

# group features by nature
cat_f = ['ps_ind_02_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat',
         'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
         'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',
         'ps_car_11_cat']
bin_f = ['ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
         'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
         'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
         'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',
         'ps_calc_19_bin', 'ps_calc_20_bin']
ord_f = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11']

cont_f = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13',
          'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03',
          'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
          'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',
          'ps_calc_14']

# transform categorical values to dummies
df_tn_proc = df_tn_z.copy().drop(['id', 'target'], axis=1)
df_tt_proc = df_tt_z.copy().drop(['id'], axis=1)
df_all_proc = pd.concat((df_tn_proc, df_tt_proc), axis=0, ignore_index=True)

for i in cat_f:
    d = pd.get_dummies(df_all_proc[i], prefix=i, prefix_sep='_')
    df_all_proc.drop(i, axis=1, inplace=True)
    df_all_proc = df_all_proc.merge(d, right_index=True, left_index=True)

# prepare X and Y
df_train = df_all_proc[:df_tn.shape[0]].copy()
df_y = df_tn['target'].copy()
df_test = df_all_proc[df_tn.shape[0]:].copy()
logging.info("train set shape {0}".format(df_train.shape))
logging.info("Y shape {0}".format(Y.shape))
logging.info("test shape {0}".format(df_test.shape))

del df_all_proc, df_tn_proc, df_tt_proc
# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## kfolds
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2017)

## define single lgbm model
single_lgbm = SingleLGBM(X=X, y=y, test=df_test, N=5, skf=skf)
gc.collect()




