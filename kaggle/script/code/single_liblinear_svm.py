# encoding=utf8
#################################################################################
# LB
#################################################################################
import pandas as pd
from fe import Processer, Compose
import numpy as np
from io_utils import check_nan
from evaluation import gini_score_svm

from sklearn.preprocessing import Imputer
from evaluation import GiniEvaluation, gini_score
from io_utils import read_data, comm_skf, write_data, Number_of_folds
from logging_manage import initialize_logger
from liblinearutil import *
import liblinear
import logging
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')
## Data Loading
df_train, df_y, df_test, df_sub, train_id = read_data()
## Common skf
skf = comm_skf

'''
#################################################################################
## Data Processing and Feature Engineering
# build up transform pipeline
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
#################################################################################


#################################################################################
## Handle missing value
# select value and category columns
value_cols = [c for c in df_train.columns if '_cat' not in c and '_bin' not in c ]
cat_cols = [c for c in df_train.columns if c not in value_cols ]
# handle category columns
logging.info('Handle category feature columns missing values')
imputer_for_cat = Imputer(strategy='most_frequent', axis=0)
df_train[cat_cols] = imputer_for_cat.fit_transform(X=df_train[cat_cols].values)
df_test[cat_cols] = imputer_for_cat.fit_transform(X=df_test[cat_cols].values)
# handle value columns
logging.info('Handle value feature columns missing values')
imputer_for_value = Imputer(strategy='mean', axis=0)
df_train[value_cols] = imputer_for_value.fit_transform(X=df_train[value_cols].values)
df_test[value_cols] = imputer_for_value.fit_transform(X=df_test[value_cols].values)
# normalize value to 0 and 1
df_train = Processer.normalization(df_train, df_train.columns, val_range=1)
df_test = Processer.normalization(df_test, df_train.columns, val_range=1)
# save memory
df_train = Processer.dtype_transform(df_train)
df_test = Processer.dtype_transform(df_test)
# write file to local disk
df_train.to_csv('../../data/input/df_train_4_svm.csv', index=False)
df_test.to_csv('../../data/input/df_test_4_svm.csv', index=False)
#################################################################################

st(context=21)
'''

#################################################################################
## Read data from local disk file
logging.info('loading already processed train from disk file')
df_train = pd.read_csv('../../data/input/df_train_4_svm.csv', index_col=None)
logging.info('loading already processed test from disk file')
df_test = pd.read_csv('../../data/input/df_test_4_svm.csv', index_col=None)
# Make sure no NaN or Inf in train or test data
assert check_nan(df_train) == 0, 'df_train contain nan value'
assert check_nan(df_test) == 0, 'df_test contain nan value'
# Save memory
# df_train = Processer.dtype_transform(df_train)
# df_test = Processer.dtype_transform(df_test)
#################################################################################


# feature and label for train
X = df_train.values
y = df_y.apply(lambda x: -1 if x == 0 else x).values

SAMPLES = 200000
model = train(y[0:SAMPLES], X[0:SAMPLES, :], '-c 200 -s 2')
p_labels, p_acc, p_vals = predict(y[SAMPLES:], X[SAMPLES:, :], model)
p_labels = [ 0 if l==-1 else l for l in p_labels ]
st(context=21)
gini_score_svm(y[SAMPLES:], p_labels)


