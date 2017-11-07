# encoding=utf8
#####################################################################
# single_xgb_001 + single_xgb_003 +
# LB : 0.285
#####################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

from logging_manage import initialize_logger
import logging
import numpy as np

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')

from ipdb import set_trace as st

root_dir = '../../data/for_stacker/'

train = pd.read_csv('../../data/input/train.csv')
train.drop([149161], axis=0, inplace=True)
df_y = train['target']

df_train_single_xgb_001 = pd.read_csv(root_dir + 'single_xgb_001_train.csv')
train_idx = df_train_single_xgb_001['id']
df_test_single_xgb_001 = pd.read_csv(root_dir + 'sub_single_xgb_001_test.csv')
test_id = df_test_single_xgb_001['id']
logging.info('df_train_single_xgb_001 shape : {0}'.format(df_train_single_xgb_001.shape))

df_train_single_xgb_002 = pd.read_csv(root_dir + 'single_xgb_002_train.csv')
df_test_single_xgb_002 = pd.read_csv(root_dir + 'sub_single_xgb_002_test.csv')
logging.info('df_train_single_xgb_002 shape : {0}'.format(df_train_single_xgb_002.shape))

df_train_simple_stacker_001 = pd.read_csv(root_dir + 'simple_stacker_001_train.csv')
df_test_simple_stacker_001 = pd.read_csv(root_dir + 'sub_simple_stacker_001_test.csv')
logging.info('df_train_simple_stacker_001 shape : {0}'.format(df_train_simple_stacker_001.shape))

df_train_complex_xgb = pd.read_csv(root_dir + 'xgb_valid.csv')
df_test_complex_xgb = pd.read_csv(root_dir + 'xgb_submit.csv')
logging.info('df_train_complex_xgb shape : {0}'.format(df_train_complex_xgb.shape))


def exp(df):
    return 1 / (1 + np.exp(-df))


df_train = pd.DataFrame()
df_train['col1'] = df_train_single_xgb_001['col1']
df_train['col2'] = df_train_single_xgb_002['prob']
df_train['col3'] = df_train_complex_xgb['col1']
df_train['col4'] = df_train_simple_stacker_001['xgb']
df_train['col5'] = df_train_simple_stacker_001['lgb1']
df_train['col6'] = df_train_simple_stacker_001['lgb2']
df_train['col7'] = df_train_simple_stacker_001['lgb3']
logging.info('df_train shape : {0}'.format(df_train.shape))

df_test = pd.DataFrame()
df_test['col1'] = df_test_single_xgb_001['col1']
df_test['col2'] = df_test_single_xgb_002['target']
df_test['col3'] = df_test_complex_xgb['col1']
df_test['col4'] = df_test_simple_stacker_001['xgb']
df_test['col5'] = df_test_simple_stacker_001['lgb1']
df_test['col6'] = df_test_simple_stacker_001['lgb2']
df_test['col7'] = df_test_simple_stacker_001['lgb3']
logging.info('df_test shape : {0}'.format(df_test.shape))

st(context=21)

lr = LogisticRegression()

lr.fit(X=df_train[['col1', 'col3', 'col4', 'col5', 'col6', 'col7']], y=df_y.values)

probs = lr.predict_proba(X=df_test.values)

sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = probs[:, 1]
sub.to_csv('../../data/output/sub_stacker_ensemble_3_kernels.csv', index=False)
