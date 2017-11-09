# encoding=utf8
#####################################################################
# single_xgb_001 + single_xgb_003 +
# LB : 0.285
#####################################################################

from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

df_train_single_lgbm_001 = pd.read_csv(root_dir + 'single_lgbm_001_train.csv')
df_test_single_lgbm_001 = pd.read_csv(root_dir + 'sub_single_lgbm_001_test.csv')
logging.info('df_train_single_lgbm_001 shape : {0}'.format(df_train_single_lgbm_001.shape))


def exp(df):
    return 1 / (1 + np.exp(-df))


df_train = pd.DataFrame()
df_train['col1'] = exp(df_train_single_xgb_001['col1'])
df_train['col2'] = exp(df_train_single_xgb_002['prob'])
df_train['col3'] = exp(df_train_complex_xgb['col1'])
df_train['col4'] = exp(df_train_simple_stacker_001['xgb'])
df_train['col5'] = exp(df_train_simple_stacker_001['lgb1'])
df_train['col6'] = exp(df_train_simple_stacker_001['lgb2'])
df_train['col7'] = exp(df_train_simple_stacker_001['lgb3'])
df_train['col8'] = exp(df_train_single_lgbm_001['prob'])
logging.info('df_train shape : {0}'.format(df_train.shape))

df_test = pd.DataFrame()
df_test['col1'] = exp(df_test_single_xgb_001['col1'])
df_test['col2'] = exp(df_test_single_xgb_002['target'])
df_test['col3'] = exp(df_test_complex_xgb['col1'])
df_test['col4'] = exp(df_test_simple_stacker_001['xgb'])
df_test['col5'] = exp(df_test_simple_stacker_001['lgb1'])
df_test['col6'] = exp(df_test_simple_stacker_001['lgb2'])
df_test['col7'] = exp(df_test_simple_stacker_001['lgb3'])
df_test['col8'] = exp(df_test_single_lgbm_001['target'])
logging.info('df_test shape : {0}'.format(df_test.shape))


# Model combination strategies :
#     1. using high lb score singles
#     2. traverse lower lb model combination by cv auc score
#     3. stacker using fast LR

single_lb284 = ['col1', 'col3']
single_lb282 = ['col2']
stack_lb284 = ['col4', 'col5', 'col6', 'col7']
single_lgbm_281 = ['col8']

stacker_estimator = LogisticRegression()
'''
models_cv_result = []
models_to_select = single_lb282 + stack_lb284 + single_lgbm_281
for model_num in range(1, len(models_to_select) + 1):
    for models in combinations(models_to_select, model_num):
        _models = single_lb284 + list(models)
        logging.info("Stacker using models : {0}".format(_models))
        _train = df_train[_models]
        results = cross_val_score(estimator=stacker_estimator, X=_train, y=df_y.values, cv=5, scoring='roc_auc')
        logging.info("Stacker score: %.6f" % (results.mean()))
        models_cv_result.append((results.mean(), _models,))

# Select single model sets for final ensemble
models_cv_result = sorted(models_cv_result, key=lambda x:x[0], reverse=True)
selected_models = models_cv_result[0][1]
logging.info('Select single model sets : {0}'.format(selected_models))
logging.info('And its local cv auc score : {0}'.format(models_cv_result[0][0]))

'''
# Stack ensemble and predict
selected_models = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']
stacker_estimator.fit(X=df_train[selected_models], y=df_y.values)
probs = stacker_estimator.predict_proba(X=df_test[selected_models].values)

# Create submit files
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = probs[:, 1]
sub.to_csv('../../data/output/sub_stacker_ensemble_3_kernels.csv', index=False)
logging.info('Creating submit file done')
