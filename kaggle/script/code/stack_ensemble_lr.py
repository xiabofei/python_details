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

from model_utils import SingleXGB
from io_utils import Number_of_folds, comm_skf
from evaluation import GiniEvaluation, gini_score

from logging_manage import initialize_logger
import logging
import numpy as np

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')

from ipdb import set_trace as st

root_dir = '../../data/for_stacker/'

train = pd.read_csv('../../data/input/train_id.csv')
df_y = train['target']

# single xgb 001 (LB 284)
df_train_single_xgb_001 = pd.read_csv(root_dir + 'single_xgb_001_train.csv')
train_idx = df_train_single_xgb_001['id']
df_test_single_xgb_001 = pd.read_csv(root_dir + 'sub_single_xgb_001_test.csv')
test_id = df_test_single_xgb_001['id']
logging.info('df_train_single_xgb_001 shape : {0}'.format(df_train_single_xgb_001.shape))

# single xgb 002 (LB 283)
df_train_single_xgb_002 = pd.read_csv(root_dir + 'single_xgb_002_train.csv')
df_test_single_xgb_002 = pd.read_csv(root_dir + 'sub_single_xgb_002_test.csv')
logging.info('df_train_single_xgb_002 shape : {0}'.format(df_train_single_xgb_002.shape))

# single xgb 003 (LB 284)
df_train_single_xgb_003 = pd.read_csv(root_dir + 'single_xgb_003_train.csv')
df_test_single_xgb_003 = pd.read_csv(root_dir + 'sub_single_xgb_003_test.csv')
logging.info('df_train_complex_xgb shape : {0}'.format(df_train_single_xgb_003.shape))

# single lgbm 001 ( LB 283 )
df_train_single_lgbm_001 = pd.read_csv(root_dir + 'single_lgbm_001_train.csv')
df_test_single_lgbm_001 = pd.read_csv(root_dir + 'sub_single_lgbm_001_test.csv')
logging.info('df_train_single_lgbm_001 shape : {0}'.format(df_train_single_lgbm_001.shape))

# simple stacker 001 ( LB 284 one mix model)
df_train_simple_stacker_001_mix = pd.read_csv(root_dir + 'simple_stacker_001_mix_train.csv')
df_test_simple_stacker_001_mix = pd.read_csv(root_dir + 'sub_simple_stacker_001_mix_test.csv')

# single rgf 002 ( LB 282 )
df_train_single_rgf_002 = pd.read_csv(root_dir + 'single_rgf_002_train.csv')
df_test_single_rgf_002 = pd.read_csv(root_dir + 'sub_single_rgf_002_test.csv')

# single catboost 001 ( LB 281 )
df_train_single_catboost_001 = pd.read_csv(root_dir + 'single_catboost_001_train.csv')
df_test_single_catboost_001 = pd.read_csv(root_dir + 'sub_single_catboost_001_test.csv')
logging.info('df_train_single_catboost_001 shape : {0}'.format(df_train_single_catboost_001.shape))



def exp(df):
    return 1 / (1 + np.exp(-df))


df_train = pd.DataFrame()
df_train['col1'] = df_train_single_xgb_001['prob']
df_train['col2'] = df_train_single_xgb_002['prob']
df_train['col3'] = df_train_single_xgb_003['prob']
df_train['col4'] = df_train_single_lgbm_001['prob']
df_train['col5'] = df_train_simple_stacker_001_mix['prob']
# df_train['col6'] = df_train_single_rgf_002['prob']
df_train['col7'] = df_train_single_catboost_001['prob']
logging.info('df_train shape : {0}'.format(df_train.shape))

df_test = pd.DataFrame()
df_test['col1'] = df_test_single_xgb_001['target']
df_test['col2'] = df_test_single_xgb_002['target']
df_test['col3'] = df_test_single_xgb_003['target']
df_test['col4'] = df_test_single_lgbm_001['target']
df_test['col5'] = df_test_simple_stacker_001_mix['target']
# df_test['col6'] = df_test_single_rgf_002['target']
df_test['col7'] = df_test_single_catboost_001['target']
logging.info('df_test shape : {0}'.format(df_test.shape))



# Model combination strategies :
#     1. using high lb score singles
#     2. traverse lower lb model combination by cv auc score
#     3. stacker using fast LR
'''
single_lb284 = ['col1', 'col3']
single_lb282 = ['col2']
stack_lb284 = ['col4', 'col5', 'col6', 'col7']
single_lgbm_281 = ['col8']
single_catboost_281 = ['col9']
lgbm_5m_281 = ['col10']

stacker_estimator = LogisticRegression()
models_cv_result = []
models_to_select = single_lb282 + stack_lb284 + single_lgbm_281 + single_catboost_281
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
selected_models = ['col1', 'col3', 'col4', 'col5', 'col7']

stacker = SingleXGB(
    X=df_train[selected_models].values,
    y=df_y.values,
    test=df_test[selected_models],
    N=Number_of_folds,
    skf=comm_skf
)
stacker_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.01,
    'max_depth': 3,
    'seed': 2017,
    'nthread': 5,
    'silent': 1,
}
st(context=21)
DO_CV = True
best_rounds = 300
if DO_CV:
    best_rounds = stacker.cv(
        params=stacker_params,
        num_boost_round=1000,
        feval=GiniEvaluation.gini_xgb,
        feval_name='gini',
        maximize=True,
        metrics=['auc'],
    )
# use oof for single submit file and train file
df_sub = pd.DataFrame()
df_sub['id'] = test_id
df_sub['target'] = 0.0
df_sub, stacker_train = stacker.oof(
    params=stacker_params,
    best_rounds=best_rounds,
    sub=df_sub,
    do_logit=False
)
df_sub.to_csv('../../data/for_stacker/xgb_ensemble.csv', index=False)
logging.info('Creating submit file done')
