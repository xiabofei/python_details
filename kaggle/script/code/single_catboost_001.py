# encoding=utf8
#################################################################################
# https://www.kaggle.com/aharless/simple-catboost-cv-lb-281
# LB 0.281
#################################################################################

MAX_ROUNDS = 650
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.05

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from model_utils import SingleCatboost
from io_utils import comm_skf, write_data, Number_of_folds
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from evaluation import gini_score

from logging_manage import initialize_logger
import logging
initialize_logger(output_dir='../../data/log/')

# Compute gini
def eval_gini(a, p):
    def _gini(actual, pred):
        assert (len(actual) == len(pred))
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
    return _gini(a, p) / _gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]

# Read data
train_df = pd.read_csv('../../data/input/train.csv', na_values="-1")
train_df.drop([149161], axis=0, inplace=True)
test_df = pd.read_csv('../../data/input/test.csv', na_values="-1")

# Process data
id_test = test_df['id'].values
id_train = train_df['id'].values

train_df = train_df.fillna(999)
test_df = test_df.fillna(999)

col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
train_df = train_df.drop(col_to_drop, axis=1)
test_df = test_df.drop(col_to_drop, axis=1)

for c in train_df.select_dtypes(include=['float64']).columns:
    train_df[c] = train_df[c].astype(np.float32)
    test_df[c] = test_df[c].astype(np.float32)
for c in train_df.select_dtypes(include=['int64']).columns[2:]:
    train_df[c] = train_df[c].astype(np.int8)
    test_df[c] = test_df[c].astype(np.int8)

y = train_df['target']
X = train_df.drop(['target', 'id'], axis=1)
y_valid_pred = 0 * y
X_test = test_df.drop(['id'], axis=1)
y_test_pred = 0

logging.info('Train data shape : {0}'.format(X.shape))
logging.info('Test data shape : {0}'.format(X_test.shape))


# Set up folds
skf = comm_skf

# Set up classifier
'''
cat_param = dict(
    learning_rate=LEARNING_RATE,
    depth=6,
    l2_leaf_reg = 14,
    rsm=1,
    iterations = MAX_ROUNDS,
    loss_function='Logloss',
    random_seed=2017,
    thread_count=2,
)
cat_param_grid = dict(
    depth=[5, 6, 7],
    l2_leaf_reg = [11, 13, 14, 15, 16],
    rsm=[0.7, 0.8, 0.9],
)
best_score, best_params, _ = single_catboost.grid_search_tuning(
    cat_param=cat_param,
    cat_param_grid=cat_param_grid,
    f_score=gini_score,
    n_jobs=5,
)
'''
single_catboost = SingleCatboost(X=X, y=y, test=X_test, skf=skf, N=Number_of_folds)

# Run CV
cat_param_submit = dict(
    learning_rate=LEARNING_RATE,
    depth=6,
    l2_leaf_reg = 16,
    rsm=0.9,
    iterations = MAX_ROUNDS,
    loss_function='Logloss',
    random_seed=2017,
    thread_count=2,
)
model = CatBoostClassifier(**cat_param_submit)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index, :], X.iloc[test_index, :]
    logging.info("Fold {0}".format(i))
    # Run model for this fold
    fit_model = model.fit(X_train, y_train)
    pred = fit_model.predict_proba(X_valid)[:, 1]
    logging.info("  Gini = {0}".format(eval_gini(y_valid, pred)))
    y_valid_pred.iloc[test_index] = pred
    y_test_pred += fit_model.predict_proba(X_test)[:, 1]

y_test_pred /= Number_of_folds  # Average test set predictions

logging.info("Gini for full training set: {0}".format(eval_gini(y, y_valid_pred)))

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['prob'] = y_valid_pred.values
val.to_csv('../../data/for_stacker/single_catboost_001_train.csv', float_format='%.7f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('../../data/for_stacker/sub_single_catboost_001_test.csv', float_format='%.7f', index=False)
