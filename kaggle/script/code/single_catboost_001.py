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
from io_utils import *
from fe import *

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


## Data Loading
df_train, df_y, df_test, df_sub, train_id = read_data()

## Data Processing and Feature Engineering
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

# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

# Set up folds
skf = comm_skf


# Set up classifier
single_catboost = SingleCatboost(X=X, y=y, test=df_test, skf=skf, N=Number_of_folds)

cat_param = dict(
    learning_rate=LEARNING_RATE,
    depth=6,
    l2_leaf_reg = 16,
    rsm=1,
    iterations = MAX_ROUNDS,
    loss_function='Logloss',
    random_seed=2017,
    thread_count=2,
    od_wait=100,
)
cat_param_distribution = dict(
    l2_leaf_reg = list(set(np.random.uniform(15,20,200))),
    rsm=list(set(np.random.uniform(0.8,0.999, 200))),
    nan_mode =['Min', 'Max'],
    bagging_temperature=list(set(np.random.uniform(0.7,1,200))),
    approx_on_full_history=[True,False],
)
best_score = single_catboost.random_grid_search_tuning(
    cat_param=cat_param,
    cat_param_distribution=cat_param_distribution,
    f_score=gini_score,
    n_iter=50,
    n_jobs=5
)

'''
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
'''
