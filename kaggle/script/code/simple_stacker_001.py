# encoding=utf8
#################################################################################
# from kernel https://www.kaggle.com/yekenot/simple-stacker-lb-0-284?scriptVersionId=1649456
# LB 0.284
#################################################################################
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier

from logging_manage import initialize_logger
import logging
import os
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')


train = pd.read_csv('../../data/input/train.csv')
test = pd.read_csv('../../data/input/test.csv')

# Preprocessing (Forza Baseline)
id_test = test['id'].values

col = [c for c in train.columns if c not in ['id', 'target']]
col = [c for c in col if not c.startswith('ps_calc_')]

train = train.replace(-1, np.NaN)
train.drop([149161], axis=0, inplace=True)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id', 'target']}


def transform(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


train = transform(train)
test = transform(test)

col = [c for c in train.columns if c not in ['id', 'target']]
col = [c for c in col if not c.startswith('ps_calc_')]

# dups = train[train.duplicated(subset=col, keep=False)]
#
# train = train[~(train['id'].isin(dups['id'].values))]

target_train = train['target']
id_train = train['id']
train = train[col]
test = test[col]
logging.info('train shape {0}, test shape {1}'.format(train.values.shape, test.values.shape))


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2017).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]

                logging.info("Fit {0} fold {1}".format(str(clf).split('(')[0], j + 1))
                clf.fit(X=X_train, y=y_train)
                #                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
                #                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:, 1]
                # record in-fold predict results
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='roc_auc')
        logging.info("Stacker score: {0}".format(results.mean()))

        self.stacker.fit(S_train, y)
        res_test = self.stacker.predict_proba(S_test)[:, 1]
        res_train = self.stacker.predict_proba(S_train)[:,1]
        return res_test, res_train, S_train, S_test


# LightGBM params
lgb_params_1 = {
    'learning_rate': 0.01,
    'n_estimators': 1250,
    # 'n_estimators': 5,
    'max_bin': 10,
    'subsample': 0.8,
    'subsample_freq': 10,
    'colsample_bytree': 0.8,
    'min_child_samples': 500,
    'nthread': 5
}

lgb_params_2 = {
    'learning_rate': 0.005,
    'n_estimators': 3700,
    # 'n_estimators': 5,
    'subsample': 0.7,
    'subsample_freq': 2,
    'colsample_bytree': 0.3,
    'num_leaves': 16,
    'nthread': 5
}

lgb_params_3 = {
    'learning_rate': 0.02,
    'n_estimators': 800,
    # 'n_estimators': 5,
    'max_depth': 4,
    'nthread': 5
}

# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.02
xgb_params['n_estimators'] = 1000
# xgb_params['n_estimators'] = 5
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9
xgb_params['min_child_weight'] = 10
xgb_params['nthread'] = 5


xgb_model = XGBClassifier(**xgb_params)

lgb_model_1 = LGBMClassifier(**lgb_params_1)

lgb_model_2 = LGBMClassifier(**lgb_params_2)

lgb_model_3 = LGBMClassifier(**lgb_params_3)

log_model = LogisticRegression()

stack = Ensemble(
    n_splits=5,
    stacker=log_model,
    base_models=(xgb_model, lgb_model_1, lgb_model_2, lgb_model_3)
)

y_pred, y_pred_train, s_train, s_test = stack.fit_predict(train, target_train, test)

## Mix predict result for stack ensemble
# ***pay attention to the value scale because they were transformed by logistic
# mix predict on test by simple stacker
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('../../data/for_stacker/sub_simple_stacker_001_mix_test.csv', index=False)
# mix predict on train by simple stacker
sub_train = pd.DataFrame()
sub_train['id'] = id_train
sub_train['prob'] = y_pred_train
sub_train.to_csv('../../data/for_stacker/simple_stacker_001_mix_train.csv', index=False)

## Individual predict result for stack ensemble
# predict on train by four single models
stacker_train = pd.DataFrame()
stacker_train['id'] = id_train
stacker_train['xgb'] = s_train[:,0]
stacker_train['lgb1'] = s_train[:,1]
stacker_train['lgb2'] = s_train[:,2]
stacker_train['lgb3'] = s_train[:,3]
stacker_train.to_csv('../../data/for_stacker/simple_stacker_001_individual_train.csv', index=False)
# predict on test by four single models
stacker_test = pd.DataFrame()
stacker_test['id'] = sub['id']
stacker_test['xgb'] = s_test[:,0]
stacker_test['lgb1'] = s_test[:,1]
stacker_test['lgb2'] = s_test[:,2]
stacker_test['lgb3'] = s_test[:,3]
stacker_test.to_csv('../../data/for_stacker/sub_simple_stacker_001_individual_test.csv', index=False)

