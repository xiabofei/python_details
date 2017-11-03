# encoding=utf8
# frome kernel https://www.kaggle.com/yekenot/simple-stacker-lb-0-284?scriptVersionId=1649456
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


train = pd.read_csv('../../data/input/train.csv')
test = pd.read_csv('../../data/input/test.csv')

# Preprocessing (Forza Baseline)
id_test = test['id'].values

col = [c for c in train.columns if c not in ['id', 'target']]
col = [c for c in col if not c.startswith('ps_calc_')]

train = train.replace(-1, np.NaN)
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

dups = train[train.duplicated(subset=col, keep=False)]

train = train[~(train['id'].isin(dups['id'].values))]

target_train = train['target']
train = train[col]
test = test[col]
print(train.values.shape, test.values.shape)


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j + 1))
                clf.fit(X=X_train, y=y_train)
                #                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
                #                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:, 1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:, 1]
        return res


# LightGBM params
lgb_params_1 = {
    'learning_rate': 0.01,
    'n_estimators': 1250,
    'max_bin': 10,
    'subsample': 0.8,
    'subsample_freq': 10,
    'colsample_bytree': 0.8,
    'min_child_samples': 500,
    'n_jobs': 5
}

lgb_params_2 = {
    'learning_rate': 0.005,
    'n_estimators': 3700,
    'subsample': 0.7,
    'subsample_freq': 2,
    'colsample_bytree': 0.3,
    'num_leaves': 16,
    'n_jobs': 5
}

lgb_params_3 = {
    'learning_rate': 0.02,
    'n_estimators': 800,
    'max_depth': 4,
    'n_jobs': 5
}

# RandomForest params
# rf_params = {}
# rf_params['n_estimators'] = 200
# rf_params['max_depth'] = 6
# rf_params['min_samples_split'] = 70
# rf_params['min_samples_leaf'] = 30


# ExtraTrees params
# et_params = {}
# et_params['n_estimators'] = 155
# et_params['max_features'] = 0.3
# et_params['max_depth'] = 6
# et_params['min_samples_split'] = 40
# et_params['min_samples_leaf'] = 18


# XGBoost params 1
xgb_params1 = {}
xgb_params1['objective'] = 'binary:logistic'
xgb_params1['learning_rate'] = 0.03
xgb_params1['n_estimators'] = 600
xgb_params1['max_depth'] = 5
xgb_params1['min_child_weight'] = 9
xgb_params1['gamma'] = 0.4
xgb_params1['subsample'] = 0.9
xgb_params1['colsample_bytree'] = 0.5
xgb_params1['reg_alpha'] = 1e-4
xgb_params1['reg_lambda'] = 20
xgb_params1['scale_pos_weight'] = 0.41
xgb_params1['nthread'] = 4

# XGBoost params 2
xgb_params2 = {}
xgb_params2['objective'] = 'binary:logistic'
xgb_params2['learning_rate'] = 0.02
xgb_params2['n_estimators'] = 1000
xgb_params2['max_depth'] = 4
xgb_params2['subsample'] = 0.9
xgb_params2['colsample_bytree'] = 0.9
xgb_params2['min_child_weight'] = 10
xgb_params2['nthread'] = 4

# CatBoost params
# cat_params = {}
# cat_params['iterations'] = 900
# cat_params['depth'] = 8
# cat_params['rsm'] = 0.95
# cat_params['learning_rate'] = 0.03
# cat_params['l2_leaf_reg'] = 3.5
# cat_params['border_count'] = 8
# cat_params['gradient_iterations'] = 4


# Regularized Greedy Forest params
# rgf_params = {}
# rgf_params['max_leaf'] = 2000
# rgf_params['learning_rate'] = 0.5
# rgf_params['algorithm'] = "RGF_Sib"
# rgf_params['test_interval'] = 100
# rgf_params['min_samples_leaf'] = 3
# rgf_params['reg_depth'] = 1.0
# rgf_params['l2'] = 0.5
# rgf_params['sl2'] = 0.005


xgb_model_1 = XGBClassifier(**xgb_params1)

xgb_model_2 = XGBClassifier(**xgb_params2)

lgb_model_1 = LGBMClassifier(**lgb_params_1)

lgb_model_2 = LGBMClassifier(**lgb_params_2)

lgb_model_3 = LGBMClassifier(**lgb_params_3)

# rf_model = RandomForestClassifier(**rf_params)

# et_model = ExtraTreesClassifier(**et_params)


# cat_model = CatBoostClassifier(**cat_params)

# rgf_model = RGFClassifier(**rgf_params)

# gb_model = GradientBoostingClassifier(max_depth=5)

# ada_model = AdaBoostClassifier()

log_model = LogisticRegression()

stack = Ensemble(
    n_splits=5,
    stacker=log_model,
    base_models=(xgb_model_1, xgb_model_2, lgb_model_1, lgb_model_2, lgb_model_3)
)

y_pred = stack.fit_predict(train, target_train, test)

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('../../data/output/stacked.csv', index=False)