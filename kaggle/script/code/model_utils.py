# encoding = utf8

import numpy as np
import logging
import time
import gc
import json

import xgboost as xgb
import lightgbm as lgbm
import catboost as cat

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

from ipdb import set_trace as st


class Single(object):
    def __init__(self, X, y, test, skf, N):
        self.X = X
        self.y = y
        self.test = test
        self.skf = skf
        self.N = N


class SingleXGB(Single):
    def grid_search_tuning(self, xgb_param, xgb_param_grid, f_score, n_jobs):
        logging.info('Grid search begin')
        best_score, best_params, grid_scores = \
            self._grid_search_cv(xgb_param, xgb_param_grid, f_score, n_jobs)
        logging.info('Grid search done')
        return best_params

    def _grid_search_cv(self, xgb_param, xgb_param_grid, f_score, n_jobs):
        xgb_estimator = xgb.XGBClassifier(**xgb_param)
        xgb_gs = GridSearchCV(
            estimator=xgb_estimator,
            param_grid=xgb_param_grid,
            cv=self.skf,
            scoring=make_scorer(f_score, greater_is_better=True, needs_proba=True),
            verbose=2,
            n_jobs=n_jobs,
            refit=False
        )
        time_begin = time.time()
        xgb_gs.fit(self.X, self.y)
        time_end = time.time()
        logging.info('Grid search eat time {0} : params {1}'.format(time_end - time_begin, xgb_param_grid))
        logging.info('best_score_ : {0}'.format(xgb_gs.best_score_))
        logging.info('best_params_ : {0}'.format(xgb_gs.best_params_))
        logging.info('grid_scores_ : {0}'.format(xgb_gs.grid_scores_))
        gc.collect()
        return xgb_gs.best_score_, xgb_gs.best_params_, xgb_gs.grid_scores_

    def cv(self, params, num_boost_round, feval, feval_name, maximize, metrics):
        cv_records = xgb.cv(
            params=params,
            dtrain=xgb.DMatrix(data=self.X, label=self.y),
            num_boost_round=num_boost_round,
            feval=feval,
            maximize=maximize,
            metrics=metrics,
            folds=self.skf.split(self.X, self.y),
            early_stopping_rounds=50,
            verbose_eval=5,
        )
        best_rounds = np.argmax(cv_records['test-' + feval_name + '-mean'])
        best_val_score = cv_records['test-' + feval_name + '-mean'][best_rounds]
        logging.info('Best val-{0}-mean: {1} at round {2}.'.format(
            feval_name, best_val_score, best_rounds))
        return best_rounds

    def oof(self, params, best_rounds, sub, do_logit=True):
        stacker_train = np.zeros((self.X.shape[0], 1))
        dtest = xgb.DMatrix(data=self.test.values)
        for index, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            dtrn = xgb.DMatrix(data=trn_x, label=trn_y)
            dval = xgb.DMatrix(data=val_x, label=val_y)
            # train model
            logging.info('Train model in fold {0}'.format(index))
            cv_model = xgb.train(
                params=params,
                dtrain=dtrn,
                num_boost_round=best_rounds,
                verbose_eval=10,
            )
            logging.info('Predict in fold {0}'.format(index))
            prob = cv_model.predict(dtest, ntree_limit=best_rounds)
            stacker_train[val_idx,0] = cv_model.predict(dval, ntree_limit=best_rounds)
            sub['target'] += prob / self.N
        if do_logit:
            sub['target'] = 1 / (1 + np.exp(-sub['target']))
            stacker_train = 1 / (1 + np.exp(-stacker_train))
        logging.info('{0} of folds'.format(self.N))
        logging.info('Oof by single xgboost model Done')
        return sub, stacker_train

class SingleLGBM(Single):

    def cv(self, params, num_boost_round, feval):
        dtrain = lgbm.Dataset(data=self.X, label=self.y)
        bst = lgbm.cv(
            params=params,
            train_set=dtrain,
            nfold=self.N,
            folds=self.skf.split(self.X, self.y),
            num_boost_round=num_boost_round,
            metrics=['auc'],
            feval=feval,
            # early_stopping_rounds=50,
            verbose_eval=10,
        )
        best_rounds = np.argmax(bst['gini-mean']) + 1
        best_score = np.max(bst['gini-mean'])
        logging.info('best rounds : {0}'.format(best_rounds))
        logging.info('best score : {0}'.format(best_score))
        logging.info('lightGBM params : \n{0}'.format(params))
        return best_rounds

    def oof(self, params, best_rounds, sub, do_logit=True):
        stacker_train = np.zeros((self.X.shape[0], 1))
        for index, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            dtrn = lgbm.Dataset(data=trn_x, label=trn_y)
            # train model
            logging.info('Train model in fold {0}'.format(index))
            cv_model = lgbm.train(
                params=params,
                train_set=dtrn,
                num_boost_round=best_rounds,
                verbose_eval=10,
            )
            logging.info('Predict in fold {0}'.format(index))
            prob = cv_model.predict(self.test.values, num_iteration=best_rounds)
            stacker_train[val_idx,0] = cv_model.predict(val_x, num_iteration=best_rounds)
            sub['target'] += prob / self.N
        if do_logit:
            sub['target'] = 1 / (1 + np.exp(-sub['target']))
            stacker_train = 1 / (1 + np.exp(-stacker_train))
        logging.info('{0} of folds'.format(self.N))
        logging.info('Oof by single lightGBM model Done')
        return sub, stacker_train

    def grid_search_tuning(self, lgbm_param, lgbm_param_grid, f_score, n_jobs):
        lgbm_estimator = lgbm.LGBMClassifier(**lgbm_param)
        lgbm_gs = GridSearchCV(
            estimator=lgbm_estimator,
            param_grid=lgbm_param_grid,
            cv=self.skf,
            scoring=make_scorer(f_score, greater_is_better=True, needs_proba=True),
            verbose=2,
            n_jobs=n_jobs,
            refit=False
        )
        time_begin = time.time()
        lgbm_gs.fit(self.X, self.y)
        time_end = time.time()
        logging.info('Grid search eat time {0} : params {1}'.format(time_end - time_begin, lgbm_param_grid))
        logging.info('best_score_ : {0}'.format(lgbm_gs.best_score_))
        logging.info('best_params_ : {0}'.format(lgbm_gs.best_params_))
        for score in lgbm_gs.grid_scores_:
            logging.info('grid_scores_ : {0}'.format(score))
        gc.collect()
        return lgbm_gs.best_params_

    def random_grid_search_tuning(self,lgbm_param, lgbm_param_distribution, f_score, n_jobs, n_iter):
        lgbm_estimator = lgbm.LGBMClassifier(**lgbm_param)
        lgbm_rgs = RandomizedSearchCV(
            estimator=lgbm_estimator,
            param_distributions=lgbm_param_distribution,
            cv=self.skf,
            scoring=make_scorer(f_score, greater_is_better=True, needs_proba=True),
            n_iter=n_iter,
            n_jobs=n_jobs,
            verbose=2,
            refit=False,
        )
        time_begin = time.time()
        lgbm_rgs.fit(self.X, self.y)
        time_end = time.time()
        logging.info('Random grid search eat time {0}'.format(time_end - time_begin))
        logging.info('best_score_ : {0}'.format(lgbm_rgs.best_score_))
        logging.info('best_params_ : {0}'.format(lgbm_rgs.best_params_))
        for score in lgbm_rgs.grid_scores_:
            logging.info('grid_scores_ : {0}'.format(score))
        gc.collect()
        return lgbm_rgs.best_params_

class SingleRF(Single):
    def random_grid_search_tuning(self,rf_param, rf_param_distribution, f_score, n_jobs, n_iter):
        rf_estimator = RandomForestClassifier(**rf_param)
        rf_rgs = RandomizedSearchCV(
            estimator=rf_estimator,
            param_distributions=rf_param_distribution,
            cv=self.skf,
            scoring=make_scorer(f_score, greater_is_better=True, needs_proba=True),
            n_iter=n_iter,
            n_jobs=n_jobs,
            verbose=2,
            refit=False,
        )
        time_begin = time.time()
        rf_rgs.fit(self.X, self.y)
        time_end = time.time()
        logging.info('Random grid search eat time {0}'.format(time_end - time_begin))
        logging.info('best_score_ : {0}'.format(rf_rgs.best_score_))
        logging.info('best_params_ : {0}'.format(rf_rgs.best_params_))
        for score in rf_rgs.grid_scores_:
            logging.info('grid_scores_ : {0}'.format(score))
        gc.collect()
        return rf_rgs.best_params_

    def oof(self, params, sub, do_logit=True):
        rf = RandomForestClassifier(**params)
        stacker_train = np.zeros((self.X.shape[0], 1))
        for index, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            # train model
            logging.info('Train model in fold {0}'.format(index))
            rf.fit(trn_x, trn_y)
            logging.info('Predict in fold {0}'.format(index))
            stacker_train[val_idx,0] = rf.predict_proba(val_x)[:,1]
            prob = rf.predict_proba(self.test.values)[:,1]
            sub['target'] += prob / self.N
        if do_logit:
            sub['target'] = 1 / (1 + np.exp(-sub['target']))
            stacker_train = 1 / (1 + np.exp(-stacker_train))
        logging.info('{0} of folds'.format(self.N))
        logging.info('Oof by single random forest model Done')
        return sub, stacker_train

class SingleCatboost(Single):
    def grid_search_tuning(self, cat_param, cat_param_grid, f_score, n_jobs):
        cat_estimator = cat.CatBoostClassifier(**cat_param)
        cat_gs = GridSearchCV(
            estimator=cat_estimator,
            param_grid=cat_param_grid,
            cv=self.skf,
            scoring=make_scorer(f_score, greater_is_better=True, needs_proba=True),
            verbose=2,
            n_jobs=n_jobs,
            refit=False
        )
        time_begin = time.time()
        cat_gs.fit(self.X, self.y)
        st(context=21)
        time_end = time.time()
        logging.info('Grid search eat time {0} : params {1}'.format(time_end - time_begin, cat_param_grid))
        logging.info('best_score_ : {0}'.format(cat_gs.best_score_))
        logging.info('best_params_ : {0}'.format(cat_gs.best_params_))
        for score in cat_gs.grid_scores_:
            logging.info('grid_scores_ : {0}'.format(score))
        gc.collect()
        return cat_gs.best_score_, cat_gs.best_params_, cat_gs.grid_scores_

    def cv(self, params, num_boost_round, feval):
        dtrain = lgbm.Dataset(data=self.X, label=self.y)
        bst = lgbm.cv(
            params=params,
            train_set=dtrain,
            nfold=self.N,
            folds=self.skf.split(self.X, self.y),
            num_boost_round=num_boost_round,
            metrics=['auc'],
            feval=feval,
            early_stopping_rounds=50,
            verbose_eval=10,
        )
        best_rounds = np.argmax(bst['gini-mean']) + 1
        best_score = np.max(bst['gini-mean'])
        logging.info('best rounds : {0}'.format(best_rounds))
        logging.info('best score : {0}'.format(best_score))
        logging.info('lightGBM params : \n{0}'.format(params))
        return best_rounds
