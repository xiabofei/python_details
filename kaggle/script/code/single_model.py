# encoding = utf8

import numpy as np
import logging
import time
import gc

import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from ipdb import set_trace as st


class Single(object):
    def __init__(self, X, y, test, N, skf):
        self.X = X
        self.y = y
        self.test = test
        self.N = N
        self.skf = skf


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
            nfold=self.N,
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

    def oof(self, params, best_rounds, feval, maximize, sub):
        dtest = xgb.DMatrix(data=self.test.values)
        for trn_idx, val_idx in self.skf.split(self.X, self.y):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            dtrn = xgb.DMatrix(data=trn_x, label=trn_y)
            dval = xgb.DMatrix(data=val_x, label=val_y)
            # train model
            cv_model = xgb.train(
                params=params,
                dtrain=dtrn,
                evals=[(dval, 'val')],
                num_boost_round=best_rounds,
                feval=feval,
                maximize=maximize,
                early_stopping_rounds=50,
                verbose_eval=10,
            )
            sub['target'] += cv_model.predict(dtest, ntree_limit=best_rounds)
        logging.info('{0} of folds'.format(self.N))
        sub['target'] = sub['target'] / self.N
        logging.info('Oof by single xgboost model Done')
        return sub


class SingleLGBM(Single):
    pass
