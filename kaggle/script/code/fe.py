# encoding=utf8
import pandas as pd
import numpy as np
import logging
import gc
import operator
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import OrderedDict

import xgboost as xgb
import lightgbm as lgbm

from ipdb import set_trace as st


class FeatureImportance(object):
    @staticmethod
    def xgb_fi(params, data, label, feval, maximize, num_boost_round, cv=False):
        if not cv:
            dtrn, deval, ltrn, leval = train_test_split(data, label, test_size=0.25, shuffle=True, random_state=99)
            trainD = xgb.DMatrix(data=dtrn, label=ltrn.values)
            evalD = xgb.DMatrix(data=deval, label=leval.values)
            model = xgb.train(
                params=params,
                dtrain=trainD,
                evals=[(trainD, 'train'), (evalD, 'valid')],
                num_boost_round=num_boost_round,
                feval=feval,
                maximize=maximize,
                verbose_eval=1,
            )
            importance = model.get_fscore()
            importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
            df = pd.DataFrame(importance, columns=['feature', 'fscore'])
            df['fscore'] = df['fscore'] / df['fscore'].sum()
            return df
        else:
            N = 5
            skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2017)
            folds = skf.split(data, label)
            cv_records = xgb.cv(
                params=params,
                dtrain=xgb.DMatrix(data=data, label=label),
                num_boost_round=num_boost_round,
                nfold=N,
                feval=feval,
                maximize=True,
                metrics=['auc'],
                folds=folds,
                early_stopping_rounds=50,
                verbose_eval=5,
            )
            return cv_records

    @staticmethod
    def lgbm_fi(params, df_data, df_label, feval, num_boost_round, feature_watch_list):
        dtrn, deval, ltrn, leval = \
            train_test_split(df_data, df_label, test_size=0.25, shuffle=True, random_state=99)
        model = lgbm.train(
            params=params,
            train_set=lgbm.Dataset(data=dtrn.values, label=ltrn.values),
            valid_sets=lgbm.Dataset(data=deval.values, label=leval.values),
            num_boost_round=num_boost_round,
            feval=feval,
            early_stopping_rounds=50,
            verbose_eval=10,
        )
        feature_importance = MinMaxScaler().fit_transform(model.feature_importance().reshape(-1, 1))
        feature_importance = OrderedDict(
            sorted(
                {name: score[0] for name, score in zip(df_data.columns, feature_importance)}.items(),
                key=lambda x: x[1],
            )
        )
        for feature in feature_watch_list:
            corr = df_data.corrwith(df_data[feature])
            corr.sort_values(inplace=True)
            logging.info('feature \'{0}\' score \'{1}\''.format(feature, feature_importance[feature]))
            logging.info('feature \'{0}\' correlation top10 \'{1}\''.format(feature, corr[0:10]))
            logging.info('feature \'{0}\' correlation bottom10 \'{1}\''.format(feature, corr[-10:]))
        return feature_importance


class Compose(object):
    def __init__(self, transforms_params):
        self.transforms_params = transforms_params

    def __call__(self, df):
        for transform_param in self.transforms_params:
            transform, param = transform_param[0], transform_param[1]
            df = transform(df, **param)
        return df


class Processer(object):
    @staticmethod
    def normalization(df, col_names, val_range=0):
        for col_name in col_names:
            if col_name not in df.columns:
                raise ValueError('col_name {0} not in df'.format(col_name))
        for col_name in col_names:
            _max = df[col_name].max()
            _min = df[col_name].min()
            if val_range==0:
                # range -1 to 1
                df[col_name] = df[col_name].apply(lambda x: ((x - _min) / (_max - _min + 1e-7)) - 0.5)
                df[col_name] = df[col_name].apply(lambda x:2.0*x)
            if val_range==1:
                # range 0 to 1
                df[col_name] = df[col_name].apply(lambda x: ((x - _min) / (_max - _min + 1e-7)))
        return df

    @staticmethod
    def remain_columns(df, col_names):
        logging.info('Before remain columns {0}'.format(df.shape))
        # check col_names
        for col_name in col_names:
            if col_name not in df.columns:
                raise ValueError('col_name {0} not in df'.format(col_name))
        df = df[col_names]
        logging.info('After remain columns {0}'.format(df.shape))
        logging.info('Remain columns {0}'.format(df.columns))
        return df

    @staticmethod
    def drop_columns(df, col_names):
        logging.info('Before drop columns {0}'.format(df.shape))
        df = df.drop(col_names, axis=1)
        logging.info('After drop columns {0}'.format(df.shape))
        return df

    @staticmethod
    def drop_rows(df, row_indexes):
        logging.info('Before drop rows {0}'.format(df.shape))
        df = df.drop(row_indexes, axis=0)
        logging.info('After drop rows {0}'.format(df.shape))
        return df

    @staticmethod
    def dtype_transform(df):
        logging.info('Before data type transform \nfloat64 \n{0} \nint64 \n{1}'.format(
            df.select_dtypes(include=['float64']).columns,
            df.select_dtypes(include=['int64']).columns)
        )
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(np.int8)
        logging.info('After data type transform \nfloat32 \n{0} \nint8 \n{1}'.format(
            df.select_dtypes(include=['float32']).columns,
            df.select_dtypes(include=['int8']).columns)
        )
        return df

    @staticmethod
    def descartes(df, left_col_names, right_col_names):
        logging.info('Before descartes transform {0}'.format(df.shape))
        # check col_names
        for col_name in left_col_names + right_col_names:
            if col_name not in df.columns:
                raise ValueError('col_name {0} not in df'.format(col_name))
        # create new columns by descartes
        descartes_columns = []
        for l_col_name in left_col_names:
            for r_col_name in right_col_names:
                descartes_name = '_x_'.join([l_col_name, r_col_name])
                descartes_columns.append(descartes_name)
                df[descartes_name] = df[l_col_name] * df[r_col_name]
        logging.info('After descartes transform {0}'.format(df.shape))
        logging.info('New created descartes columns {0}'.format(descartes_columns))
        return df

    @staticmethod
    def descartes_interaction(df, col_names):
        logging.info('Before descartes interaction transform {0}'.format(df.shape))
        # check col_names
        for col_name in col_names:
            if col_name not in df.columns:
                raise ValueError('col_name {0} not in df'.format(col_name))
        # create new columns by descartes interaction
        if len(col_names) > 1:
            for index in range(len(col_names) - 1):
                df = Processer.descartes(
                    df,
                    left_col_names=[col_names[index]],
                    right_col_names=col_names[index + 1:]
                )
        else:
            logging.warn('descartes interaction only with one column {0}'.format(col_names[0]))
        logging.info('After descartes interaction transform {0}'.format(df.shape))
        return df

    @staticmethod
    def median_mean_range(df, opt_median=True, opt_mean=True):
        col_names = [c for c in df.columns if (
        '_cat' not in c and '_bin' not in c and '_oh_' not in c and 'negative_one' not in c and '_x_' not in c)]
        logging.info('Columns be median range and mean range transformed {0}'.format(col_names))
        logging.info('Before {0}'.format(df.shape))
        if opt_median:
            _df_median = df[col_names].median(axis=0)
            for col_name in col_names:
                created_col_name = '_'.join([col_name, 'median_range'])
                df[created_col_name] = (df[col_name] > _df_median[col_name]).astype(np.int8)
        if opt_mean:
            _df_mean = df[col_names].mean(axis=0)
            for col_name in col_names:
                created_col_name = '_'.join([col_name, 'mean_range'])
                df[created_col_name] = (df[col_name] > _df_mean[col_name]).astype(np.int8)
        gc.collect()
        logging.info('After {0}'.format(df.shape))
        return df

    @staticmethod
    def negative_one_vals(df):
        df['negative_one_vals'] = MinMaxScaler().fit_transform(df.isnull().sum(axis=1).values.reshape(-1, 1))
        return df

    @staticmethod
    def convert_reg_03(df):
        def ps_reg_03_recon(reg):
            integer = int(np.round((40 * reg) ** 2))
            for f in range(28):
                if (integer - f) % 27 == 0:
                    F = f
                    break
            M = (integer - F) // 27
            return F, M

        df['ps_reg_A_cat'] = df['ps_reg_03'].apply(lambda x: ps_reg_03_recon(x)[0] if not np.isnan(x) else x)
        # df['ps_reg_M'] = df['ps_reg_03'].apply(lambda x: ps_reg_03_recon(x)[1] if not np.isnan(x) else x)
        return df

    @staticmethod
    def add_combine(df_train, df_test, left_column, right_column):
        logging.info('Before add combine : train {0}, test {1}'.format(df_train.shape, df_test.shape))
        add_column = ''.join([left_column, '_plus_', right_column])
        df_train[add_column] = \
            df_train[left_column].apply(lambda x: str(x)) + "_" + df_train[right_column].apply(lambda x: str(x))
        df_test[add_column] = \
            df_test[left_column].apply(lambda x: str(x)) + "_" + df_test[right_column].apply(lambda x: str(x))
        # Label Encode
        lbl = LabelEncoder()
        lbl.fit(list(df_train[add_column].values) + list(df_test[add_column].values))
        df_train[add_column] = lbl.transform(list(df_train[add_column].values))
        df_test[add_column] = lbl.transform(list(df_test[add_column].values))
        logging.info('After add combine : train {0}, test {1}'.format(df_train.shape, df_test.shape))
        gc.collect()
        return df_train, df_test

    @staticmethod
    def ohe(df_train, df_test, cat_features, threshold=50):
        # pay attention train & test should get_dummies together
        logging.info('Before ohe : train {0}, test {1}'.format(df_train.shape, df_test.shape))
        combine = pd.concat([df_train, df_test], axis=0)
        for column in cat_features:
            logging.info('Feature {0} OHE'.format(column))
            temp = pd.get_dummies(pd.Series(combine[column]), prefix=column)
            _abort_cols = []
            for c in temp.columns:
                if temp[c].sum() < threshold:
                    logging.info(
                        'column {0} unique value {1} less than threshold {2}'.format(c, temp[c].sum(), threshold))
                    _abort_cols.append(c)
            logging.info('Abort cat columns : {0}'.format(_abort_cols))
            logging.info('feature {0} ohe columns {1}'.format(column, temp.columns))
            _remain_cols = [c for c in temp.columns if c not in _abort_cols]
            # check category number
            combine = pd.concat([combine, temp[_remain_cols]], axis=1)
            combine = combine.drop([column], axis=1)
        train = combine[:df_train.shape[0]]
        test = combine[df_train.shape[0]:]
        logging.info('After ohe : train {0}, test {1}'.format(train.shape, test.shape))
        return train, test

    @staticmethod
    def ohe_by_unique(df):
        one_hot = {c: list(df[c].unique()) for c in df.columns}
        for c in one_hot:
            if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
                for val in one_hot[c]:
                    df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
        return df
