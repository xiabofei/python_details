# encoding=utf8
import pandas as pd
import numpy as np
import logging
import gc


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
    def median_mean_range(df, opt_median=True, opt_mean=True):
        col_names = [ c for c in df.columns if ('_bin' not in c and '_cat' not in c) ]
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
    def ohe(df_train, df_test, cat_features):
        # pay attention train & test should get_dummies together
        logging.info('Before ohe : train {0}, test {1}'.format(df_train.shape, df_test.shape))
        combine = pd.concat([df_train, df_test], axis=0)
        for column in cat_features:
            temp = pd.get_dummies(pd.Series(combine[column]))
            combine = pd.concat([combine, temp], axis=1)
            combine = combine.drop([column], axis=1)
        train = combine[:df_train.shape[0]]
        test = combine[df_train.shape[0]:]
        logging.info('After ohe : train {0}, test {1}'.format(train.shape, test.shape))
        return train, test
