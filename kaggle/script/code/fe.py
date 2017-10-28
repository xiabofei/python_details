# encoding=utf8
import pandas as pd
import numpy as np
import logging

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
    def

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



