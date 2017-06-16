#encoding=utf8

from config import FEATURE_NAME_CONNECT
from config import MIN_FEATURE_CATEGORY_NUM, MAX_FEATURE_CATEGORY_NUM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd


class featureProcessor(object):

    @classmethod
    def default_processor(cls, input):
        return input

    @classmethod
    def date_processor(cls, input):
        return 'date'

    @classmethod
    def numeric_processor(cls, input):
        return 'numeric'

    @classmethod
    def none_processor(cls, input):
        return 'none'

    @classmethod
    def one_hot_transformer(cls, df, column_names):
        # Step1. before use one hot encoder, all categories of feature should be encoded
        column_le = {}
        for column in column_names:
            le = LabelEncoder()
            le.fit(df[column])
            if le.classes_.shape[0]>MAX_FEATURE_CATEGORY_NUM or le.classes_.shape[0]<=MIN_FEATURE_CATEGORY_NUM:
                raise ValueError(
                    "\'%s\' can not be label encoded because category num is \'%s\'"
                    % column, le.classes_.shape[0]
                )
            column_le[column] = le
            df[column] = le.transform(df[column])
        # Step2. transform the float-labeled feature to one hot format
        ohe = OneHotEncoder()
        data_derived = ohe.fit_transform(df[column_names]).toarray().astype('float32')
        columns_derived = [ FEATURE_NAME_CONNECT.join([column, str(_class)])
                           for column in column_names for _class in column_le[column].classes_ ]
        # Step3. drop origin feature column
        df.drop(column_names, axis=1, inplace=True)
        # Step4. create new dataframe and return
        return df.join(pd.DataFrame(data=data_derived, columns=columns_derived, index=df.index))


