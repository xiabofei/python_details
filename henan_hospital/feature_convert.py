# encoding=utf8
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from config import CHUNK_SIZE_THRESHOLD
from config import MIN_FEATURE_CATEGORY_NUM, MAX_FEATURE_CATEGORY_NUM
from config import FEATURE_NAME_CONNECT
from config import ONE_HOT_ENCODER, HML_CONVERT, NA_FILTER, BINARY_CONVERT, NORMALIZATION_CONVERT, COLUMN_FILTER
from config import TWO_TO_ONE_CONVERT, USER_DEFINED_ONE_TO_ONE
from config import MINUS,PLUS,MULTIPLY, DERIVE_AGE_FROM_DATE
from config import MERGE_POS_NEG
from util import OneToOneConvert, NToOneConvert
from collections import defaultdict, OrderedDict
import os

NAN = 'nan'

from ipdb import set_trace as st


class featureConvert(object):
    def __init__(self,
                 raw_csv_path,
                 mining_flow,
                 sep=',',
                 quote_char='"',
                 header=None,
                 chunk_size=50):
        self.raw_csv_path = raw_csv_path
        self.mining_flow = mining_flow
        self.sep = sep
        self.quote_char = quote_char
        self.header = header
        self.chunk_size = chunk_size

    def __setattr__(self, key, value):
        if key == 'mining_flow':
            if isinstance(value, list):
                object.__setattr__(self, key, value)
            else:
                raise TypeError("feature_processor is not list type")
        object.__setattr__(self, key, value)

    def load_from_csv(self, input_path, chunk_size=CHUNK_SIZE_THRESHOLD):
        return pd.read_csv(
            input_path,
            quotechar=self.quote_char,
            sep=self.sep,
            header=self.header,
            dtype=object,
            index_col=False,
            chunksize=chunk_size
        )

    def pre_process_label(self, input_path, columns, chunk_size=500000):
        column_labels = {column: defaultdict(int) for column in columns}
        # go through the data chunk by chunk
        for df in self.load_from_csv(input_path, chunk_size=chunk_size):
            for column in column_labels.keys():
                df[column] = self._fill_na_in_pandas(df, column)
                labels, counts = df[column].value_counts().keys().values, df[column].value_counts().values
                for l, c in zip(labels, counts):
                    column_labels[column][l] += c
        # filter label data
        max_cat = MAX_FEATURE_CATEGORY_NUM
        min_cat = MIN_FEATURE_CATEGORY_NUM
        column_labels_filtered = {k: v for k, v in column_labels.items() if
                                  len(v.keys()) > min_cat and len(v.keys()) < max_cat}
        self.column_labels = column_labels_filtered

    def _fill_na_in_pandas(self, df, column):
        return df[column].fillna(NAN)

    def filter_row_contain_na_value(self, df, columns, name):
        _df = df.loc[:, columns]
        _df.dropna(axis=0, inplace=True)
        return df.ix[_df.index, :]

    def one_to_one_convert(self, df, columns, name):
        if name == HML_CONVERT:
            for item in columns:
                _col, _min, _max = item[0], item[1][0], item[1][1]
                df[_col] = df[_col].astype('float32')
                df[_col] = df[_col].apply(OneToOneConvert.H_M_L_convert, args=(float(_min), float(_max),))
            return df
        else:
            pass

    def one_hot_transform(self, df, columns, name):
        return self._one_hot_transform(df)

    def _one_hot_transform(self, df):
        if hasattr(self, 'column_labels'):
            column_le = {}
            # use label to encode data
            try:
                for column, labels in self.column_labels.items():
                    le = LabelEncoder()
                    le.fit(labels.keys())
                    column_le[column] = le
                    df[column] = le.transform(self._fill_na_in_pandas(df, column))
            except Exception, e:
                st(context=21)
                print 'hh'
            # transform category data to one hot
            columns_to_encode = column_le.keys()
            _n_values = [len(self.column_labels[column].keys()) for column in columns_to_encode]
            ohe = OneHotEncoder(n_values=_n_values)
            data_derived = ohe.fit_transform(df[columns_to_encode]).toarray().astype('float32')
            columns_derived = [FEATURE_NAME_CONNECT.join([column, str(_class)])
                               for column in columns_to_encode for _class in column_le[column].classes_]
            # drop origin feature column
            # df.drop(columns_to_encode, axis=1, inplace=True)
            # create new dataFrame and return
            return df.join(pd.DataFrame(data=data_derived, columns=columns_derived, index=df.index))
            # return pd.DataFrame(data=data_derived, columns=columns_derived, index=df.index)
        else:
            raise AttributeError("instance has no attribute \'column_labels\'")

    def binary_convert(self, df, columns, name):
        for column in columns:
            _name, _one = column[0], column[1]
            df[_name] = df[_name].apply(OneToOneConvert.binary_convert, args=(_one,))
        return df

    def normalization_convert(self, df, columns, name):
        for column in columns:
            _name, _min, _max = column[0], column[1][0], column[1][1]
            df[_name] = df[_name].astype('float32')
            df[_name] = df[_name].apply(OneToOneConvert.normalization_convert, args=(_min, _max,))
        return df

    def column_filter(self, df, columns, name):
        return df[columns]

    def flow_execute_engine(self):
        # st(context=21)
        if hasattr(self, 'mining_flow'):
            for index, flow in enumerate(self.mining_flow):
                if index == 0:
                    self._execute_single_flow_actions(index, flow, self.raw_csv_path)
                else:
                    self._execute_single_flow_actions(index, flow, self._flow_file_name(index - 1))
        else:
            raise AttributeError("instance has no attribute \'mining_flow\'")

    def _execute_single_flow_actions(self, index, flow, last_flow_file):
        output_path = self._flow_file_name(index)
        # st(context=21)
        if os.path.isfile(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f_output:
            _header = True
            _not_pre_process_label = True
            for df in self.load_from_csv(last_flow_file, chunk_size=self.chunk_size):
                for name, columns in flow.items():
                    if name == ONE_HOT_ENCODER and _not_pre_process_label:
                        self.pre_process_label(last_flow_file, columns, chunk_size=self.chunk_size)
                        _not_pre_process_label = False
                    df = getattr(self, self.name_action[name])(df, columns, name)
                df.to_csv(f_output, quotechar=self.quote_char, index=False, header=_header)
                _header = False

    def _flow_file_name(self, index):
        return str(index) + '_flow'

    def two_to_one_convert(self, df, columns, name):
        for column in columns:
            _col1, _col2, _action, _derived_name = column[0], column[1], column[2], column[3]
            df[_derived_name] = df.apply(NToOneConvert.two_to_one_convert, args=(_col1, _col2, _action,), axis=1)
        return df

    def user_defined_one_to_one(self, df, columns, name):
        for column in columns:
            _name, _method, _strategy = column[0], column[1], column[2]
            df[_name] = self._fill_na_in_pandas(df, _name)
            df[_name] = df[_name].apply(_method, args=(_strategy,))
        return df


    name_action = {
        NA_FILTER: 'filter_row_contain_na_value',
        HML_CONVERT: 'one_to_one_convert',
        ONE_HOT_ENCODER: 'one_hot_transform',
        BINARY_CONVERT: 'binary_convert',
        NORMALIZATION_CONVERT: 'normalization_convert',
        COLUMN_FILTER: 'column_filter',
        TWO_TO_ONE_CONVERT: 'two_to_one_convert',
        USER_DEFINED_ONE_TO_ONE: 'user_defined_one_to_one'
    }


if __name__ == '__main__':
    # raw_csv_path = './xiehe.csv'
    # xiehe_mining_flow = [
    #     OrderedDict(
    #         [
    #             (
    #                 COLUMN_FILTER,
    #                 ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
    #                  'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ',
    #                  'DS', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JF']
    #             ),
    #             (
    #                 NA_FILTER,
    #                 ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
    #                  'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ', 'DS', 'IY']
    #             ),
    #         ]
    #     ),
    #     OrderedDict(
    #         [
    #             (
    #                 NORMALIZATION_CONVERT,
    #                 [('AK', (26, 81)), ('BJ', (36, 862)), ('BK', (0.2, 70)), ('DS', (0, 98))]
    #             ),
    #             # (
    #             #     HML_CONVERT,
    #             #     [('BD', (1.98, 23)), ('BE', (0.23, 82.4)), ('BF', (0.46, 6.03)),
    #             #      ('BG', (50, 386)), ('BH', (210, 667)), ('BI', (5, 101))]
    #             # ),
    #             (
    #                 BINARY_CONVERT,
    #                 [('IZ', "有"), ('JA', "有"), ('JB', "有"), ('JC', "有"), ('JF', "有")]
    #             ),
    #         ]
    #     ),
    #     OrderedDict(
    #         [
    #             (
    #                 ONE_HOT_ENCODER,
    #                 ['P', 'Q', 'R', 'S', 'T', 'U', 'V',
    #                  'AE', 'AF', 'AG', 'AH', 'BN', 'BP', 'BQ', 'IY']
    #             ),
    #         ]
    #     ),
    #     OrderedDict(
    #         [
    #             (
    #                 TWO_TO_ONE_CONVERT,
    #                 [('BD','BE', PLUS, 'BD_BE')]
    #             ),
    #             (
    #                 COLUMN_FILTER,
    #                 ['BD','BE','BD_BE']
    #             ),
    #         ]
    #     ),
    # ]
    raw_csv_path = './C13.csv'
    xiehe_mining_flow = [
        OrderedDict(
            [
                (
                    TWO_TO_ONE_CONVERT,
                    [('PECR_CheckDate','PAPAT_DE_Dob', DERIVE_AGE_FROM_DATE, 'DER_AGE')]
                ),
                (
                    USER_DEFINED_ONE_TO_ONE,
                    [
                        ('C-14_碳14吹气试验', OneToOneConvert.negative_or_positive_convert, 'one'),
                        ('C-13_碳13吹气试验', OneToOneConvert.negative_or_positive_convert, 'two'),
                        ('PAPAT_DE_SexCode', OneToOneConvert.negative_or_positive_convert, 'three'),
                    ]
                ),
                # (
                #     COLUMN_FILTER,
                #     ['PECR_CheckDate', 'PAPAT_DE_Dob', 'DER_AGE', 'C-14_碳14吹气试验','C-13_碳13吹气试验','PAPAT_DE_SexCode']
                # ),
            ]
        ),
        OrderedDict(
            [
                (
                    TWO_TO_ONE_CONVERT,
                    [('C-14_碳14吹气试验','C-13_碳13吹气试验', MERGE_POS_NEG, 'COMPREHENSIVE_NP')]
                ),
            ]
        ),
    ]
    fc = featureConvert(raw_csv_path, xiehe_mining_flow, header=0)
    fc.flow_execute_engine()
    print 'exit'
