# encoding=utf8
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from config import CHUNK_SIZE_THRESHOLD
from config import MIN_FEATURE_CATEGORY_NUM, MAX_FEATURE_CATEGORY_NUM
from config import FEATURE_NAME_CONNECT
from config import ONE_HOT_ENCODER, HML_CONVERT, NA_FILTER
from util import OneToOneConvert
from collections import defaultdict, OrderedDict
import os
import numpy as np

NAN = np.NaN

from ipdb import set_trace as st


class featureConvert(object):
    __doc__ = "做特征转换的类:输入是csv的文本, 每一列是一个原始的feature;输出也是csv的文本, 每一列是处理后的feature"

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

    def pre_process_label(self, chunk_size=500000):
        column_labels = \
            {k: defaultdict(int) for k, v in self.feature_processor.items() if v.get(ONE_HOT_ENCODER, False)}
        # go through the data chunk by chunk
        # st(context=21)
        for df in self.load_from_csv(chunk_size=chunk_size):
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

    def category_feature_transform(self, chunk_size=500000):
        # delete existing file
        if os.path.isfile(self.output_csv_path):
            os.remove(self.output_csv_path)
        # conduct one hot transform and write to output_csv_path
        with open(self.output_csv_path, 'a') as f:
            _header = True
            for df in fc.load_from_csv(chunk_size=chunk_size):
                # st(context=21)
                df = fc._one_hot_transform(df)
                df.to_csv(f, index=False, header=_header)
                _header = False

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
            df.drop(columns_to_encode, axis=1, inplace=True)
            # create new dataFrame and return
            # return df.join(pd.DataFrame(data=data_derived, columns=columns_derived, index=df.index))
            return pd.DataFrame(data=data_derived, columns=columns_derived, index=df.index)
        else:
            raise AttributeError("instance has no attribute \'column_labels\'")

    def _fill_na_in_pandas(self, df, column):
        return df[column].fillna(NAN)

    def filter_row_contain_na_value(self, df, columns, convert_type):
        return df[pd.notnull(df[columns])]

    def one_to_one_convert(self, df, columns, convert_type):
        if convert_type == HML_CONVERT:
            for item in columns:
                _col, _min, _max = item[0], item[1][0], item[1][1]
                df[_col] = df[_col].apply(OneToOneConvert.H_M_L_convert, args=(_min, _max))
        else:
            pass

    name_action = {
        NA_FILTER: filter_row_contain_na_value,
        HML_CONVERT: one_to_one_convert
    }

    def flow_execute_engine(self):
        if hasattr(self, 'mining_flow'):
            for index, flow in enumerate(self.mining_flow):
                if index==0:
                    self._execute_single_flow_actions(index, flow)
                else:
                    self._execute_single_flow_actions(index, flow)
        else:
            raise AttributeError("instance has no attribute \'mining_flow\'")

    def _execute_single_flow_actions(self, index, flow):
        output_file_name = './' + str(index) + '_flow.csv'
        # load in chunk
        _header = True
        for df in self.load_from_csv(chunk_size=self.chunk_size):
            # conduct each action in this single flow
            for name, columns in flow.items():
                df = self.name_action[name](df, columns, name)
            df.to_csv(output_file_name, index=False, header=_header)
            _header = False


if __name__ == '__main__':
    input_csv_path = './xiehe.csv'
    # 1) list中的每个dict结构相当于走了一次文件
    # 2) dict中的各个操作, 通过OrderedDict来控制先后顺序
    xiehe_mining_flow = [
        OrderedDict(
            (
                NA_FILTER,
                ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
                 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ', 'DS', 'IY']
            ),
            (
                HML_CONVERT,
                [('BD', (1.98, 23)), ('BE', (0.23, 82.4)), ('BF', (0.46, 6.03)),
                 ('BG', (50, 386)), ('BH', (210, 667)), ('BI', (5, 101))]
            )
        ),
        OrderedDict(
            (
                ONE_HOT_ENCODER,
                ['P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'AE', 'AF', 'AG', 'AH', 'BN', 'BP', 'BQ', 'IY']
            )
        )
    ]
    fc = featureConvert(input_csv_path, xiehe_mining_flow, header=0)
    fc.pre_process_label(chunk_size=50)
    fc.category_feature_transform(chunk_size=50)
    # st(context=21)
    print 'exit'
