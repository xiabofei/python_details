# encoding=utf8

from config import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict
import pandas as pd
from eat_io import IO
from utils import fill_na

from ipdb import set_trace as st

__all__ = [
    'OneToOneConvert', 'NToOneConvert', 'OneHotConvert', 'RowFilterConvert', 'ColumnFilterConvert'
]


class BaseConvert(object):
    def __init__(self, method_name, columns, last_file_flow):
        self.method = self.__getattribute__(method_name)
        self.columns = columns
        self.last_file_flow = last_file_flow


class OneToOneConvert(BaseConvert):
    def __init__(self, method_name, columns, last_file_flow):
        super(OneToOneConvert, self).__init__(method_name, columns, last_file_flow)

    def execute(self, df):
        # 强行要求第一个参数是要处理的列名 后面跟的是参数
        for column in self.columns:
            if isinstance(column, tuple) and len(column) > 1:
                df[column[0]] = df[column[0]].apply(self.method, args=tuple(column[1:]))
            else:
                df[column] = df[column].apply(self.method)
        return df

    def plus_one(self, input):
        return float(input) + 1

    def H_M_L_convert(self, input, _min, _max):
        """
        根据检验项目的正常值区间(如'白细胞') 并将该检验项目划分为[高 正常 低]离散值 并编码为[H M L]
        :param input: float, 检验值
        :param _min: float, 检验值正常范围的下界
        :param _max: float, 检验值正常范围的上界
        :return: str, H or M or L
        """
        input, _min, _max = float(input), float(_min), float(_max)
        if input < _min:
            ret = 'L'
        elif input >= _min and input <= _max:
            ret = 'M'
        else:
            ret = 'H'
        return ret

    def binning_numeric(self, input, _min, _max, gap):
        input, _min, _max, gap = float(input), int(_min), int(_max), int(gap)
        if input <= _min:
            return '<' + str(_min)
        elif input >= _max:
            return '>' + str(_max)
        else:
            for i in range(_min + gap, _max, gap):
                if input <= i:
                    return ''.join([str(i - gap), '_', str(i)])
            return ''.join([str(_max - gap), '_', str(_max)])

    def binary_convert(self, input, _one):
        """
        处理'有'或'无'类的变量(如'有无腹主动脉淋巴结转移') 并将该列的值[0,1]二值化
        :param input: str, 实际值
        :param _one: str, 标记'有'或'无'的类型
        :return:
        """
        return 1.0 if input == _one else 0.0

    def normalization_convert(self, input, _min, _max):
        """
        数值型特征归一化处理 放缩到[0,1] 大于上限值或小于下限值的情况 单独处理
        :param input: float, 实际变量值
        :param _min: float, 特征的下限值
        :param _max: float, 特征的上限值
        :return: str, 归一化后的值
        """
        if input >= _max:
            return 1.0
        elif input <= _min:
            return 0.0
        else:
            return "{0:.4f}".format(1.0 * (input - _min) / (_max - _min))

    def negative_or_positive_convert(self, input, strategy):
        def _strategy_one(input):
            if input == 'nan':
                return 'nan'
            else:
                return POSITIVE if float(input) > 100 else NEGATIVE

        def _strategy_two(input):
            if input == 'nan':
                return 'nan'
            elif isinstance(input, float):
                return POSITIVE if float(input) > 4.0 else NEGATIVE
            elif "阴" in input:
                return NEGATIVE

        def _strategy_three(input):
            return '男' if int(input) == 1 else '女'

        _strategy = {
            'one': _strategy_one,
            'two': _strategy_two,
            'three': _strategy_three
        }
        return _strategy[strategy](input)


class NToOneConvert(BaseConvert):
    def __init__(self, method_name, columns, last_file_flow):
        super(NToOneConvert, self).__init__(method_name, columns, last_file_flow)

    def execute(self, df):
        # 这个地方 如果给对象增加一个df属性 会不会单开一部分内容 因为df是比较大的数
        self.df = df
        for column in self.columns:
            derived_col_name, args = column[-1], column[:-1]
            df[derived_col_name] = self.method(*args)
        return df

    def plus(self, col1, col2):
        return self.df[col1].astype('float') + self.df[col2].astype('float')

    def minus(self, col1, col2):
        return self.df[col1].astype('float') - self.df[col2].astype('float')

    def derive_age_from_date(self, col1, col2):
        return self.df[col1].apply(lambda x: x.split('-')[0]).astype('int16') - \
               self.df[col2].apply(lambda x: x.split('-')[0]).astype('int16')

    def merge_positive_negative(self, col1, col2):

        if self.df[col1] == POSITIVE or self.df[col2] == POSITIVE:
            return POSITIVE
        if self.df[col1] == NEGATIVE or self.df[col2] == NEGATIVE:
            return NEGATIVE

    def find_largest(self, col1, col2, col3):
        self.df[col1], self.df[col2], self.df[col3] = \
            self.df[col1].astype('float'), self.df[col2].astype('float'), self.df[col3].astype('float')
        return self.df.apply(lambda d: max(d[col1], d[col2], d[col3]), axis=1)


class OneHotConvert(BaseConvert):
    def __init__(self, method_name, columns, last_file_flow):
        self.column_labels = self.pre_process_label(last_file_flow, columns)

    def execute(self, df):
        return self.one_hot_convert(df)

    def pre_process_label(self, input_path, columns):
        column_labels = {column: defaultdict(int) for column in columns}
        for df in IO.read_from_csv(input_path):
            for column in column_labels.keys():
                df[column] = fill_na(df, column)
                labels, counts = df[column].value_counts().keys().values, df[column].value_counts().values
                for l, c in zip(labels, counts):
                    column_labels[column][l] += c
        max_cat = MAX_FEATURE_CATEGORY_NUM
        min_cat = MIN_FEATURE_CATEGORY_NUM
        return {k: v for k, v in column_labels.items() if
                len(v.keys()) > min_cat and len(v.keys()) < max_cat}

    def one_hot_convert(self, df):
        column_le = {}
        for column, labels in self.column_labels.items():
            le = LabelEncoder()
            le.fit(labels.keys())
            column_le[column] = le
            df[column] = le.transform(fill_na(df, column))
        columns_to_encode = column_le.keys()
        _n_values = [len(self.column_labels[column].keys()) for column in columns_to_encode]
        ohe = OneHotEncoder(n_values=_n_values)
        data_derived = ohe.fit_transform(df[columns_to_encode]).toarray().astype('float32')
        df.drop(columns_to_encode, axis=1, inplace=True)
        columns_derived = [FEATURE_NAME_CONNECT.join([column, str(_class)])
                           for column in columns_to_encode for _class in column_le[column].classes_]
        return df.join(pd.DataFrame(data=data_derived, columns=columns_derived, index=df.index))


class RowFilterConvert(BaseConvert):
    def __init__(self, method_name, columns, last_file_flow):
        super(RowFilterConvert, self).__init__(method_name, columns, last_file_flow)

    def execute(self, df):
        return self.method(df, self.columns)

    def filter_row_contain_na_value(self, df, columns):
        _df = df.loc[:, columns]
        _df.dropna(axis=0, inplace=True)
        return df.ix[_df.index, :]


class ColumnFilterConvert(BaseConvert):
    def __init__(self, method_name, columns, last_file_flow):
        super(ColumnFilterConvert, self).__init__(method_name, columns, last_file_flow)

    def execute(self, df):
        return self.method(df, self.columns)

    def remain_columns(self, df, columns):
        return df[columns]


class FeatureConvertFactory(object):
    def __init__(self, last_flow_file, convert_info):
        self.last_flow_file = last_flow_file
        self.convert_info = convert_info
        self.convert_list = self.get_convert_list()

    def get_convert_list(self):

        def _get_convert(class_method, columns, last_file_flow):
            return class_method.im_class(class_method.im_func.func_name, columns, last_file_flow)

        convert_list = []
        for info in self.convert_info:
            _convert, _args = info[0], info[1]
            convert_list.append(_get_convert(_convert, _args, self.last_flow_file))
        return convert_list

    def execute_all(self, df):
        for convert in self.convert_list:
            df = convert.execute(df)
        return df
