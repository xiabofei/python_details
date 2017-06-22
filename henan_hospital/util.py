# encoding=utf8

import pandas as pd
from config import ACTION_TYPE
from config import PLUS, MINUS, MULTIPLY, DERIVE_AGE_FROM_DATE
from config import POSITIVE, NEGATIVE
from config import MERGE_POS_NEG
from ipdb import set_trace as st


class OneToOneConvert(object):
    @classmethod
    def H_M_L_convert(cls, input, _min, _max):
        """
        根据检验项目的正常值区间(如'白细胞') 并将该检验项目划分为[高 正常 低]离散值 并编码为[H M L]
        :param input: float, 检验值
        :param _min: float, 检验值正常范围的下界
        :param _max: float, 检验值正常范围的上界
        :return: str, H or M or L
        """
        if input < _min:
            ret = 'L'
        elif input >= _min and input <= _max:
            ret = 'M'
        else:
            ret = 'H'
        return ret

    @classmethod
    def binary_convert(cls, input, _one):
        """
        处理'有'或'无'类的变量(如'有无腹主动脉淋巴结转移') 并将该列的值[0,1]二值化
        :param input: str, 实际值
        :param _one: str, 标记'有'或'无'的类型
        :return:
        """
        return 1.0 if input == _one else 0.0

    @classmethod
    def normalization_convert(cls, input, _min, _max):
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

    @classmethod
    def negative_or_positive_convert(cls, input, strategy):
        def _strategy_one(input):
            if input=='nan':
                return 'nan'
            else:
                return POSITIVE if float(input)>100 else NEGATIVE
        def _strategy_two(input):
            if input=='nan':
                return 'nan'
            elif "阴" in input:
                return NEGATIVE
            else:
                return POSITIVE if float(input)>4.0 else NEGATIVE
        def _strategy_three(input):
            return '男' if int(input)==1 else '女'
        _strategy = {
            'one':_strategy_one,
            'two':_strategy_two,
            'three':_strategy_three
        }
        return _strategy[strategy](input)


class NToOneConvert(object):
    @classmethod
    def two_to_one_convert(cls, df, col1, col2, action_type):
        # st(context=21)
        if action_type not in ACTION_TYPE:
            raise ValueError("action type not valid")
        if action_type == PLUS:
            return float(df[col1]) + float(df[col2])
        if action_type == MINUS:
            return float(df[col1]) - float(df[col2])
        if action_type == MULTIPLY:
            return float(df[col1]) * float(df[col2])
        if action_type == DERIVE_AGE_FROM_DATE:
            return int(df[col1].split("-")[0]) - int(df[col2].split("-")[0])
        if action_type == MERGE_POS_NEG:
           if df[col1]==POSITIVE or df[col2]==POSITIVE:
               return POSITIVE
           if df[col1]==NEGATIVE or df[col2]==NEGATIVE:
               return NEGATIVE

