# encoding=utf8

import pandas as pd

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
        return 1.0 if input==_one else 0.0
