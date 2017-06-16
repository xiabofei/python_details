# encoding=utf8
import pandas as pd
from config import CHUNK_SIZE_THRESHOLD
from config import FEATURE_PROCESSOR, ONE_HOT_ENCODER
from util import featureProcessor
from collections import defaultdict

from pprint import pprint

from ipdb import set_trace as st


class featureConvert(object):

    __doc__ = "做特征转换的类:输入是csv的文本, 每一列是一个原始的feature;输出也是csv的文本, 每一列是处理后的feature"

    def __init__(self, file_path, feature_processor, sep=',', quote_char='"', header=None, featureProcessor=featureProcessor):
        self.file_path = file_path
        self.feature_processor = feature_processor
        self.sep = sep
        self.quote_char = quote_char
        self.header = header
        self.featureProcessor = featureProcessor

    def __setattr__(self, key, value):
        if key == 'feature_processor':
            if isinstance(value, dict):
                object.__setattr__(self, key, value)
            else:
                raise TypeError("feature_processor is not dict type")
        object.__setattr__(self, key, value)

    def load_from_csv(self, chunk_size=CHUNK_SIZE_THRESHOLD):
        return pd.read_csv(
            self.file_path,
            quotechar=self.quote_char,
            sep=self.sep,
            header=self.header,
            chunksize=chunk_size
        )

    def process_each_column(self, df):
        for column in df.columns:
            if column in self.feature_processor.keys():
                df[column] = df[column].map(self.feature_processor[column][FEATURE_PROCESSOR])
            else:
                df.drop(column, axis=1, inplace=True)

    def encoder_category_feature(self, df):
        column_names_to_encoder = []
        for column in df.columns:
            if column in self.feature_processor.keys() and feature_processor[column].get(ONE_HOT_ENCODER, False):
                column_names_to_encoder.append(column)
        return self.featureProcessor.one_hot_transformer(df, column_names_to_encoder)

    def pre_process_label(self):
        """
        处理每个category feature的编码问题
        如果文件较大没法一次用LabelEncoder处理完数据, 则需要每次地带更新category的全集, 再进行编码
        :param df:
        :return:
        """
        # st(context=21)
        column_distinctLabels = \
            { k:defaultdict(int) for k,v in self.feature_processor.items() if v.get(ONE_HOT_ENCODER, False) }
        for df in self.load_from_csv(chunk_size=500000):
            for column in column_distinctLabels.keys():
                labels, counts = df[column].value_counts().keys().values, df[column].value_counts().values
                for l,c in zip(labels,counts):
                    column_distinctLabels[column][l] += c
        print 'hh'


if __name__ == '__main__':
    file_path = './tijian.csv'
    feature_processor = {
        'PAPAT_DE_Name':{
            FEATURE_PROCESSOR:featureProcessor.default_processor,
        },
        'PAPAT_DE_Dob':{
            FEATURE_PROCESSOR:featureProcessor.default_processor,
            ONE_HOT_ENCODER:True
        },
        'PAPAT_DE_SexCode':{
            FEATURE_PROCESSOR:featureProcessor.default_processor,
            ONE_HOT_ENCODER:True
        },
        'CTPEI_Desc':{
            FEATURE_PROCESSOR:featureProcessor.default_processor
        },
        'PECRD_ItemUp':{
            FEATURE_PROCESSOR:featureProcessor.default_processor
        },
        'PECRD_Desc':{
            ONE_HOT_ENCODER:True
        }
    }
    # step1. create a feature convert instance and initialize it with a 'feature_processor'
    fc = featureConvert(file_path, feature_processor, header=0)
    # step2. pre process label for LabelEncoder of category features, need to go through the whole data once
    fc.pre_process_label()
    # step3. make further feature processing
    # for df in fc.load_from_csv(chunk_size=100000):
    #     pprint(df)
