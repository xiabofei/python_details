# encoding=utf8

import sys
import pandas as pd
import jieba.posseg as pos
import jieba

sys.path.append('../../')
from utils.prepare_utils import JiebaTuning as JT

CLRF = '\n'

from ipdb import set_trace as st


class SentencesIterator(object):
    def __init__(
            self,
            input_file_path,
            usr_dict_path,
            usr_suggest_path,
            target_keshi_path,
            cut_columns,
            wanted_column,
            sep=',',
            HMM=False
    ):
        self.input_file_path = input_file_path
        self.__tuning_jieba(usr_dict_path, usr_suggest_path)
        self.target_keshi = self.__get_target_keshi(target_keshi_path)
        self.cut_columns = cut_columns
        self.wanted_column = wanted_column
        self.sep = sep
        self.HMM = HMM

    def __get_target_keshi(self, path):
        ret = []
        with open(path, 'r') as f:
            for l in f.readlines():
                ret.append(l.strip())
        return set(ret)

    @classmethod
    def __tuning_jieba(cls, usr_dict_path, usr_suggest_path):
        # user domain dict path
        if usr_dict_path:
            if isinstance(usr_dict_path, str):
                JT.add_usr_dict(usr_dict_path, sep='\t')
            elif isinstance(usr_dict_path, list):
                for udp in usr_dict_path:
                    JT.add_usr_dict(udp, sep='\t')
            else:
                raise TypeError('usr_dict_path %s wrong type' % usr_dict_path)
        # user domain suggest path
        if usr_suggest_path:
            if isinstance(usr_suggest_path, str):
                JT.suggest_usr_dict(usr_suggest_path, sep='\t')
            elif isinstance(usr_suggest_path, list):
                for usp in usr_suggest_path:
                    JT.suggest_usr_dict(usp, sep='\t')
            else:
                raise TypeError('usr_suggest_path %s wrong type' % usr_suggest_path)

    def __conduct_jieba_cut(self, content):
        if isinstance(content, float):
            return [u'']
        return [w for w, l in pos.cut(content, HMM=self.HMM) if l != 'x']
        # return [w for w in jieba.cut(content, HMM=self.HMM)]

    def __iter__(self):
        for df in pd.read_csv(self.input_file_path, chunksize=30000, sep=self.sep):
            # remain target keshi
            df = df[df['keshi'].isin(self.target_keshi)].reset_index(drop=True)
            # cut fields
            for col in self.cut_columns:
                df[col + '_cut'] = df[col].astype('object').apply(self.__conduct_jieba_cut)
            # yield sentences
            try:
                for i in df.index:
                    yield df.iloc[i][self.wanted_column + '_cut']
            except Exception,e:
                st(context=21)
                print 'ee'


if __name__ == '__main__':
    # user define dict path
    usr_dict_dir = '../data/input/domain_dict/'
    usr_dict_files = (
        '0_central', '1_auxiliary', '2_manual',
        '3_zhenduan_addressed', '4_shoushu_addressed', '5_jianyan_addressed', '6_yaopin_addressed',
    )
    usr_dict_path_list = [usr_dict_dir + f for f in usr_dict_files]
    # user suggest dict path
    usr_suggest_path_list = []
    # target keshi path name
    target_keshi_path = '../data/input/target_keshi.dat'
    # cut columns
    cut_columns = ('zhenduan', 'zhusu', )
    # wanted column
    wanted_column = 'zhenduan'

    sentence_it = SentencesIterator(
        '../data/output/merged/all_merged.csv',
        usr_dict_path_list,
        usr_suggest_path_list,
        target_keshi_path,
        cut_columns,
        wanted_column,
        '\t',
        True
    )

    for it in sentence_it:
        print 'okay'
