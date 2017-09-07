# encoding=utf8

import sys
import pandas as pd
import jieba.posseg as posseg
import jieba
import cPickle
from collections import defaultdict

sys.path.append('../../')
from utils.prepare_utils import JiebaTuning as JT
from utils.prepare_utils import GeneralAddressing as GA

CLRF = '\n'

from ipdb import set_trace as st


class SentencesIterator(object):
    def __init__(
            self,
            input_file_path,
            usr_dict_path,
            usr_suggest_path,
            columns,
            stop_words_path,
            sep=',',
            HMM=False
    ):
        self.input_file_path = input_file_path
        self.__tuning_jieba(usr_dict_path, usr_suggest_path)
        self.columns = columns
        self.stop_words = set(w for w in self.__loading_available_words(stop_words_path))
        self.sep = sep
        self.HMM = HMM

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

    def __loading_available_words(self, path):
            with open(path, 'rb') as f:
                for l in f.readlines():
                    yield

    def __iter__(self):
        for df in pd.read_csv(self.input_file_path, chunksize=50000, sep=self.sep):
            for i in df.index:
                sentence = GA.execute_general_addressing(
                    df.loc[i, self.columns[0]],
                    [GA.replace_punctuation, GA.negative_positive, GA.clear_trivial_head]
                )
                if self.stop_words:
                    yield filter(lambda x:x not in self.stop_words, [k for k in jieba.cut(sentence, HMM=self.HMM)])
                else:
                    yield [k for k in jieba.cut(sentence, HMM=self.HMM)]


def main_dis_icd10():
    dis_sentence_iterator = SentencesIterator(
        '../data/output/extract_from_excel/dis.csv',
        [
            '../data/output/domain_dict/central_word_dict.csv',
            '../data/output/domain_dict/subcategory_reference_word_dict.csv',
            '../data/output/domain_dict/manual_word_dict.csv'
        ],
        '../data/output/domain_dict/stop_words.csv',
        ['icd-desc', 'icd-code'],
        None,
        '\t',
        True
    )
    icd10_sentence_iterator = SentencesIterator(
        '../data/output/extract_from_excel/icd10.csv',
        [
            '../data/output/domain_dict/central_word_dict.csv',
            '../data/output/domain_dict/subcategory_reference_word_dict.csv',
            '../data/output/domain_dict/manual_word_dict.csv'
        ],
        '../data/output/domain_dict/stop_words.csv',
        ['icd-desc', 'icd-code'],
        None,
        '\t',
        True
    )
    # manually fix suffix words
    suffix_words = (u'病', u'症', u'后', u'癌', u'型', u'期', u'史', u'程', u'级', u'性', )
    def _connect_suffix_word(sentence):
        ret = []
        for i,c in enumerate(sentence):
            if c in suffix_words and i>0:
                ret[-1] += sentence[i]
            else:
                ret.append(c)
        return ret
    # load central category term and sub category term
    word_category = {}
    def _add_category_info(sentence):
        for i,c in enumerate(sentence):
            if word_category.get(c, None):
                sentence[i] = sentence[i]+'['+word_category.get(c)+']'
        return sentence
    with open('../data/output/domain_dict/central_word_dict.csv','r') as f_central,\
            open('../data/output/domain_dict/subcategory_reference_word_category.csv','r') as f_subcategory,\
            open('../data/output/domain_dict/manual_word_category.csv','r') as f_manual_category :
        for l in f_central.readlines():
            word_category[l.split('\t')[0].decode('utf-8')] = u'中心词'
        for l in f_subcategory.readlines():
            word_category[l.split('\t')[0].decode('utf-8')] = l.split('\t')[1].rstrip().decode('utf-8')
        for l in f_manual_category.readlines():
            word_category[l.split('\t')[0].decode('utf-8')] = l.split('\t')[1].rstrip().decode('utf-8')

    with open('../data/output/cut_result/dis_cut.csv', 'w') as f_dis, \
            open('../data/output/cut_result/icd10_cut.csv', 'w') as f_icd10:
        for sentence in dis_sentence_iterator:
            sentence = _add_category_info(sentence)
            sentence = _connect_suffix_word(sentence)
            f_dis.write('/'.join(sentence).encode('utf-8') + CLRF)
        for sentence in icd10_sentence_iterator:
            sentence = _add_category_info(sentence)
            sentence = _connect_suffix_word(sentence)
            f_icd10.write('/'.join(sentence).encode('utf-8') + CLRF)


if __name__ == '__main__':
    main_dis_icd10()
