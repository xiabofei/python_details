# encoding=utf8


import gensim
import logging

from sentence_iterator import SentencesIterator

from itertools import chain
from collections import Iterable

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# user define dict path
usr_dict_dir = '../data/input/domain_dict/'
usr_dict_files = (
    '0_central',
    '1_auxiliary',
    '2_manual',
    # '3_zhenduan_addressed',
    # '4_shoushu_addressed',
    '5_jianyan_addressed',
    '6_yaopin_addressed',
)
usr_dict_path_list = [usr_dict_dir + f for f in usr_dict_files]
# user suggest dict path
usr_suggest_path_list = []
# target keshi path name
target_keshi_path = '../data/input/target_keshi.dat'


def hnrm_corpus():
    # cut_columns = ('zhenduan', 'zhusu', 'xianbingshi', 'xianbingshi')
    cut_columns = ('zhenduan',)
    wanted_columns = ('zhenduan',)
    sentences = SentencesIterator(
        '../data/output/merged/all_merged.csv',
        usr_dict_path_list,
        usr_suggest_path_list,
        target_keshi_path,
        cut_columns,
        wanted_columns,
        '\t',
        True
    )
    return sentences


def xiehe_corpus():
    cut_columns = ('icd-desc',)
    wanted_columns = ('icd-desc',)
    sentences = SentencesIterator(
        '../data/output/merged/dis.csv',
        usr_dict_path_list,
        usr_suggest_path_list,
        False,
        cut_columns,
        wanted_columns,
        '\t',
        True
    )
    return sentences


def icd10_corpus():
    cut_columns = ('icd-desc',)
    wanted_columns = ('icd-desc',)
    sentences = SentencesIterator(
        '../data/output/merged/icd10.csv',
        usr_dict_path_list,
        usr_suggest_path_list,
        False,
        cut_columns,
        wanted_columns,
        '\t',
        True
    )
    return sentences


class MultiCorpus(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __setattr__(self, key, value):
        if key == 'corpus':
            for c in value:
                if not isinstance(c, Iterable):
                    raise TypeError('corpus %s is not iterable' % type(c))
        object.__setattr__(self, key, value)

    def __iter__(self):
        for sentences in self.corpus:
            for sentence in sentences:
                yield sentence


corpus_list = MultiCorpus([hnrm_corpus(), xiehe_corpus(), icd10_corpus()])

model = gensim.models.Word2Vec(corpus_list, min_count=30, size=64, workers=1, window=2)
