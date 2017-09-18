# encoding=utf8

import jieba
import re
import types
import functools

numerical_type = (types.IntType, types.FloatType, types.LongType, types.ComplexType)
CLRF = '\n'
SEG_SPLIT = '/'
re_INTEGER = re.compile('^[-+]?[0-9]+$')


class JiebaTuning(object):
    @staticmethod
    def add_usr_dict(path, sep=','):
        with open(path, 'r') as f:
            for l in f.xreadlines():
                items = l.split(sep)
                if len(items) == 3:
                    jieba.add_word(items[0].rstrip(), int(items[1].rstrip()), items[2].rstrip())
                elif len(items) == 2:
                    jieba.add_word(items[0].rstrip(), int(items[1].rstrip()))
                elif len(items) == 1:
                    jieba.add_word(items[0].rstrip())
                else:
                    raise ValueError('too less number of word info \'%s\'' % (l.strip()))

    @staticmethod
    def suggest_usr_dict(path, sep=','):
        with open(path, 'r') as f:
            for l in f.xreadlines():
                word1, word2 = l.split(sep)[0].rstrip(), l.split(sep)[1].rstrip()
                jieba.suggest_freq((word1, word2), True)

def create_multi_replace_re(replace_dict):
    return re.compile('|'.join(map(re.escape, replace_dict)))


def combiner(flow):
        return functools.reduce(lambda f1, f2: lambda x: f1(f2(x)), flow, lambda x:x)

class PreClean(object):

    ch_eng = {
        u'，': u',', u'、': u',', u'（': u'(',
        u'）': u')', u'。': u'.', u'；': u';',
        u'：': u':', u'“': u'"', u"”": u'"',
        u'－': u'-', u' ': u'', u'Ca': u'癌',
    }
    re_ch_eng = create_multi_replace_re(ch_eng)

    @classmethod
    def replace_punctuation(cls, content):
        return cls.re_ch_eng.sub(lambda m: cls.ch_eng[m.group(0)], content)

    sym_neg_pos = {u'(-)': u'阴性', u'(+)': u'阳性'}
    re_sym_neg_pos = create_multi_replace_re(sym_neg_pos)

    @classmethod
    def replace_negative_positive(cls, content):
        return cls.re_sym_neg_pos.sub(lambda m: cls.sym_neg_pos[m.group(0)], content)

    @classmethod
    def rm_dis_digits(cls, dis):
        return re.sub("\d*[.,]", "", dis)



