# encoding=utf8

import pandas as pd
import jieba
import re

from utils import JiebaTuning as JT
from utils import PreClean as Clean
from utils import numerical_type
from utils import combiner as func_combiner
from utils import CLRF
from utils import re_INTEGER
from utils import SEG_SPLIT

from ipdb import set_trace as st

replace_punctuation = Clean.replace_punctuation
replace_negative_positive = Clean.replace_negative_positive
rm_digits_dis = Clean.rm_dis_digits


class Seg4Diagnosis(object):
    def __init__(
            self,
            input_file_path,
            usr_dict_path,
            usr_suggest_path,
            stop_words_path,
            column_name,
            sep=',',
            HMM=False,
            SUFFIX=False
    ):
        self.input_file_path = input_file_path
        self.__tuning_jieba(usr_dict_path, usr_suggest_path)
        self.column_name = column_name
        self.stop_words = set(w for w in self.loading_stop_words(stop_words_path)) if stop_words_path else None
        self.sep = sep
        self.HMM = HMM
        self.seg_pipe = self.build_seg_pipe(SUFFIX)
        self.clean_pipe = self.build_clean_pipe()

    def build_seg_pipe(self, suffix):
        if suffix:
            return func_combiner(reversed([self.clean, self.word_seg, self.conn_time_word, self.conn_suffix_word]))
        else:
            return func_combiner(reversed([self.clean, self.word_seg, self.conn_time_word]))

    def build_clean_pipe(self):
        return func_combiner([replace_punctuation, replace_negative_positive, rm_digits_dis])

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

    def loading_stop_words(self, path):
        with open(path, 'rb') as f:
            for l in f.readlines():
                yield l.strip().decode('utf-8')

    def clean(self, content):
        if isinstance(content, numerical_type):
            return str(numerical_type).decode('utf8')
        return self.clean_pipe(content.decode('utf8'))

    def word_seg(self, sentence):
        if self.stop_words:
            return filter(self.word_seg_filter, [k for k in jieba.cut(sentence, HMM=self.HMM)])
        else:
            return [k for k in jieba.cut(sentence, HMM=self.HMM)]

    def word_seg_filter(self, term):
        if term in self.stop_words:
            return False
        # if re_INTEGER.match(term):
        #     return False
        return True

    suffix_words = (
        u'病', u'症', u'后', u'型', u'期', u'史', u'程', u'级',
        u'性', u'区', u'周', u'天', u'方案', u'分'
    )

    time_words = (
        u'年', u'余年', u'年余',
        u'月', u'余月', u'月余',
        u'周', u'余周', u'周余',
        u'天', u'余天', u'天余',
        u'日', u'余日', u'日余',
        u'小时', u'分钟', u'分', u'次'
    )

    def conn_suffix_word(self, sentence):
        ret = []
        for i, c in enumerate(sentence):
            if c in self.suffix_words and i > 0:
                ret[-1] += c
            else:
                ret.append(c)
        return ret

    def conn_time_word(self, sentence):
        ret = []
        for i, c in enumerate(sentence):
            if c in self.time_words and i > 0 and re_INTEGER.match(ret[-1]):
                ret[-1] += c
            else:
                ret.append(c)
        return ret

    def seg(self, sentence):
        return self.seg_pipe(sentence)

    def __iter__(self):
        # when create seg_pipe keep mind on the funcs order
        for df in pd.read_csv(self.input_file_path, chunksize=50000, sep=self.sep):
            series_target = df[self.column_name].dropna(axis=0)
            # patient_id = df.loc[series_target.index]['patient_id']
            del df
            series_target = series_target.reset_index(drop=True)
            # patient_id = patient_id.reset_index(drop=True)
            series_target = series_target.astype('object').apply(self.seg_pipe)
            for i in series_target.index:
                # series_target.loc[i].append(patient_id.loc[i])
                yield series_target.loc[i]


def seg_template(sentence_iterator, output_path):
    with open(output_path, 'w') as f:
        dat = []
        for sentence in sentence_iterator:
            dat.append(SEG_SPLIT.join(sentence))
        f.write(CLRF.join(map(lambda x: x.encode('utf8'), dat)))


def seg_dis():
    dis_sentence_iterator = Seg4Diagnosis(
        '../data/output/extract_from_excel/dis.csv',
        [
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        '../data/output/domain_dict/suggest_freq_dict.csv',
        '../data/output/domain_dict/stop_words.csv',
        'dis-desc',
        '\t',
        True,
        # True
    )
    seg_template(dis_sentence_iterator, '../data/output/cut_result/dis_cut.csv')
    print('seg dis complete... ')


def seg_icd10():
    icd10_sentence_iterator = Seg4Diagnosis(
        '../data/output/extract_from_excel/icd_national.csv',
        [
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        '../data/output/domain_dict/suggest_freq_dict.csv',
        '../data/output/domain_dict/stop_words.csv',
        'icd-desc',
        '\t',
        True,
        # True
    )
    seg_template(icd10_sentence_iterator, '../data/output/cut_result/icd_national_cut.csv')
    print('seg icd10 complete... ')


def seg_zhenduan():
    zhenduan_sentence_iterator = Seg4Diagnosis(
        '../data/output/cut_result/all_merged.csv',
        [
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        '../data/output/domain_dict/suggest_freq_dict.csv',
        '../data/output/domain_dict/stop_words.csv',
        'zhenduan',
        '\t',
        True
    )
    seg_template(zhenduan_sentence_iterator, '../data/output/cut_result/zhenduan_cut.csv')
    print('seg zhenduan complete... ')


def seg_zhusu():
    zhusu_sentence_iterator = Seg4Diagnosis(
        '../data/output/cut_result/all_merged.csv',
        [
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        '../data/output/domain_dict/suggest_freq_dict.csv',
        '../data/output/domain_dict/stop_words.csv',
        'zhusu',
        '\t',
        True
    )
    seg_template(zhusu_sentence_iterator, '../data/output/cut_result/zhusu_cut.csv')
    print('seg zhusu complete... ')


def seg_xianbingshi():
    xianbingshi_sentence_iterator = Seg4Diagnosis(
        '../data/output/cut_result/all_merged.csv',
        [
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        '../data/output/domain_dict/suggest_freq_dict.csv',
        '../data/output/domain_dict/stop_words.csv',
        'xianbingshi',
        '\t',
        True
    )
    seg_template(xianbingshi_sentence_iterator, '../data/output/cut_result/xianbingshi_cut.csv')
    print('seg xianbingshi complete... ')


def seg_bingchengjilu():
    bingchengjilu_sentence_iterator = Seg4Diagnosis(
        '../data/output/cut_result/all_merged.csv',
        [
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        '../data/output/domain_dict/suggest_freq_dict.csv',
        '../data/output/domain_dict/stop_words.csv',
        'bingchengjilu',
        '\t',
        True
    )
    seg_template(bingchengjilu_sentence_iterator, '../data/output/cut_result/bingchengjilu_cut.csv')
    print('seg bingchengjilu complete... ')


if __name__ == '__main__':
    # icd
    seg_icd10()
    # xiehe
    # seg_dis()
    # henan
    # seg_zhenduan()
    # seg_zhusu()
    # seg_xianbingshi()
    # seg_bingchengjilu()
