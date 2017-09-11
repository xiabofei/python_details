# encoding=utf8

import sys
import pandas as pd
import jieba.posseg as posseg
import jieba
import cPickle
from collections import defaultdict
import re

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
            stop_words_path,
            columns,
            sep=',',
            HMM=False
    ):
        self.input_file_path = input_file_path
        self.__tuning_jieba(usr_dict_path, usr_suggest_path)
        self.columns = columns
        self.stop_words = set(w for w in self.loading_stop_words(stop_words_path)) if stop_words_path else None
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

    def loading_stop_words(self, path):
        with open(path, 'rb') as f:
            for l in f.readlines():
                yield l.strip().decode('utf-8')

    @staticmethod
    def remove_dis_digits(dis):
        return re.sub("\d*[.,]", "", dis)

    def __iter__(self):
        for df in pd.read_csv(self.input_file_path, chunksize=50000, sep=self.sep):
            for i in df.index:
                sentence = GA.execute_general_addressing(
                    df.loc[i, self.columns[0]],
                    [
                        GA.replace_punctuation, GA.negative_positive, GA.clear_trivial_head,
                        self.remove_dis_digits
                    ]
                )
                if self.stop_words:
                    yield filter(lambda x: x not in self.stop_words, [k for k in jieba.cut(sentence, HMM=self.HMM)])
                else:
                    yield [k for k in jieba.cut(sentence, HMM=self.HMM)]


def main_dis_icd10():
    dis_sentence_iterator = SentencesIterator(
        '../data/output/extract_from_excel/dis.csv',
        [
            # '../data/output/domain_dict/annotated_dict.csv',
            # '../data/output/domain_dict/fixed_annotated_dict.csv',
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        None,
        '../data/output/domain_dict/stop_words.csv',
        ['icd-desc', 'icd-code'],
        '\t',
        True
    )
    icd10_sentence_iterator = SentencesIterator(
        '../data/output/extract_from_excel/icd10.csv',
        [
            # '../data/output/domain_dict/annotated_dict.csv',
            # '../data/output/domain_dict/fixed_annotated_dict.csv',
            '../data/output/domain_dict/snowball_dict.csv',
        ],
        None,
        '../data/output/domain_dict/stop_words.csv',
        ['icd-desc', 'icd-code'],
        '\t',
        True
    )

    # load central category term and sub category term
    word_category = {}
    # with open('../data/output/domain_dict/fixed_annotated_category.csv', 'r') as f_in:
    with open('../data/output/domain_dict/snowball_category.csv', 'r') as f_in:
        for l in f_in.readlines():
            try:
                w, c = l.split('\t')[0].strip().replace('其他', ''), l.split('\t')[1].strip()
                # w, c = l.split('\t')[0].strip(), l.split('\t')[1].strip()
                if w:
                    word_category[w.decode('utf8')] = c.decode('utf8')
            except Exception, e:
                st(context=21)
                print e

    # manually fix suffix words and trick new dict
    suffix_words = (u'病', u'症', u'后', u'型', u'期', u'史', u'程', u'级', u'性', u'区', u'周', u'天',u'方案')

    def connect_suffix_word(sentence, word_category, f_dict, f_category):
        ret = []
        hit_count = 0
        last_category = None
        for i, c in enumerate(sentence):
            curr_category = word_category.get(c, None)
            # 2. address suffix word issue
            if c in suffix_words and i > 0:
                ret[-1] = re.sub(r"\[.*?\]", "", ret[-1])
                ret[-1] += c
                # 2.1 term not in word_category
                if not word_category.get(ret[-1], None):
                    f_dict.append('\t'.join([ret[-1].encode('utf8'), '50']))
                    # 2.1.1 last term category not None then keep last_category unchanged
                    if last_category:
                        f_category.append('\t'.join([ret[-1].encode('utf8'), last_category.encode('utf8')]))
                        ret[-1] = ''.join([ret[-1], '[', last_category, ']'])
                # 2.2 term already in word_category
                else:
                    last_category = word_category.get(ret[-1])
                    ret[-1] = ''.join([ret[-1], '[', word_category.get(ret[-1]), ']'])
            # 3. not suffix word
            else:
                if curr_category:
                    hit_count += 1
                    ret.append(''.join([c, '[', curr_category, ']']))
                else:
                    ret.append(c)
                last_category = curr_category
        return ret, hit_count

    # load offline words dict
    # with open('../data/output/domain_dict/central_word_dict.csv', 'r') as f_central, \
    #         open('../data/output/domain_dict/subcategory_reference_word_category.csv', 'r') as f_subcategory, \
    #         open('../data/output/domain_dict/manual_word_category.csv', 'r') as f_manual_category:
    #     for l in f_central.readlines():
    #         l = GA.execute_general_addressing(l, [GA.replace_punctuation, GA.negative_positive])
    #         word_category[l.split('\t')[0].decode('utf-8')] = u'中心词'
    #     for l in f_subcategory.readlines():
    #         l = GA.execute_general_addressing(l, [GA.replace_punctuation, GA.negative_positive])
    #         word_category[l.split('\t')[0].decode('utf-8')] = l.split('\t')[1].rstrip().decode('utf-8')
    #     for l in f_manual_category.readlines():
    #         l = GA.execute_general_addressing(l, [GA.replace_punctuation, GA.negative_positive])
    #         word_category[l.split('\t')[0].decode('utf-8')] = l.split('\t')[1].rstrip().decode('utf-8')
    # store offline words dict
    # with open('../data/output/domain_dict/all_annotated_words.csv', 'w') as f:
    #     for word, category in sorted(word_category.items(), key=lambda x: x[1]):
    #         f.write('\t'.join([word.encode('utf8'), category.encode('utf8')]) + CLRF)
    # cut dis and icd
    with open('../data/output/cut_result/dis_cut.csv', 'w') as f_dis, \
            open('../data/output/cut_result/icd10_cut.csv', 'w') as f_icd10, \
            open('../data/output/domain_dict/trick_dict.csv', 'w') as f_trick_dict, \
            open('../data/output/domain_dict/trick_category.csv', 'w') as f_trick_category:
        trick_dict, trick_category = [], []
        all_count, hit_count = 0, 0
        for sentence in dis_sentence_iterator:
            sentence, hit = connect_suffix_word(sentence, word_category, trick_dict, trick_category)
            hit_count += hit
            all_count += len(sentence)
            f_dis.write('/'.join(sentence).encode('utf-8') + CLRF)
        print('[DIS] hit_count / all_count = %s / %s = %s'%(hit_count, all_count, hit_count*1.0/all_count))
        all_count, hit_count = 0, 0
        for sentence in icd10_sentence_iterator:
            sentence, hit = connect_suffix_word(sentence, word_category, trick_dict, trick_category)
            hit_count += hit
            all_count += len(sentence)
            f_icd10.write('/'.join(sentence).encode('utf-8') + CLRF)
        print('[ICD10] hit_count / all_count = %s / %s = %s'%(hit_count, all_count, hit_count*1.0/all_count))
        f_trick_dict.write(CLRF.join(set(trick_dict)))
        f_trick_category.write(CLRF.join(set(trick_category)))


if __name__ == '__main__':
    main_dis_icd10()
