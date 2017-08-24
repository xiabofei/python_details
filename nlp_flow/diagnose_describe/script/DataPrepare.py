# encoding=utf8

import sys
import pandas as pd
import jieba.posseg as posseg
import cPickle

sys.path.append('../../')
from utils.prepare_utils import JiebaTuning as JT
from utils.prepare_utils import GeneralAddressing as GA

CLRF = '\n'


class SentencesIterator(object):
    def __init__(
            self,
            input_file_path,
            usr_dict_path,
            usr_suggest_path,
            columns,
            available_words_path,
            sep=',',
            HMM=False
    ):
        self.input_file_path = input_file_path
        self.__tuning_jieba(usr_dict_path, usr_suggest_path)
        self.columns = columns
        self.available_words = self.__loading_available_words(available_words_path)
        self.sep = sep
        self.HMM = HMM

    def __tuning_jieba(self, usr_dict_path, usr_suggest_path):
        if usr_dict_path:
            JT.add_usr_dict(usr_dict_path)
        if usr_suggest_path:
            JT.suggest_usr_dict(usr_suggest_path)

    def if_available(self, term):
        return term in self.available_words

    def __loading_available_words(self, path):
        if path:
            available_words = []
            with open(path, 'rb') as f:
                word_frequency = cPickle.load(f)
                assert type(word_frequency) == dict
                for k, _ in sorted(word_frequency.items(), key=lambda x: x[1]):
                    available_words.append(k)
                return set(available_words)
        return None

    def __iter__(self):
        for df in pd.read_csv(self.input_file_path, chunksize=50000, sep=self.sep):
            for i in df.index:
                sentence = GA.execute_general_addressing(
                    df.loc[i, self.columns[0]],
                    [GA.replace_punctuation, GA.negative_positive, GA.clear_trivial_head]
                )
                if self.available_words:
                    yield filter(self.if_available, [k for k, _ in posseg.cut(sentence, HMM=self.HMM)])
                else:
                    yield [k for k, _ in posseg.cut(sentence, HMM=False)]


def main():
    dis_sentence_iterator = SentencesIterator(
        '../data/output/dis.csv', None, None, ['icd-desc', 'icd-code'], None, '\t'
    )
    icd10_sentence_iterator = SentencesIterator(
        '../data/output/icd10.csv', None, None, ['icd-desc', 'icd-code'], None, '\t'
    )
    with open('../data/output/dis_cut.csv', 'w') as f_dis, open('../data/output/icd10_cut.csv', 'w') as f_icd10:
        for sentence in dis_sentence_iterator:
            f_dis.write('/'.join(sentence).encode('utf-8') + CLRF)
        for sentence in icd10_sentence_iterator:
            f_icd10.write('/'.join(sentence).encode('utf-8') + CLRF)


if __name__ == '__main__':
    main()
