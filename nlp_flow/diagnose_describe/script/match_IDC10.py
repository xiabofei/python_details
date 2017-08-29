# encoding=utf8

import cPickle
import json
import pandas as pd
from collections import Counter, defaultdict
import re

import sys

sys.path.append('../../')
from utils.prepare_utils import GeneralAddressing as GA

CLRF = '\n'


class MatchingICD(object):
    def __init__(self,
                 input_excel_path,
                 dis_sheet_name,
                 icd10_sheet_name,
                 dis_pkl_path,
                 icd10_pkl_path
                 ):
        self.input_excel_path = input_excel_path
        self.dis_sheet_name = dis_sheet_name
        self.icd10_sheet_name = icd10_sheet_name
        self.dis_pkl_path = dis_pkl_path
        self.icd10_pkl_path = icd10_pkl_path
        self.icd10_dict = {}
        self.dis_content = []

    @staticmethod
    def replace_punctuation(content):

        if isinstance(content, float):
            return str(content)

        _chinese_english = [
            (u'，', u','),
            (u'、', u','),
            (u'（', u'('),
            (u'）', u')'),
            (u'。', u'.'),
            (u'；', u';'),
            (u'：', u':'),
            (u'“', u'"'),
            (u'－', u'-'),
            (u' ', u''),
        ]
        for i in _chinese_english:
            content = content.replace(i[0], i[1])

        _filter_punctuation = [u'"', u'\'']
        for i in _filter_punctuation:
            content = content.replace(i, u'')

        return content

    def parse_content_from_excel(self):
        content = pd.ExcelFile(self.input_excel_path)
        dis = content.parse(self.dis_sheet_name)[['dis', 'ICD-code']]
        dis['dis'] = dis['dis'].apply(self.replace_punctuation)
        icd10 = content.parse(self.icd10_sheet_name)[['ICD-desc', 'ICD-code']]
        icd10['ICD-desc'] = icd10['ICD-desc'].apply(self.replace_punctuation)
        # record data for display
        dis.to_csv(
            '../data/output/extract_from_excel/dis.csv',
            encoding='utf-8',
            sep="\t",
            header=['icd-desc', 'icd-code'],
            index=False
        )
        icd10.to_csv(
            '../data/output/extract_from_excel/icd10.csv',
            encoding='utf-8',
            sep="\t",
            header=['icd-desc', 'icd-code'],
            index=False
        )
        # record data in pickle format for reload
        self.dis_content = dis['dis'].tolist()
        self.icd10_dict = icd10.set_index('ICD-desc')['ICD-code'].to_dict()
        cPickle.dump(self.dis_content, open(self.dis_pkl_path, 'wb'), cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.icd10_dict, open(self.icd10_pkl_path, 'wb'), cPickle.HIGHEST_PROTOCOL)
        print 'okay'

    def dress_and_count_matched_dis(self):
        # load data from pkl
        dis = cPickle.load(open(self.dis_pkl_path, 'rb'))
        icd10_dict = cPickle.load(open(self.icd10_pkl_path, 'rb'))
        dis_count = Counter(dis)
        print('origin distinct desc : %s' % len(dis_count.keys()))
        print('origin all desc : %s' % sum(dis_count.values()))
        # rule 1 completely match
        left_dis = self.__match_completely(dis_count.keys(), icd10_dict.keys())
        print('[epoch 1]:')
        self.__count_hitting(left_dis, icd10_dict, dis_count)
        # rule 2 dis contained in ICD-desc
        left_dis = self.__match_partial_contained(left_dis, icd10_dict.keys())
        print('[epoch 2]')
        self.__count_hitting(left_dis, icd10_dict, dis_count)

    def __match_completely(self, dis, icd10):
        hitting_dis = set(dis).intersection(icd10)
        return list(set(dis).difference(hitting_dis))

    def __match_partial_contained(self, dis, icd10):
        combined_icd10 = '#'.join(icd10)

        def _contained(d):
            r = re.compile(r'.*' + re.escape(d) + r'.*')
            return r.match(combined_icd10)

        hitting_dis = filter(_contained, set(dis))
        return list(set(dis).difference(hitting_dis))

    def __count_hitting(self, left_dis, icd10_dict, dis_count):
        print('\thitting distinct desc : %s' % (len(dis_count.keys()) - len(left_dis)))
        hitting_count = 0
        for dis in set(dis_count.keys()).difference(left_dis):
            hitting_count += dis_count[dis]
        print('\thitting all desc : %s' % hitting_count)


class LoadCentralWords(object):
    def __init__(
            self,
            input_path,
            central_words_sheet_name,
            central_words_nickname_sheet_name,
            subcategory_reference_words_sheet_name,
    ):
        self.input_path = input_path
        self.central_words_sheet_name = central_words_sheet_name
        self.central_words_nickname_sheet_name = central_words_nickname_sheet_name
        self.subcategory_reference_words_sheet_name = subcategory_reference_words_sheet_name
        self.load_sheets()

    def load_sheets(self):
        excel = pd.ExcelFile(self.input_path)
        self.central_words = \
            excel.parse(self.central_words_sheet_name)[['TKBTRB_RowId', 'TKBTRB_Desc']]
        self.central_words_nickname = \
            excel.parse(self.central_words_nickname_sheet_name)[['TKBTRC_TRE_Dr', 'TKBTRC_Desc']]
        self.subcategory_reference_words = \
            excel.parse(self.subcategory_reference_words_sheet_name)[['TKBTD_Desc', 'ExtTypeDesc']]

    def merge_central_words_nickname(self):
        central_words = defaultdict(list)
        for i in self.central_words.index:
            id, word = self.central_words.iloc[i]['TKBTRB_RowId'], self.central_words.iloc[i]['TKBTRB_Desc']
            central_words[id].append(word.encode('utf-8'))
        for i in self.central_words_nickname.index:
            foreign_id = self.central_words_nickname.iloc[i]['TKBTRC_TRE_Dr']
            if central_words.get(foreign_id, None):
                central_words[foreign_id].append(self.central_words_nickname.iloc[i]['TKBTRC_Desc'].encode('utf8'))
        return central_words

    def subcategory_reference_words_iterator(self):
        for i in self.subcategory_reference_words.index:
            yield (
                self.subcategory_reference_words.iloc[i]['TKBTD_Desc'].encode('utf8'),
                self.subcategory_reference_words.iloc[i]['ExtTypeDesc'].encode('utf8')
            )


def central_word():
    input_path = '../data/input/central-word.xlsx'
    central_words_sheet_name = 'central_words'
    central_words_nickname_sheet_name = 'central_words_nickname'
    subcategory_reference_words_sheet_name = 'subcategory_reference_words'

    central_info = LoadCentralWords(
        input_path,
        central_words_sheet_name,
        central_words_nickname_sheet_name,
        subcategory_reference_words_sheet_name
    )

    central_words = central_info.merge_central_words_nickname()
    with open('../data/output/domain_dict/central_word_dict.csv', 'w') as f:
        for words in central_words.values():
            for word in words:
                f.write('\t'.join([word, '50']) + CLRF)

    with open('../data/output/domain_dict/subcategory_reference_word_dict.csv', 'w') as f_dict, \
            open('../data/output/domain_dict/subcategory_reference_word_category.csv', 'w') as f_category:
        for it in central_info.subcategory_reference_words_iterator():
            f_dict.write('\t'.join([GA.replace_punctuation(it[0]), '50']) + CLRF)
            f_category.write('\t'.join([GA.replace_punctuation(it[0]), GA.replace_punctuation(it[1])]) + CLRF)


def dis_icd10():
    input_excel_path = '../data/input/all-desc.xlsx'

    dis_sheet_name = 'dis3'
    icd10_sheet_name = 'ICD10-V6.01'

    dis_pkl_path = '../data/output/extract_from_excel/dis.pkl'
    icd10_pkl_path = '../data/output/extract_from_excel/icd10.pkl'

    m_icd = MatchingICD(
        input_excel_path,
        dis_sheet_name,
        icd10_sheet_name,
        dis_pkl_path,
        icd10_pkl_path
    )
    m_icd.parse_content_from_excel()
    # m_icd.dress_and_count_matched_dis()


if __name__ == '__main__':
    dis_icd10()
    # central_word()
