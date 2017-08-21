# encoding=utf8

import cPickle
import pandas as pd
from collections import Counter
import re


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
        dis.to_csv('../data/output/dis.csv', encoding='utf-8', sep="\t", header=False, index=False)
        icd10.to_csv('../data/output/icd10.csv', encoding='utf-8', sep="\t", header=False, index=False)
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
            r = re.compile(r'.*'+ re.escape(d) +r'.*')
            return r.match(combined_icd10)
        hitting_dis = filter(_contained, set(dis))
        return list(set(dis).difference(hitting_dis))

    def __count_hitting(self, left_dis, icd10_dict, dis_count):
        print('\thitting distinct desc : %s' % (len(dis_count.keys()) - len(left_dis)))
        hitting_count = 0
        for dis in set(dis_count.keys()).difference(left_dis):
            hitting_count += dis_count[dis]
        print('\thitting all desc : %s' % hitting_count)


if __name__ == '__main__':
    input_excel_path = '../data/input/all-desc.xlsx'

    dis_sheet_name = 'dis3'
    icd10_sheet_name = 'ICD10-V6.01'

    dis_pkl_path = '../data/output/dis.pkl'
    icd10_pkl_path = '../data/output/icd10.pkl'

    m_icd = MatchingICD(
        input_excel_path,
        dis_sheet_name,
        icd10_sheet_name,
        dis_pkl_path,
        icd10_pkl_path
    )
    # m_icd.parse_content_from_excel()
    m_icd.dress_and_count_matched_dis()
