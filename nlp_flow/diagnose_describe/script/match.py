# encoding=utf8

from ipdb import set_trace as st

import gensim
from utils import SEG_SPLIT
from utils import CLRF
from collections import OrderedDict
from collections import defaultdict
from types import NoneType

import numpy as np
import json

NORM = np.linalg.norm
SIZE = 128


class MatchFromDIS2ICD(object):
    def __init__(self, icd_f_path, wv_model_path, zhenduan_f_path, result_path):
        self.icd_f_path = icd_f_path
        self.wv_model = gensim.models.Word2Vec.load(wv_model_path)
        self.wv_model_vocab = self.load_vocab()
        self.icd_norm_wv_2norm = {
            '0_core_term': [],
            '1_type_term': [],
            '2_stage_term': [],
            '3_degree_term': [],
            '4_region_term': [],
            'dummy_term': []
        }
        self.icd_norm_wv = {
            '0_core_term': [],
            '1_type_term': [],
            '2_stage_term': [],
            '3_degree_term': [],
            '4_region_term': [],
            'dummy_term': []
        }
        self.norm_weight = {
            '0_core_term': 0.6,
            '1_type_term': 0.0,
            '2_stage_term': 0.0,
            '3_degree_term': 0.0,
            '4_region_term': 0.3,
            'dummy_term': 0.0
        }
        self.icd_desc = []

    def load_vocab(self):
        return set(self.wv_model.wv.vocab.keys()) if self.wv_model else None

    def find_wv(self, item):
        if item in self.wv_model_vocab:
            return self.wv_model[item]
        return None

    def fetch_wv_array(self, items):
        if items is None:
            return []
        return filter(lambda x: x is not None, map(self.find_wv, items))

    def get_sentence_norm_wv(self, l):
        cut_sentence, norm_term = l.strip().split('\t')[0], json.loads(l.strip().split('\t')[1])
        norm_wv = {}
        norm_wv_2norm = {}
        for norm in self.icd_norm_wv_2norm.keys():
            if not norm_term.has_key(norm):
                norm_wv[norm] = np.zeros(SIZE).astype('float32')
                norm_wv_2norm[norm] = NORM(norm_wv[norm])
            else:
                ret = self.fetch_wv_array(norm_term.get(norm))
                if len(ret) == 0:
                    norm_wv[norm] = np.zeros(SIZE).astype('float32')
                    norm_wv_2norm[norm] = NORM(norm_wv[norm])
                else:
                    norm_wv[norm] = np.mean(ret, axis=0)
                    norm_wv_2norm[norm] = NORM(norm_wv[norm])
        return cut_sentence, norm_wv, norm_wv_2norm

    def convert_sentence_to_norm_vector(self):

        with open(self.icd_f_path, 'r') as f:
            for l in f.readlines():
                cut_sentence, norm_wv, norm_wv_2norm = self.get_sentence_norm_wv(l)
                self.icd_desc.append(cut_sentence)
                for norm in self.icd_norm_wv_2norm.keys():
                    self.icd_norm_wv_2norm[norm].append(norm_wv_2norm[norm])
                for norm in self.icd_norm_wv.keys():
                    self.icd_norm_wv[norm].append(norm_wv[norm])

        for k, v in self.icd_norm_wv_2norm.items():
            self.icd_norm_wv_2norm[k] = np.array(v)
        for k, v in self.icd_norm_wv.items():
            self.icd_norm_wv[k] = np.array(v)

    def calculate_similar_by_norm(self, l):
        cut_sentence, norm_wv, norm_wv_2norm = self.get_sentence_norm_wv(l)
        norm_scores = {}
        for norm in self.icd_norm_wv_2norm.keys():
            norm_scores[norm] = self.calculate_one_norm_score(norm, norm_wv[norm], norm_wv_2norm[norm])
        return norm_scores

    def find_most_similar_by_norm(self, norm_scores):
        ret = np.sum(map(lambda norm: norm_scores[norm] * self.norm_weight[norm], norm_scores.keys()), axis=0)
        return np.argmax(ret), np.max(ret)

    def calculate_one_norm_score(self, norm, wv, wv_2norm):
        denominator = np.dot(self.icd_norm_wv_2norm[norm], wv_2norm)
        numerator = np.dot(self.icd_norm_wv[norm], wv)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    def match_zhenduan_to_icd_by_norm(self, f_zhenduan, f_result):
        with open(f_zhenduan, 'r') as f, open(f_result, 'w') as f_out:
            for l in f.readlines():
                norm_scores = self.calculate_similar_by_norm(l)
                index, similarity = self.find_most_similar_by_norm(norm_scores)
                f_out.write('\t'.join(
                    [l.strip().split('\t')[0], self.icd_desc[index], l.strip().split('\t')[1], str(similarity)]) + CLRF)
                f_out.write('\t'.join([str(k) + ':' + str(v[index]) for k, v in norm_scores.items()]) + CLRF)


if __name__ == '__main__':
    # icd_f_path = '../data/output/cut_result/cut_result_for_w2v/icd10_cut.csv'
    icd_f_path = '../data/output/cut_result/cut_result_for_w2v/icd_national_cut_norm_mc10_s128_w3_cbow.csv'
    wv_model_path = '../data/output/model/mc10_s128_w3_cbow.model'
    # zhenduan_f_path = '../data/output/cut_result/cut_result_for_w2v/dis_cut_norm_mincount5_size64_window2_dict_updated.csv'
    zhenduan_f_path = '../data/output/cut_result/cut_result_for_w2v/uniq_dis_cut_norm_mc10_s128_w3_cbow.csv'
    result_path = '../data/output/cut_result/cut_result_for_w2v/match_result.csv'
    ins = MatchFromDIS2ICD(icd_f_path, wv_model_path, zhenduan_f_path, result_path)
    # ins.match_zhenduan_to_icd()
    ins.convert_sentence_to_norm_vector()
    ins.match_zhenduan_to_icd_by_norm(zhenduan_f_path, result_path)
