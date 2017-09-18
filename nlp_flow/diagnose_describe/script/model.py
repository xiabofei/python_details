# encoding=utf8

import gensim
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MultiCorpus(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for f_name in os.listdir(self.dir_name):
            for line in open(os.path.join(self.dir_name, f_name)):
                yield line.decode('utf8').strip().split('/')


corpus_dir_name = '../data/output/cut_result/cut_result_for_w2v/'

model = gensim.models.Word2Vec(MultiCorpus(corpus_dir_name), min_count=10, size=64, workers=1, window=3 ,iter=5)
model.save('../data/output/model/mc10_s64_w3_cbow.model')

model = gensim.models.Word2Vec(MultiCorpus(corpus_dir_name), min_count=10, size=128, workers=1, window=3 ,iter=5)
model.save('../data/output/model/mc10_s128_w3_cbow.model')

