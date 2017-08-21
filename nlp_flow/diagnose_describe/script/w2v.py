# encoding=utf8

import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]

model = gensim.models.Word2Vec(sentences, min_count=1, size=5, workers=1)
