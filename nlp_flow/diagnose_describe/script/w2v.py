# encoding=utf8

import gensim
import logging

from DataPrepare import SentencesIterator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = [['first', 'sentence'], ['second', 'sentence']]
sentences =  SentencesIterator(
    '../../data/input/all_mri.csv',
    '../../data/input/mri_dict.dat',
    '../../data/input/mri_suggest.dat',
    ['ExamDescExResult', 'ResultDescExResult'],
    '../../data/output/word_frequency.pkl',
)

model = gensim.models.Word2Vec(sentences, min_count=5, size=100, workers=1)

model.most_similar(u'子宫')

model.similarity(u'子宫', u'宫体')
