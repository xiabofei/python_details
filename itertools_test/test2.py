#encoding=utf8

import gensim
import logging

from itertools import chain

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
测试itertools.chain串起来的generator如何匹配到gensim中
"""

class sentence1(object):
    def __iter__(self):
        for i in [ [x] for x in "this is the second sentence test1".split(' ') ]:
            yield i

class sentence2(object):
    def __iter__(self):
        for i in [ [x] for x in "this is the second sentence test2".split(' ') ]:
            yield i

it_with_joined = [ s for s in chain(sentence1(), sentence2()) ]

class corpus(object):
    def __init__(self, corpus):
        self.corpus = corpus 
    def __iter__(self):
        for sentences in self.corpus:
            for s in sentences:
                yield s


# 1.测试 __iter__ 的class instance是可以被反复迭代的
# test_sentence = sentence1()
# print('first round:')
# for s in test_sentence:print s
# print('second round:')
# for s in test_sentence:print s

# 2.测试 generator 不可以被反复迭代的
# test_generator = (x for x in range(5))
# print('first round:')
# for s in test_generator:print s
# print('second round:')
# for s in test_generator:print s

# from ipdb import set_trace as st
# st(context=21)
logging.info('model_with_joined')
sentences =  corpus([sentence1(), sentence2()])
model_with_joined = gensim.models.Word2Vec(sentences, min_count=1, size=10, workers=1, window=2)
