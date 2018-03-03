# encoding=utf8

import pandas as pd
from comm_preprocessing import COMMENT_COL
from nltk import tokenize

# train_toxic = pd.read_csv('../data/input/train_toxic.csv')
# train_toxic = pd.read_csv('../data/input/train.csv')
train_toxic = pd.read_csv('../data/input/test.csv')


def split_sentence_from_document(document):
    max_counts = 0
    for sent in tokenize.sent_tokenize(document):
        max_counts = max(max_counts, len(tokenize.wordpunct_tokenize(sent)))
    # if max_counts>4000:
    #     print(document)
    return max_counts


train_toxic['word_counts'] = train_toxic[COMMENT_COL].apply(split_sentence_from_document)
print('done')
