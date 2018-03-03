# encoding=utf8

import pandas as pd
from comm_preprocessing import COMMENT_COL
from comm_preprocessing import toxic_indicator_words

toxic_indicator_words = set(toxic_indicator_words)

def check_toxic_word(t):
    for w in t.split():
        if w in toxic_indicator_words:
            return True
    return False


df_train = pd.read_csv('../data/input/data_comm_preprocessed/train.csv')


