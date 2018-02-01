# encoding

import pandas as pd
import numpy as np
from math import log

test_file = '../data/input/test.csv'
hack_submit = '../data/input/hack_submit.csv'
df = pd.read_csv(test_file)
del df['comment_text']
N_samples = len(df.index)

COMM_PROB = 0.5
HACK_PROB = 0.2

from keras.losses import binary_crossentropy


def create_hack_submit():
    def common_prob():
        return np.array([COMM_PROB] * N_samples)

    def hack_prob():
        return np.array([HACK_PROB] * N_samples)

    df['toxic'] = common_prob()
    df['severe_toxic'] = common_prob()
    df['obscene'] = common_prob()
    df['threat'] = common_prob()
    df['insult'] = common_prob()
    df['identity_hate'] = hack_prob()

    df.to_csv(hack_submit, index=False)


def estimate_1_proportion(public_score):
    '''
    Column-wise equation :
        p * ln(1/0.2) + ( 1- p ) * ln(1/0.8) = 6*score - 5*ln(2)
    p :
        p = ( 6*score - 5ln(2) - ln(5/4) ) / ( ln(5) - ln(5/4) )
    '''
    numerator = 6 * public_score - 5 * log(2) - log(5/4)
    denominator = log(5) - log(5/4)
    return numerator / denominator


if __name__ == '__main__':
    # create_hack_submit()
    # print(estimate_1_proportion(0.617))
    pass
