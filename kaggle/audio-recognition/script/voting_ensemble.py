# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter

from ipdb import set_trace as st


submit_candidates = '../data/output/submit_candidates/'


vote_candidates = []
fname_list = []
for fname in os.listdir(submit_candidates):
    df = pd.read_csv(submit_candidates + fname, sep=',', index_col=False)
    fname_list = df['fname'].values
    vote_candidates.append(list(df['label'].values))

unknown_counts = 0

def voting(votes):
    # 3rd 85
    '''
    if Counter(votes).most_common(2)[0][1] == 2 \
            and Counter(votes).most_common(2)[1][1] == 2 \
            and Counter(votes).most_common(2)[1][0] == 'unknown':
        return 'unknown'
    # 1st 85 maybe overfitting to public LB !
    if Counter(votes).most_common(2)[0][1] == 3 \
            and Counter(votes).most_common(2)[1][1] == 2 \
            and Counter(votes).most_common(2)[1][0] == 'unknown':
        return 'unknown'
    # 2nd 85
    if 'unknown' in votes:
        return 'unknown'
    '''
    if Counter(votes)['unknown']>=6:
        return 'unknown'
    return Counter(votes).most_common(1)[0][0]


vote_candidates = np.array(vote_candidates).T


voted_label = list(map(voting, vote_candidates))

submit = pd.DataFrame()
submit['fname'] = fname_list
submit['label'] = voted_label
submit.to_csv('../data/output/submit/submit.csv', index=False)


