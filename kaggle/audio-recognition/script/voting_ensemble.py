# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from pprint import pprint
from ipdb import set_trace as st

counts = 0

def voting(votes):
    # global counts
    # if Counter(votes)['unknown']>=2:
    #     return 'unknown'
    return Counter(votes).most_common(1)[0][0]

if __name__ == '__main__':
    submit_candidates = '../data/output/submit_candidates/'


    vote_candidates = []
    fname_list = []
    submit_names = []
    for fname in os.listdir(submit_candidates):
        df = pd.read_csv(submit_candidates + fname, sep=',', index_col=False)
        fname_list = df['fname'].values
        vote_candidates.append(list(df['label'].values))
        submit_names.append(fname)

    vote_candidates = np.array(vote_candidates).T


    voted_label = list(map(voting, vote_candidates))

    submit = pd.DataFrame()
    submit['fname'] = fname_list
    submit['label'] = voted_label
    submit.to_csv('../data/output/submit/submit.csv', index=False)
    # pprint('counts : {0}'.format(counts))


