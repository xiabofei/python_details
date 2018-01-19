# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter

def voting(votes):
    return Counter(votes).most_common(1)[0][0]

if __name__ == '__main__':
    submit_candidates = '../data/output/submit_candidates/'

    vote_candidates = []
    fname_list = []
    submit_names = []
    for fname in os.listdir(submit_candidates):
        df = pd.read_csv(submit_candidates + fname, sep=',', index_col=False)
        fname_list = df['fname'].values
        tmp = [(f, v) for f, v in zip(df['fname'].values, df['label'].values)]
        tmp.sort(key=lambda x: x[0])
        fname_list = [i[0] for i in tmp]
        label_list = [i[1] for i in tmp]
        vote_candidates.append(label_list)

    vote_candidates = np.array(vote_candidates).T

    voted_label = list(map(voting, vote_candidates))

    submit = pd.DataFrame()
    submit['fname'] = fname_list
    submit['label'] = voted_label
    submit.to_csv('../data/output/submit/submit_9_voting.csv', index=False)
