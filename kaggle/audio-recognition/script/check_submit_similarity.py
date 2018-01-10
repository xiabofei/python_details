# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict

def similarity(li, lj):
    L = len(li)
    S = 0
    for i,j in zip(li, lj):
        S += 1 if i==j else 0
    return '%.4f'% (S/L)

if __name__ == '__main__':
    submit_candidates = '../data/output/submit_candidates/'
    N = os.listdir(submit_candidates)
    similar_matrix = np.zeros((len(N),len(N)))
    vote_candidates = []
    fname_list = []
    submit_names = []
    for fname in os.listdir(submit_candidates):
        df = pd.read_csv(submit_candidates + fname, sep=',', index_col=False)
        fname_list = df['fname'].values
        vote_candidates.append(list(df['label'].values))
        submit_names.append(fname)


    for i in range(len(vote_candidates)):
        label_i = vote_candidates[i]
        for j in range(len(vote_candidates)):
            label_j = vote_candidates[j]
            similar_matrix[i][j] = similarity(label_i, label_j)

    print('similar matrix :\n {0}'.format(similar_matrix))

