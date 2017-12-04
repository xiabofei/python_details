# encoding=utf8

import os
import pandas as pd
import numpy as np

from ipdb import set_trace as st

path = '../../data/mean_rank/'

candidates = os.listdir(path)

submits = [pd.read_csv(os.path.join(path, f), index_col=0) for f in candidates]

concat_df = pd.concat(submits, axis=1)

candidate_cols = concat_df.columns

concat_df = concat_df.rank() / concat_df.shape[0]

st(context=21)
'''
target_value = np.zeros(len(concat_df.index))
nn_max_count = 0
for idx in range(len(concat_df.index)):
    if idx % 100000 == 0:
        print(idx)
    if concat_df.iloc[idx]['nn-fold20'] == concat_df.iloc[idx].max():
        nn_max_count += 1
        target_value[idx] = concat_df.iloc[idx]['nn-fold20']
    else:
        target_value[idx] = concat_df.iloc[idx].mean()
st(context=21)

'''



concat_df.to_csv('../../data/submit_candidates/concat_df.csv', index=False, sep='\t')

'''
NUM_SUBMIT = 6
submit_weight = [
    ('3xgb_lgbm_nn_lb285', 1.0 / 8),
    ('froza_and_pascal_lb284', 1.0 / 8),
    ('gpari_lb283', 1.0 / 8),
    ('mix_lb287', 1.0 / 4),
    ('nn', 1.0 / 4),
    ('rgf_lb282', 1.0 / 8)
]

concat_df['target'] = 0.0
for s_w in submit_weight:
    submit, weight = s_w[0], s_w[1]
    concat_df['target'] += concat_df[submit] * weight
'''

concat_df['target'] = concat_df.mean(axis=1)

concat_df.drop(candidate_cols, axis=1, inplace=True)

concat_df.to_csv('../../data/submit_candidates/mean_rank_mix.csv')
