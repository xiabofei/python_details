# encoding=utf8

import os
import pandas as pd

from ipdb import set_trace as st

path = '../../data/mean_rank/'

candidates =  os.listdir(path)

submits = [pd.read_csv(os.path.join(path, f), index_col=0) for f in candidates]

concat_df = pd.concat(submits, axis=1)


candidate_cols = concat_df.columns

concat_df = concat_df.rank() / concat_df.shape[0]
# concat_df = concat_df.rank()


concat_df.to_csv('../../data/submit_candidates/concat_df.csv', index=False, sep='\t')


concat_df['target'] = concat_df.mean(axis=1)

concat_df.drop(candidate_cols, axis=1, inplace=True)


concat_df.to_csv('../../data/submit_candidates/mean_rank_mix.csv')
