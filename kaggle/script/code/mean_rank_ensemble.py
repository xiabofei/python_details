# encoding=utf8

import os
import pandas as pd

path = '../../data/mean_rank/'

candidates =  os.listdir(path)

submits = [pd.read_csv(os.path.join(path, f), index_col=0) for f in candidates]

concat_df = pd.concat(submits, axis=1)

cols = list(map(lambda x: 'target_' + str(x), range(len(concat_df.columns))))

concat_df.columns = cols


concat_df['target'] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)

concat_df.drop(cols, axis=1, inplace=True)


concat_df.to_csv('../../data/submit_candidates/mean_rank_mix.csv')
