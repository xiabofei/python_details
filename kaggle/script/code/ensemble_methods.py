# encoding=utf8

import pandas as pd
import numpy as np


def ensemble_by_rank_averaging(root_dir, sub_path_list):
    def _rank_averaging(path):
        df = pd.read_csv(path)
        rank_index = [(r, i) for r, i in enumerate(np.argsort(df['target']))]
        rank = np.array([r_i[0] for r_i in sorted(rank_index, key=lambda x: x[1])]).astype('float32')
        df['rank'] = rank / (len(df.index) - 1)
        return df['rank']

    sub = pd.read_csv(root_dir + sub_path_list[0])['id'].to_frame()
    sub['target'] = 0.0
    for sub_path in [root_dir + p for p in sub_path_list]:
        sub['target'] += _rank_averaging(sub_path)
    sub['target'] = sub['target'] / len(sub_path_list)
    sub.to_csv(root_dir + 'sub_rank_averaging.csv', index=False)


if __name__ == '__main__':
    root_dir = '../../data/output/lucky_ensemble/'
    sub_path_list = ['sub_xgb_1.csv','sub_xgb_1.csv','sub_xgb_1.csv', 'sub_xgb_2.csv', 'sub_lgbm.csv']
    ensemble_by_rank_averaging(root_dir, sub_path_list)
