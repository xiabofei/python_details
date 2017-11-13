# encoding=utf8

import pandas as pd
import numpy as np

root_dir = '../../data/for_stacker/'
single_xgb_001 = pd.read_csv(root_dir + 'sub_single_xgb_001_test.csv')
single_xgb_002 = pd.read_csv(root_dir + 'sub_single_xgb_002_test.csv')
single_xgb_003 = pd.read_csv(root_dir + 'sub_single_xgb_003_test.csv')
single_catboost_001 = pd.read_csv(root_dir + 'sub_single_catboost_001_test.csv')
single_rgf_001 = pd.read_csv(root_dir + 'sub_single_rgf_001_test.csv')
simple_stacker_001 = pd.read_csv(root_dir + 'sub_simple_stacker_001_mix_test.csv')
forza_pascal = pd.read_csv(root_dir + 'Froza_and_Pascal.csv')
aggregate_20_kernels = pd.read_csv(root_dir + 'median_rank_submission.csv')

sub = pd.DataFrame()
sub['id'] = single_xgb_001['id']
sub['target'] = np.exp(
    np.mean(
        [
            single_xgb_001['target'].apply(lambda x: np.log(x)*0.1),
            single_xgb_003['target'].apply(lambda x: np.log(x)*0.1),
            simple_stacker_001['target'].apply(lambda x: np.log(x)*0.1),
            forza_pascal['target'].apply(lambda x: np.log(x)*0.1),
            aggregate_20_kernels['target'].apply(lambda x: np.log(x)*0.4)
        ]
    ),
    axis=0
)
sub.to_csv(root_dir + 'mix5.csv', index=False)
