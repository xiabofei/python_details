# encoding=utf8
from data_split import K,label_candidates
from comm_preprocessing import ID_COL
import pandas as pd

candidates = [
    # '../data/output/preds/maxpool_cnn/with_stopwords/lb9852_cv98956_pre/',
    '../data/output/preds/maxpool_cnn/with_stopwords/lb9854_cv99001_post/',
    # '../data/output/preds/pool_cnn/lb9856_cv99042_post/',
    '../data/output/preds/pool_cnn/lb9859_cv99038_pre/',
    # '../data/output/preds/deep_gru/no_stopwords/lb9867_cv9908_post/',
    # '../data/output/preds/deep_gru/no_stopwords/lb9869_cv99053_pre/',
]
output_dir = '../data/output/preds/pool_cnn/avg_ensemble/'
# output_dir = '../data/output/preds/deep_gru/no_stopwords/avg_ensemble/'

# Average Valid (similar to more runs per fold)
for k in range(K):
    df0 = pd.read_csv(candidates[0]+'{0}fold_valid.csv'.format(k))
    df_average_valid = pd.DataFrame()
    df_average_valid[ID_COL] = df0[ID_COL]
    for label in label_candidates:
        df_average_valid[label] = df0[label] / len(candidates)
    for valid in candidates[1:]:
        df = pd.read_csv(valid+'{0}fold_valid.csv'.format(k))
        for label in label_candidates:
            df_average_valid[label] += df[label] / len(candidates)
    df_average_valid.to_csv(output_dir+'{0}fold_valid.csv'.format(k), index=False)

# Average Submit
df0 = pd.read_csv(candidates[0]+'avg_submit.csv')
df_average_submit = pd.DataFrame()
df_average_submit[ID_COL] = df0[ID_COL]
for label in label_candidates:
    df_average_submit[label] = df0[label] / len(candidates)
for submit in candidates[1:]:
    df = pd.read_csv(submit+'avg_submit.csv')
    for label in label_candidates:
        df_average_submit[label] += df[label] / len(candidates)
df_average_submit.to_csv(output_dir+'avg_submit.csv', index=False)
