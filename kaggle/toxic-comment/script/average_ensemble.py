# encoding=utf8
from data_split import K,label_candidates
from comm_preprocessing import ID_COL
import pandas as pd

candidates = [
    ## rnn
    '../data/output/preds/deep_gru/no_stopwords/fasttext/lb9864_cv9911_post/',
    '../data/output/preds/deep_gru/no_stopwords/lb9869_cv99053_pre/',
    # '../data/output/preds/maxpool_gru/no_stopwords/lb9864_cv99027_pre/',
    # '../data/output/preds/maxpool_gru/no_stopwords/lb9863_cv9907_post/',

    ## cnn
    # '../data/output/preds/pool_cnn_skip/fasttext/lb9861_cv99042_pre/',
    # '../data/output/preds/pool_cnn/lb9859_cv99038_pre/',
    # '../data/output/preds/maxpool_deep_cnn/lb9857_cv99053_post/',
    # '../data/output/preds/maxpool_cnn/with_stopwords/lb9855_cv99035_post/',
    # '../data/output/preds/maxpool_cnn/with_stopwords/lb9852_cv98956_pre/',
]
# output_dir = '../data/output/preds/avg_ensemble_cnn/'
output_dir = '../data/output/preds/avg_ensemble_rnn/'

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
