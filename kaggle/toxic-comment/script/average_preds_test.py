# encoding=utf8

from data_split import K,label_candidates
from comm_preprocessing import ID_COL
import pandas as pd
import zipfile

#root_dir = '../data/output/preds/deep_gru/no_stopwords/add_star/0.4/'
root_dir = '../data/output/preds/deep_gru/no_stopwords/tta/0.4/'

df_0fold_test = pd.read_csv(root_dir+'0fold_test.csv')
df_average_submit = pd.DataFrame()
df_average_submit[ID_COL] = df_0fold_test[ID_COL]
for label in label_candidates:
    df_average_submit[label] = df_0fold_test[label] / K

for k in range(1, K):
    df = pd.read_csv(root_dir+'{0}fold_test.csv'.format(k))
    for label in label_candidates:
        df_average_submit[label] += df[label] / K

df_average_submit.to_csv(root_dir+'avg_submit.csv', index=False)


