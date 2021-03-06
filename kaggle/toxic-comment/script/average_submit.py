# encoding=utf8

from data_split import K,label_candidates
from comm_preprocessing import ID_COL
import pandas as pd

# submit0 = '../data/output/preds/maxpool_cnn/with_stopwords/avg_ensemble/avg_submit.csv'
submit0 = '../data/output/preds/avg_ensemble_rnn/avg_submit.csv'
submit1 = '../data/output/preds/avg_ensemble_cnn/avg_submit.csv'
# submit2 = '../data/output/kernel_blends/hight_of_blend_v2.csv'
# submit1 = '../data/output/preds/glove_fasttext_cnn/avg_submit_98792.csv'
# submit2 = '../data/output/preds/glove_fasttext_cnn/lb9843_cv9886/avg_submit.csv'
# submit2 = '../data/output/preds/fasttext_gru/avg_submit_9883.csv'
candidates = [submit0, submit1]

df0 = pd.read_csv(candidates[0])
df_average_submit = pd.DataFrame()
df_average_submit[ID_COL] = df0[ID_COL]
for label in label_candidates:
    df_average_submit[label] = df0[label] / len(candidates)

for submit in candidates[1:]:
    df = pd.read_csv(submit)
    for label in label_candidates:
        df_average_submit[label] += df[label] / len(candidates)

df_average_submit.to_csv('../data/output/avg_submit.csv', index=False)


