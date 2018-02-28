# encoding=utf8

from data_split import label_candidates
import pandas as pd

submit_candidates = [
    ('../data/output/preds/deep_gru/lb9856_cv98997/avg_submit.csv', 'deep_gru'),
    ('../data/output/preds/skip_gru/lb9854_cv98988/avg_submit.csv', 'skip_gru'),
    ('../data/output/preds/maxpool_gru/lb9855_cv98978/avg_submit.csv', 'maxpool_gru'),
    ('../data/output/preds/largekernel_cnn/lb9843_cv9886/avg_submit.csv', 'largekernel_cnn'),
]

# load all candidates submit data
subName_subDF = []
for candidate in submit_candidates:
    subName_subDF.append((candidate[1], pd.read_csv(candidate[0])))

# cal label-wise similarity
df_corr = []
for label in label_candidates:
    print('label : {0}'.format(label))
    df = pd.DataFrame()
    for name_df in subName_subDF:
        df[name_df[0]] = name_df[1][label]
    print(df.corr())
    df_corr.append(df.corr())
    print()

# avg label-size similarity
for df in df_corr[1:]:
    df_corr[0] += df
df_avgCorr = df_corr[0] / len(label_candidates)

print(df_avgCorr)

