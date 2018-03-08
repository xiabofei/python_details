# encoding=utf8

from data_split import label_candidates
import pandas as pd

submit_candidates = [
    ('../data/output/preds/deep_gru/no_stopwords/lb9865_cv99109_post/avg_submit.csv', 'deep_gru_post'),
    ('../data/output/preds/deep_gru/no_stopwords/lb9866_cv9905_pre/avg_submit.csv', 'deep_gru_pre'),
    # ('../data/output/preds/deep_gru/no_stopwords/fasttext/lb9864_cv9907/avg_submit.csv', 'fasttext_deep_gru'),
    ('../data/output/preds/skip_gru/no_stopwords/lb9864_cv99029/avg_submit.csv', 'skip_gru'),
    ('../data/output/preds/maxpool_gru/no_stopwords/lb9864_cv99027/avg_submit.csv', 'maxpool_gru'),
    # ('../data/output/preds/gru_conv1d/lb9862_cv99011/avg_submit.csv', 'gru_conv1d'),
    # ('../data/output/preds/maxpool_cnn/with_stopwords/lb9852_cv98964/avg_submit.csv', 'maxpool_cnn'),
    # ('../data/output/preds/lr/avg_submit.csv', 'lr'),
    # ('../data/output/preds/nbsvm/lb9782_cv98168/avg_submit.csv', 'nbsvm'),
    # ('../data/output/preds/gbdt/lb9776_cv9820/avg_submit.csv', 'gbdt'),
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

