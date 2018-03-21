import pandas as pd
from sklearn.model_selection import KFold

TOXIC = 'toxic'
SEVERE_TOXIC = 'severe_toxic'
OBSCENE = 'obscene'
THREAT = 'threat'
INSULT = 'insult'
IDENTITY_HATE = 'identity_hate'
label_candidates = [TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE]

K = 6

if __name__ == '__main__':
    df_train = pd.read_csv('../data/input/train.csv')

    # 6 folds data split
    data_split_dir = '../data/input/data_split/'
    kf = KFold(n_splits=K, shuffle=True, random_state=2018)
    last_trn_idx = []
    last_val_idx = []
    for k, (trn_idx, val_idx) in enumerate(kf.split(df_train)):
        print('data split for fold {0}'.format(k))
        print('  train length : {0}'.format(len(trn_idx)))
        print('  valid length : {0}'.format(len(val_idx)))
        trn_in_fold = df_train.iloc[trn_idx]
        val_in_fold = df_train.iloc[val_idx]
        assert len(list(set(trn_idx).intersection(set(val_idx)))) == 0, 'data leak in train and valid'
        assert len(list(set(val_idx).intersection(set(last_val_idx)))) == 0, 'data leak among valid'
        print('intersection among train {0}'.format(len(list(set(trn_idx).intersection(set(last_trn_idx)))) / len(trn_idx)))
        last_trn_idx, last_val_idx = trn_idx, val_idx
        trn_in_fold.to_csv(data_split_dir + '{0}_train.csv'.format(k), index=False)
        val_in_fold.to_csv(data_split_dir + '{0}_valid.csv'.format(k), index=False)


    # check label distribution among folds
    def check_label_distribution(df):
        return {label: '%.5f' % (df[label].sum() / len(df.index)) for label in label_candidates}


    for k in range(K):
        print('fold {0} label distribution:'.format(k))
        tmp_trn = pd.read_csv(data_split_dir + '{0}_train.csv'.format(k))
        print('  train {0}'.format(check_label_distribution(tmp_trn)))
        tmp_val = pd.read_csv(data_split_dir + '{0}_valid.csv'.format(k))
        print('  valid {0}'.format(check_label_distribution(tmp_val)))
