# encoding=utf8
import pandas as pd
import numpy as np
from data_split import K
from data_split import label_candidates
from comm_preprocessing import ID_COL
from sklearn.metrics import roc_auc_score

from ipdb import set_trace as st

from data_split import TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE

labels = [TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE]

# root_dir = '../data/output/preds/glove_gru/0.3/'
# root_dir = '../data/output/preds/glove_gru/0.35/'
root_dir = '../data/output/preds/glove_gru/0.375/0.2/'
# root_dir = '../data/output/preds/glove_gru/0.4/'
# root_dir = '../data/output/preds/glove_gru/0.45/'
# root_dir = '../data/output/preds/glove_gru/0.5/'
# root_dir = '../data/output/preds/glove_gru/0.55/'
# root_dir = '../data/output/preds/glove_gru/0.6/'
# root_dir = '../data/output/preds/glove_cnn/0.35/'
# root_dir = '../data/output/preds/glove_cnn/0.4/'
# root_dir = '../data/output/preds/glove_cnn/0.45/'
# root_dir = '../data/output/preds/glove_cnn/0.5/'
# root_dir = '../data/output/preds/glove_cnn/0.55/'
# root_dir = '../data/output/preds/glove_cnn/0.6/'
#root_dir = '../data/output/preds/glove_cnn/0.625/'
# root_dir = '../data/output/preds/gbdt/'
# root_dir = '../data/output/preds/lr/'
#root_dir = '/home/baibing/Kaggle/Toxic/data/output/stack_ensemble/'
#root_dir = '../data/output/preds/cnn_rnn/0.2/'

# get id and true label
df_train = pd.read_csv('../data/input/train.csv')
ids = df_train[ID_COL].values.tolist()

# calculate label-wise roc-auc value
for label in labels:
    y_true = df_train[label].values
    # get oof valid preds
    id_preds = {}
    for k in range(K):
        print('fold {0}'.format(k))
        df = pd.read_csv(root_dir+'{0}fold_{1}_valid.csv'.format(k, label))
        id_list = df[ID_COL].values.tolist()
        preds_list = df[label].values.tolist()
        for id, preds in zip(id_list, preds_list):
            id_preds[id] = preds

    # match valid preds and origin label by id
    y_score = []
    for id in ids:
        y_score.append(id_preds[id])
    y_score = np.array(y_score)

    # single Label AUC
    score = roc_auc_score(y_true=y_true, y_score=y_score)
    print('{0} : {1}'.format(label, score))

