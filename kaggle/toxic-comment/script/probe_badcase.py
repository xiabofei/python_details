# encoding=utf8
import pandas as pd
import numpy as np
from data_split import K
from data_split import label_candidates
from comm_preprocessing import ID_COL
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from comm_preprocessing import COMMENT_COL

from ipdb import set_trace as st

from data_split import TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE

# labels = [TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE]
labels = [TOXIC]
threshold = 0.5

# root_dir = '../data/output/preds/glove_gru/0.3/'
# root_dir = '../data/output/preds/glove_gru/0.35/'
root_dir = '../data/output/preds/glove_gru/0.375/0.3/'
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
df_train_processed = pd.read_csv('../data/input/data_comm_preprocessed/train.csv')
df_train = pd.read_csv('../data/input/train.csv')
ids = df_train[ID_COL].values.tolist()


# calculate label-wise roc-auc value
for label in labels:
    y_true = df_train[label].values
    # get oof valid preds
    id_preds = {}
    for k in range(K):
        print('fold {0}'.format(k))
        df = pd.read_csv(root_dir+'{0}fold_'.format(k) + 'valid.csv')
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

    # catch bad case
    false_negative_idx = []
    false_positive_idx = []
    for idx in range(y_true.shape[0]):
        t = y_true[idx]
        p = y_score[idx]
        if t==1 and p<threshold:
            false_negative_idx.append(idx)
        if t==0 and p>threshold:
            false_positive_idx.append(idx)

    ori_ids = df_train[ID_COL].values
    comments = df_train[COMMENT_COL].values
    comments_p = df_train_processed[COMMENT_COL].values

    df_FN = pd.DataFrame()
    df_FN[ID_COL] = ori_ids[false_negative_idx]
    df_FN[COMMENT_COL] = comments[false_negative_idx]
    for idx, l in enumerate(label_candidates):
        df_FN[l] = df_train[l].values[false_negative_idx]
    df_FN.to_csv('./FN_{0}.csv'.format(label), index=False)

    df_FN_p = pd.DataFrame()
    df_FN_p[ID_COL] = ori_ids[false_negative_idx]
    df_FN_p[COMMENT_COL] = comments_p[false_negative_idx]
    for idx, l in enumerate(label_candidates):
        df_FN_p[l] = df_train[l].values[false_negative_idx]
    df_FN_p.to_csv('./FN_{0}_p.csv'.format(label), index=False)

    df_FP = pd.DataFrame()
    df_FP[ID_COL] = ori_ids[false_positive_idx]
    df_FP[COMMENT_COL] = comments[false_positive_idx]
    for idx ,l in enumerate(label_candidates):
        df_FP[l] = df_train[l].values[false_positive_idx]
    df_FP.to_csv('./FP_{0}.csv'.format(label), index=False)

    df_FP_p = pd.DataFrame()
    df_FP_p[ID_COL] = ori_ids[false_positive_idx]
    df_FP_p[COMMENT_COL] = comments_p[false_positive_idx]
    for idx ,l in enumerate(label_candidates):
        df_FP_p[l] = df_train[l].values[false_positive_idx]
    df_FP_p.to_csv('./FP_{0}_p.csv'.format(label), index=False)

    y_score = y_score > threshold
    print(confusion_matrix(y_true, y_score))

