# encoding=utf8

import pandas as pd
import os


fold_results_dir = '../data/output/cv_results/'


def corrects(y_truth, y_preds):
    counts = 0
    for truth, preds in zip(y_truth, y_preds):
        counts += 1 if truth == preds else 0
    return counts

correct_counts = 0
all_counts = 0

last_fname = []

for f in os.listdir(fold_results_dir):
    df = pd.read_csv(fold_results_dir + f, sep=',', index_col=False)

    # check data leak
    fname = df['fname'].values
    assert len(list(set(fname).intersection(set(last_fname)))) == 0, 'intersection data among folds valid'
    last_fname = fname

    # count correct preds
    truth = df['truth'].values
    preds = df['preds'].values
    correct_counts += corrects(truth, preds)
    all_counts += len(truth)

print('\n5 folds CV accuracy : %.5f ' % (correct_counts / all_counts))
