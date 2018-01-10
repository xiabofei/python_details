# encoding=utf8

import argparse
import pandas as pd
import os


def get_cv_acc_score(valid_dir):

    def corrects(y_truth, y_preds):
        counts = 0
        for truth, preds in zip(y_truth, y_preds):
            counts += 1 if truth == preds else 0
        return counts

    correct_counts = 0
    all_counts = 0

    for f in os.listdir(valid_dir):
        df = pd.read_csv(valid_dir + f, sep=',', index_col=False)
        # count correct preds
        truth = df['truth'].values
        preds = df['preds'].values
        correct_counts += corrects(truth, preds)
        all_counts += len(truth)
    print('\n5 folds CV accuracy : %.5f ' % (correct_counts / all_counts))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enhance', type=str, default='0', help='which enhance')
    FLAGS, _ = parser.parse_known_args()
    get_cv_acc_score('../data/output/valid/enhance{0}/'.format(FLAGS.enhance))
