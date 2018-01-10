# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from pprint import pprint
from ipdb import set_trace as st

from fe_and_augmentation import LEGAL_LABELS
from fe_and_augmentation import LABEL_INDEX
from pprint import pprint

confusion_matrix = np.zeros((len(LABEL_INDEX), len(LABEL_INDEX)))

def judge_pair(truth, preds):
    global confusion_matrix
    for t, p in zip(truth, preds):
        if t==p:
            continue
        else:
            confusion_matrix[LABEL_INDEX[p]][LABEL_INDEX[t]] += 1

if __name__ == '__main__':
    valid_candidates = '../data/output/valid/'


    # check all valid result
    vote_candidates = []
    valid_names = []
    for fname in os.listdir(valid_candidates):
        df = pd.read_csv(valid_candidates + fname, sep=',', index_col=False)
        truth = df['truth'].values
        preds = df['preds'].values
        judge_pair(truth, preds)

    print(LEGAL_LABELS)
    print(confusion_matrix)




