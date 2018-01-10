# encoding=utf8

import os
import numpy as np
import pandas as pd
from data_split import K, TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP
from collections import Counter

from ipdb import set_trace as st

data_dir = '../data/input/train/audio/'

print()

for k in range(K):
    train_in_fold = []
    train_label = []
    with open(data_dir+str(k)+TRAIN_SPLIT_FILE_TEMP, 'r') as f_train:
        for l in f_train.readlines():
            train_in_fold.append(l.strip().split('\t')[1])
            train_label.append(l.strip().split('\t')[0])
    print(Counter(train_label))
    valid_in_fold = []
    valid_label = []
    with open(data_dir+str(k)+VALID_SPLIT_FILE_TEMP, 'r') as f_valid:
        for l in f_valid.readlines():
            valid_in_fold.append(l.strip().split('\t')[1])
            valid_label.append(l.strip().split('\t')[0])
    print(Counter(valid_label))
    print(list(set(train_in_fold).intersection(set(valid_in_fold))))



