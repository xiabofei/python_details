# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter
from ipdb import set_trace as st


def voting(votes):
    return Counter(votes).most_common(1)[0][0]


def get_voting_result(input_dir, output_path):
    vote_candidates, fname_list, submit_names = [], [], []
    for fname in os.listdir(input_dir):
        df = pd.read_csv(input_dir + fname, sep=',', index_col=False)
        fname_list = df['fname'].values
        vote_candidates.append(list(df['label'].values))
        submit_names.append(fname)
    vote_candidates = np.array(vote_candidates).T
    voted_label = list(map(voting, vote_candidates))
    tmp = [(f, v) for f, v in zip(fname_list, voted_label)]
    tmp.sort(key=lambda x: x[0])
    submit = pd.DataFrame()
    submit['fname'] = [i[0] for i in tmp]
    submit['label'] = [i[1] for i in tmp]
    submit.to_csv(output_path, index=False)
    return submit['fname'], submit['label']

if __name__ == '__main__':

    root_dir = '../data/output/submit_candidates/'
    output_dir = '../data/output/submit/'

    # Get original submit
    df = pd.read_csv('../data/output/submit_candidates/submit_cv9667.csv', index_col=False)
    fname_list = df['fname'].values
    original_label = df['label'].values
    tmp = [(f, v) for f, v in zip(fname_list, original_label)]
    tmp.sort(key=lambda x: x[0])
    fname_list = [i[0] for i in tmp]
    original_label = [i[1] for i in tmp]

    # Get tta submit
    tta0_subdir = 'tta0/'
    tta1_subdir = 'tta1/'
    tta2_subdir = 'tta2/'
    tta3_subdir = 'tta3/'
    tta4_subdir = 'tta4/'
    tta_candidates = [tta0_subdir, tta1_subdir, tta2_subdir, tta3_subdir, tta4_subdir]
    tta_submit = []
    for idx, tta_subdir in enumerate(tta_candidates):
        _, submit = get_voting_result(root_dir + tta_subdir, output_dir + 'tta{0}_submit.csv'.format(idx))
        tta_submit.append(submit)

    vote_candidates = []
    vote_candidates.append(original_label)
    vote_candidates.extend(tta_submit)

    # record original and tta result
    submit = pd.DataFrame()
    submit['fname'] = fname_list
    submit['original'] = vote_candidates[0]
    submit['tta0'] = vote_candidates[1]
    submit['tta1'] = vote_candidates[2]
    submit['tta2'] = vote_candidates[3]
    submit['tta3'] = vote_candidates[4]
    submit['tta4'] = vote_candidates[5]
    submit.to_csv('../data/output/submit/submit_all.csv', sep='\t', index=False)
    vote_candidates = np.array(vote_candidates).T

    voted_label = list(map(voting, vote_candidates))

    submit = pd.DataFrame()
    submit['fname'] = fname_list
    submit['label'] = voted_label
    submit.to_csv('../data/output/submit/submit_cv9667_tta.csv', index=False)
