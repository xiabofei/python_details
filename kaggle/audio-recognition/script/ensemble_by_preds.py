# encoding=utf8
from model_fit_enhance import AudioGenerator, n_classes, LEGAL_LABELS, RUNS_IN_FOLD, batch_size
from data_split_enhance import UNKNOWN_ENHANCE_RATE
from data_split import K
import pickle
import numpy as np
from collections import Counter
import pandas as pd
import gc

from ipdb import set_trace as st

PREDS_TEST_DIR = '../data/output/silence10/preds_test/'
PREDS_VALID_DIR = '../data/output/silence10/preds_valid/'


def load_preds(path):
    return pickle.load(open(path, 'rb'))


def get_valid_truth(fold):
    print('\nPrepare valid data in fold {0}\n'.format(fold))
    valid_generator = AudioGenerator(
        root_dir='../data/input_backup/train_augmentation/',
        k=fold,
        batch_size=batch_size,
        train_or_valid='valid',
        enhance='enhance'
    )
    _, y_truth = valid_generator.ori_data['data'], np.array(valid_generator.ori_data['label'])
    print('Fold {0} data shape {1}'.format(fold, y_truth.shape))
    return y_truth


def count_same_preds(preds_A, preds_B):
    labels_A = [LEGAL_LABELS[index] for index in np.argmax(preds_A, axis=1)]
    labels_B = [LEGAL_LABELS[index] for index in np.argmax(preds_B, axis=1)]
    same_counts = 0
    for a, b in zip(labels_A, labels_B):
        same_counts += 1 if a==b else 0
    return same_counts

def nn_average_blending(preds_data, y_truth):
    y_preds = np.zeros(y_truth.shape)
    for preds in preds_data:
        y_preds += preds / len(preds_data)
    acc = count_same_preds(y_truth, y_preds)
    return acc

def vote_ensemble(preds_data, y_truth):
    def _vote(votes):
        if Counter(votes)['unknown']>=2:
            if 'silence' in dict(Counter(votes)).keys():
                return 'silence'
            return 'unknown'
        return Counter(votes).most_common(1)[0][0]

    truth_labels = [LEGAL_LABELS[index] for index in np.argmax(y_truth, axis=1)]
    preds_labels = []
    for preds in preds_data:
        label = [LEGAL_LABELS[index] for index in np.argmax(preds, axis=1)]
        preds_labels.append(label)
    vote_candidates = np.array(preds_labels).T
    voted_label = list(map(_vote, vote_candidates))
    same = 0
    for l1, l2 in zip(truth_labels, voted_label):
        same += 1 if l1==l2 else 0
    return same


def simulate_ensemble_via_cv(y_truth_list):
    naive_count = 0
    vote_count = 0
    opt_count = 0
    all_count = 0
    for fold in range(K):

        # Select preds in fold
        preds_name, preds_data = [], []
        for enhance_idx in range(UNKNOWN_ENHANCE_RATE):
            for run in range(RUNS_IN_FOLD):
                # print('Loads preds_valid from fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                path = PREDS_VALID_DIR + 'fold{0}/enhance{1}/runs{2}_valid.pkl'.format(fold, enhance_idx, run)
                name = '_'.join(['enhance{0}'.format(enhance_idx), 'run{0}'.format(run)])
                data = load_preds(path)
                preds_name.append(name)
                preds_data.append(data)
        print(preds_name)

        # Use all naive average blending
        correct = nn_average_blending(preds_data, y_truth_list[fold])
        naive_count += correct
        all = preds_data[0].shape[0]
        all_count += all
        print('{0}/{1}'.format(correct, all))

        # Use all naive voting
        vote_count += vote_ensemble(preds_data, y_truth_list[fold])
        print('{0}/{1}'.format(vote_count, all))

        # Calculate preds similarity in fold
        assert len(preds_name)==len(preds_data), 'preds_name and preds_data not match'
        preds_similar = np.zeros((len(preds_data), len(preds_data)))
        for idx_A, preds_A in enumerate(preds_data):
            for idx_B, preds_B in enumerate(preds_data):
                preds_similar[idx_A][idx_B] = count_same_preds(preds_A, preds_B)
        select_preds_data = [ preds_data[idx] for idx in np.argsort(preds_similar.sum(axis=1))]
        select_preds_name = [ preds_name[idx] for idx in np.argsort(preds_similar.sum(axis=1))]
        select_preds_data, select_preds_name = select_preds_data[:13], select_preds_name[:13]
        print(select_preds_name)
        # Use low correlated model blending
        correct = nn_average_blending(select_preds_data, y_truth_list[fold])
        opt_count += correct
        print('{0}/{1}'.format(correct, all))
    print('naive cv : {0}'.format(naive_count / all_count))
    print('opt cv : {0}'.format(opt_count / all_count))

def get_test_fname():
    print('Prepare test data')
    test_dir = '../data/input_backup/test_augmentation/'
    d_test = pickle.load(open(test_dir + 'test_original.pkl', 'rb'))
    fname_test = d_test['fname']
    del d_test
    gc.collect()
    return fname_test

def nn_average_blending_test(preds_data):
    y_preds = np.zeros((preds_data[0].shape[0], n_classes))
    for preds in preds_data:
        y_preds += preds / len(preds_data)
    return y_preds

def ensemble():
    def _vote(votes):
        if Counter(votes)['unknown']>=2:
            if 'silence' in dict(Counter(votes)).keys():
                return 'silence'
            return 'unknown'
        return Counter(votes).most_common(1)[0][0]
    fname = get_test_fname()
    vote = []
    for fold in range(K):
        preds_name, preds_data = [], []
        for enhance_idx in range(UNKNOWN_ENHANCE_RATE):
            for run in range(RUNS_IN_FOLD):
                print('Loads preds_test from fold {0} enhance{1} run {2}'.format(fold, enhance_idx, run))
                path = PREDS_TEST_DIR + 'fold{0}/enhance{1}/runs{2}_test.pkl'.format(fold, enhance_idx, run)
                name = '_'.join(['enhance{0}'.format(enhance_idx), 'run{0}'.format(run)])
                data = load_preds(path)
                preds_name.append(name)
                preds_data.append(data)
        print(preds_name)
        preds_labels = [LEGAL_LABELS[index] for index in np.argmax(nn_average_blending_test(preds_data), axis=1)]
        vote.append(preds_labels)

    vote_candidates = np.array(vote).T
    voted_label = list(map(_vote, vote_candidates))
    submit = pd.DataFrame()
    submit['fname'] = fname
    submit['label'] = voted_label
    submit.to_csv('../data/output/submit/submit.csv', index=False)



if __name__ == '__main__':
    # y_truth_list = []
    # for k in range(K):
    #     y_truth_list.append(get_valid_truth(k))
    # simulate_ensemble_via_cv(y_truth_list)
    ensemble()
