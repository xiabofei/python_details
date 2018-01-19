# encoding=utf8

import os
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from pprint import pprint
from ipdb import set_trace as st


def voting2(votes, fname):
    if Counter(votes)['unknown'] == 2:
        counter = dict(Counter(votes))
        return (fname, '_'.join([ str(i[0])+str(i[1]) for i in sorted(counter.items(), key=lambda t:t[0])]))
    else:
        return None

def voting3(votes, fname):
    if Counter(votes)['unknown'] == 3:
        counter = dict(Counter(votes))
        return (fname, '_'.join([ str(i[0])+str(i[1]) for i in sorted(counter.items(), key=lambda t:t[0])]))
    else:
        return None

def voting1(votes, fname):
    if Counter(votes)['unknown'] == 1:
        counter = dict(Counter(votes))
        return (fname, '_'.join([ str(i[0])+str(i[1]) for i in sorted(counter.items(), key=lambda t:t[0])]))
    else:
        return None


def voting_tie(votes, fname):
    if len(Counter(votes))>=2:
        if Counter(votes).most_common(2)[0][1]==2 and Counter(votes).most_common(2)[1][1]==2:
            counter = dict(Counter(votes))
            return (fname, '_'.join([ str(i[0])+str(i[1]) for i in sorted(counter.items(), key=lambda t:t[0])]))
    else:
        return None

if __name__ == '__main__':
    submit_candidates = '../data/output/submit_candidates/'

    vote_candidates = []
    fname_list = []
    submit_names = []
    for fname in os.listdir(submit_candidates):
        df = pd.read_csv(submit_candidates + fname, sep=',', index_col=False)
        fname_list = df['fname'].values
        vote_candidates.append(list(df['label'].values))
        submit_names.append(fname)

    vote_candidates = np.array(vote_candidates).T
    tie_fnameList = defaultdict(list)
    tie_list = [info for info in list(map(voting_tie, vote_candidates, fname_list)) if info is not None]
    for t in tie_list:
        tie_fnameList[t[1]].append(t[0])
    os.system('mkdir ../data/input/tie/')
    for tie, flist in tie_fnameList.items():
        os.system('mkdir ../data/input/tie/{0}_{1}/'.format(len(flist),tie))
        for f in flist:
            os.system('cp ../data/input/test/audio/{0} ../data/input/tie/{1}_{2}/'.format(f, len(flist), tie))

    '''
    unknown1Type_fnameList = defaultdict(list)
    unknown1_list = [info for info in list(map(voting1, vote_candidates, fname_list)) if info is not None]
    for u in unknown1_list:
        unknown1Type_fnameList[u[1]].append(u[0])
    submit = pd.DataFrame()
    submit['fname'] = [u[0] for u in unknown1_list]
    submit['count'] = [u[1] for u in unknown1_list]
    submit.to_csv('../data/input/unknown1.csv', index=False)
    os.system('mkdir ../data/input/unknown1/')
    for u1type, flist in unknown1Type_fnameList.items():
        os.system('mkdir ../data/input/unknown1/{0}_{1}/'.format(len(flist),u1type))
        for f in flist:
            os.system('cp ../data/input/test/audio/{0} ../data/input/unknown1/{1}_{2}/'.format(f, len(flist), u1type))
    '''

    '''
    unknown2Type_fnameList = defaultdict(list)
    unknown2_list = [info for info in list(map(voting2, vote_candidates, fname_list)) if info is not None]
    for u in unknown2_list:
        unknown2Type_fnameList[u[1]].append(u[0])
    submit = pd.DataFrame()
    submit['fname'] = [u[0] for u in unknown2_list]
    submit['count'] = [u[1] for u in unknown2_list]
    submit.to_csv('../data/input/unknown2.csv', index=False)
    os.system('mkdir ../data/input/unknown2/')
    for u2type, flist in unknown2Type_fnameList.items():
        os.system('mkdir ../data/input/unknown2/{0}_{1}/'.format(len(flist),u2type))
        for f in flist:
            os.system('cp ../data/input/test/audio/{0} ../data/input/unknown2/{1}_{2}/'.format(f, len(flist), u2type))


    unknown3Type_fnameList = defaultdict(list)
    unknown3_list = [info for info in list(map(voting3, vote_candidates, fname_list)) if info is not None]
    for u in unknown3_list:
        unknown3Type_fnameList[u[1]].append(u[0])
    submit = pd.DataFrame()
    submit['fname'] = [u[0] for u in unknown3_list]
    submit['count'] = [u[1] for u in unknown3_list]
    submit.to_csv('../data/input/unknown3.csv', index=False)
    os.system('mkdir ../data/input/unknown3/')
    for u3type, flist in unknown3Type_fnameList.items():
        os.system('mkdir ../data/input/unknown3/{0}_{1}/'.format(len(flist),u3type))
        for f in flist:
            os.system('cp ../data/input/test/audio/{0} ../data/input/unknown3/{1}_{2}/'.format(f, len(flist), u3type))
    '''
