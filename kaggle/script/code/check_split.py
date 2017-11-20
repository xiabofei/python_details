#encoding=utf8

import pickle
import os
import pandas as pd

from ipdb import set_trace as st


path = '../../data/split_idx/'

candidates =  os.listdir(path)

idx_dict = {f:pickle.load(open(os.path.join(path, f),'r')) for f in candidates}


