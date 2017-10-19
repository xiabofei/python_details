# encoding=utf8

import numpy as np

class GiniEvaluation(object):

    @classmethod
    def gini(cls, actual, pred, cmpcol=0, sortcol=1):
        assert (len(actual) == len(pred))
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)

    @classmethod
    def gini_normalized(cls, a, p):
        return cls.gini(a, p) / cls.gini(a, a)

    @classmethod
    def gini_xgb(cls, preds, dtrain):
        labels = dtrain.get_label()
        gini_score = cls.gini_normalized(labels, preds)
        return [('gini', -1* gini_score)]

    @classmethod
    def gini_lgb(cls, preds, dtrain):
        labels = dtrain.get_label()
        gini_score = cls.gini_normalized(labels, preds)
        return [('gini', gini_score, True)]
