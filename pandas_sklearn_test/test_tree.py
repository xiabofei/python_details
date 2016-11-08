#encoding=utf8
"""
学习sklearn中tree的用法
资料来源:
http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

运行这段程序 对输出结果的分析如下:
    把四个array：left，right，threshold 的节点这么理解：
        node 0是分叉节点：
            左边是node 1，右边是node 2，分叉特征是feature 0，分叉规则是feature0 ≤0.5往left走（否则往右走）
        node 1是叶子节点：
            左右都是-1
        node 2是分叉节点：
            左边是node 3，右边是node 4，分叉特征是feature 1， 分叉的规则是feature1≤4.5往left走（否则往右走）
        node 3是叶子节点：
            左右都是-1
        node 4是分叉节点：
            左边是node 5，右边是node 6，分叉特征是feature 0，分叉的规则是feature≤2.5往left走（否则往右走）
        node 5，node 6都是叶子节点：左右都是-1

"""
import pandas as pd
import numpy  as np
from sklearn.tree import DecisionTreeClassifier

def get_lineage(tree, feature_names):
    # from ipdb import set_trace as st
    # st(context=21)
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [ feature_names[i] for i in tree.tree_.feature ]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for child in idx:
        for node in recurse(left, right, child):
            print node

# dummy data
df = pd.DataFrame({'col1':[0,1,2,3], 'col2':[3,4,5,6], 'dv':[0,1,0,1]})
print df.ix[:,:3]

# create decision tree
dt = DecisionTreeClassifier( max_depth = 5, min_samples_leaf = 1 )
dt.fit(df.ix[:,:2], df.dv)
print "left child array:"
print "    " + str(dt.tree_.children_left)
print "right child array:"
print "    " + str(dt.tree_.children_right)
print "threshold value array:"
print "    " + str(dt.tree_.threshold)
print len(dt.tree_.threshold)
print "feature array:"
print "    " + str(dt.tree_.feature)

# 遍历内容
get_lineage(dt, df.columns)
