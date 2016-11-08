#encoding=utf8
"""
http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
关于如何理解每个leaf node中如何判断category的内容
参考
http://stackoverflow.com/questions/23557545/how-to-explain-the-decision-tree-from-scikit-learn
"""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# from ipdb import set_trace as st
# st(context=21)
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier( max_leaf_nodes = 3, random_state = 0 )
estimator.fit(X_train, y_train)

# The decision estimator has an attribute called tree_ which stores the entire 
# tree structure and allows access to low level attributes. The binary tree tree_
# is represented as a number of parallel arrays. 
# The i-th element of each array holds information about the node 'i'. 
# Node 0 is the tree's root.
# NOTE: Some of the arrays only apply to either leaves or split nodes, resp.
# In this case the values of nodes of the other type are arbitrary!

# Among those arrays, we have:
#   -left_child, id of the left child of the node
#   -right_child, id of the right child of the node
#   -feature, feature used for splitting the node
#   -threshold, threshold value at the node

# Using those arrays, we can parse the tree structure:
n_nodes = estimator.tree_.node_count
print "n_nodes:"
print "    " + str(n_nodes)
children_left = estimator.tree_.children_left
print "children_left:"
print "    " + str(children_left)
children_right = estimator.tree_.children_right
print "children_right:"
print "    " + str(children_right)
feature = estimator.tree_.feature
print "feature:"
print "    " + str(feature)
threshold = estimator.tree_.threshold
print "threshold:"
print "    " + str(threshold)

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes)
is_levaes = np.zeros(shape=n_nodes, dtype = bool)
# seed is the root node id and its parent depth
stack = [(0, 1)]
while len(stack)>0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1
    # If we have test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append(( children_left[node_id], parent_depth+1 ))
        stack.append(( children_right[node_id], parent_depth+1 ))
    else:
        is_levaes[node_id] = True

print("The binary tree structure has %s nodes and has the following tree structure:" % n_nodes)

for i in range(n_nodes):
    if is_levaes[i]:
        print("%snode=%s leaf node. " % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to node %s."
                % (
                    node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                    )
            )
print()

# Frist let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i,j) indicates that the sample i goes
# through the node j.
node_indicator = estimator.decision_path(X_test)
# Similarly, we can also have the leaves ids reached by each sample.
leave_id = estimator.apply(X_test)
# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample
sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]
print('Rules used to predict sample %s: ' % sample_id)
for node_id in node_index:
    if leave_id[sample_id] != node_id:
        continue
    if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"
    
    print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
            %(  node_id,
                sample_id,
                feature[node_id],
                X_test[i, feature[node_id]],
                threshold_sign,
                threshold[node_id]
                ))

# For a group of samples, we have the following common node.
sample_ids = [0, 1]
common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                        len(sample_ids))

common_node_id = np.arange(n_nodes)[common_nodes]

print("\nThe following samples %s share the node %s in the tree" % (sample_ids, common_node_id))
print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
