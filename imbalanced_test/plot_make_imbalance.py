import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

from imblearn.datasets import make_imbalance

print(__doc__)

from ipdb import set_trace as st
st(context=21)

sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

# Generate the dataset
X, y = make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=10)

f, axs = plt.subplots(1, 2)

# Original
axs[0].scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0",
            alpha=0.5, facecolor=palette[0],
            linewidth=0.15)
axs[0].scatter(X[y == 1, 0], X[y == 1, 1], label="Class #0",
            alpha=0.5, facecolor=palette[2],
            linewidth=0.15)
# Make imbalance
X_, y_ = make_imbalance(X, y, ratio=0.5, min_c_=1)
X_0, y_0 = make_imbalance(X, y, ratio=0.5, min_c_=0)
# After making imbalance
axs[1].scatter(X_[y_ == 0, 0], X_[y_ == 0, 1], label="Class #0",
            alpha=0.5, facecolor=palette[0],
            linewidth=0.15)
axs[1].scatter(X_[y_ == 1, 0], X_[y_ == 1, 1], label="Class #0",
            alpha=0.5, facecolor=palette[2],
            linewidth=0.15)
plt.show()
