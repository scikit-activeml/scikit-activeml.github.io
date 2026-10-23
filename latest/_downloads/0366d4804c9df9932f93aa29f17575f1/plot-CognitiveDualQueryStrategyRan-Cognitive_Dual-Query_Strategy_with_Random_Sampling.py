"""
Cognitive Dual-Query Strategy with Random Sampling
==================================================
"""

# %%
# **Idea:** The Cognitive Dual-Query Strategy with Random Sampling query strategy is an extension to the uncertainty based query strategies proposed by Žliobaitė et al. and follows the same idea as DBALStream where querying a labels is only allowed if the local density around the corresponding sample is sufficiently high. The authors propose the use of a cognitive window that monitors the most representative samples within a data stream.

# %%
# | **Google Colab Note**: If the notebook fails to run after installing the
#   needed packages, try to restart the runtime (Ctrl + M) under
#   Runtime -> Restart session.
#
# .. image:: https://colab.research.google.com/assets/colab-badge.svg
#    :target: https://colab.research.google.com/github/scikit-activeml/scikit-activeml.github.io/blob/gh-pages/1.0/generated/sphinx_gallery_notebooks//4-stream/plot-CognitiveDualQueryStrategyRan-Cognitive_Dual-Query_Strategy_with_Random_Sampling.ipynb
#
# | **Notebook Dependencies**
# | Uncomment the following cell to install all dependencies for this
#   tutorial.

# !pip install scikit-activeml

# %%
import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL
from skactiveml.visualization import (
    plot_stream_training_data,
    plot_stream_decision_boundary,
)

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.stream import CognitiveDualQueryStrategyRan

# Set a fixed random state for reproducibility.
random_state = np.random.RandomState(0)

# Initial training set size.
init_size = 0  # e.g. 0 or any integer value

# Build a dataset.
X, y_true = make_blobs(
    n_samples=200 + init_size,
    n_features=1,
    centers=[[0], [-3], [1], [2], [-0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
y_true = y_true % 2  # Convert labels to binary (0, 1)

# Split the data into initial training and streaming parts.
X_init = X[:init_size]
y_init = y_true[:init_size]
X_stream = X[init_size:]
y_stream = y_true[init_size:]

# Initialise the classifier.
clf = ParzenWindowClassifier(classes=[0, 1], random_state=random_state)

# Initialise the query strategy.
qs = CognitiveDualQueryStrategyRan(budget=0.2)
plot_step = 5

# Initialize training data with initial examples.
X = list(X_init)
y = list(y_init)
classes = np.unique(y_true)

# Preparation for plotting.
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.825)
feature_bound = [[0, len(X_stream)], [min(X_stream), max(X_stream)]]
ax.set_xlim(0, len(X_stream))
ax.set_ylim(bottom=min(X_stream), top=max(X_stream))
artists = []  # List to store frames for the animation

# List to track whether each sample was queried (True) or not (False).
queried_indices = [True] * len(y_init)
# List to store decision boundary predictions over time.
predictions_list = []
# List to store if the current samples were classified correctly.
correct_classification_list = []

# Process each streaming sample.
for t_x, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):

    X_cand = x_t.reshape([1, -1])
    y_cand = [y_t]

    # get prediction of current sample


    # Fit the classifier with current training data.
    clf.fit(X, y)

    # Check whether to query the current sample or not.
    sampled_indices, utilities = qs.query(
        candidates=X_cand, clf=clf, return_utilities=True
    )
    budget_manager_param_dict = {"utilities": utilities}
    # Update the query strategy and budget manager to calculate the right budget.
    qs.update(candidates=X_cand, queried_indices=sampled_indices)

    # Label the sample based on whether it was queried.
    X.append(x_t)
    y.append(y_t if len(sampled_indices) else MISSING_LABEL)

    queried_indices.append(len(sampled_indices) > 0)

    # Plot the current state at intervals defined by plot_step.
    if t_x % plot_step == 0:
        # Save current plot elements to determine what is new.
        coll_old = list(ax.collections)
        ax, predictions_list = plot_stream_decision_boundary(
            ax, t_x, plot_step, clf, X_stream, predictions_list, res=25
        )
        data_lines = plot_stream_training_data(
            ax, X, y, queried_indices, classes, feature_bound
        )

        correct_classification_list.append(clf.score(X_cand, y_cand))

        mean_accuracy = np.mean(correct_classification_list)

        title_string = (
            f"Decision boundary after {t_x} new samples\n"
            f"Utility: {utilities[0]:.4f} | "
            f"Budget: {sum(queried_indices) / (t_x + 1):.4f}\n"
            f"Prequential Evaluation Accuracy: {mean_accuracy:.4f}"
        )
        title = ax.text(
            x=0.5,
            y=1.05,
            s=title_string,
            size=plt.rcParams["axes.titlesize"],
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel('Number of Observed Samples')
        ax.set_ylabel('Feature')
        coll_new = list(ax.collections)
        coll_new.append(title)

        # Collect new artists (plot elements) to animate.
        artists.append(
            [x for x in coll_new if (x not in coll_old)] + data_lines
        )

# Create an animation from the collected frames.
ani = animation.ArtistAnimation(fig, artists, interval=500, blit=True)

# %%
# .. image:: ../../examples/stream_classification_legend.png

# %%
# .. rubric:: References:
# 
# The implementation of this strategy is based on :footcite:t:`liu2023active`.
#
# .. footbibliography::


