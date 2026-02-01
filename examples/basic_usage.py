"""
Basic Usage of ClusterTuner
===========================

This example demonstrates how to use ClusterTuner to find
optimal hyperparameters for clustering algorithms.
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from cluster_tuner import ClusterTuner

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
X = StandardScaler().fit_transform(X)

# %%
# Basic example with DBSCAN
# -------------------------
# Search over eps and min_samples parameters using silhouette score.

tuner = ClusterTuner(
    DBSCAN(),
    param_grid={"eps": [0.3, 0.5, 0.7], "min_samples": [5, 10]},
    scoring="silhouette",
)
tuner.fit(X)

print("DBSCAN Results:")
print(f"  Best parameters: {tuner.best_params_}")
print(f"  Best score: {tuner.best_score_:.3f}")
print(f"  Number of clusters: {len(np.unique(tuner.labels_[tuner.labels_ >= 0]))}")

# %%
# Multi-metric scoring
# --------------------
# Evaluate multiple metrics simultaneously.

tuner = ClusterTuner(
    KMeans(n_init="auto", random_state=42),
    param_grid={"n_clusters": [2, 3, 4, 5, 6]},
    scoring=["silhouette", "calinski_harabasz", "neg_davies_bouldin"],
    refit="silhouette",
)
tuner.fit(X)

print("\nKMeans Multi-metric Results:")
print(f"  Best parameters: {tuner.best_params_}")
print(f"  Best silhouette: {tuner.best_score_:.3f}")

# Access all metrics from results
results = tuner.results_
best_idx = tuner.best_index_
print(f"  Calinski-Harabasz at best: {results['test_calinski_harabasz'][best_idx]:.1f}")
print(f"  Davies-Bouldin at best: {-results['test_neg_davies_bouldin'][best_idx]:.3f}")

# %%
# Using supervised metrics
# ------------------------
# When ground truth labels are available, use supervised metrics.

tuner = ClusterTuner(
    KMeans(n_init="auto", random_state=42),
    param_grid={"n_clusters": [2, 3, 4, 5, 6]},
    scoring="adjusted_rand",
)
tuner.fit(X, y=y_true)

print("\nKMeans with Adjusted Rand Score:")
print(f"  Best parameters: {tuner.best_params_}")
print(f"  Best ARI: {tuner.best_score_:.3f}")

# %%
# Controlling noise and cluster size
# ----------------------------------
# Use max_noise and min_cluster_size to reject degenerate solutions.

tuner = ClusterTuner(
    DBSCAN(),
    param_grid={"eps": [0.1, 0.3, 0.5, 0.7, 1.0], "min_samples": [3, 5, 10]},
    scoring="silhouette",
    max_noise=0.1,  # Reject if more than 10% noise
    min_cluster_size=10,  # Reject if smallest cluster < 10 points
    error_score=np.nan,
)
tuner.fit(X)

print("\nDBSCAN with constraints:")
print(f"  Best parameters: {tuner.best_params_}")
print(f"  Best score: {tuner.best_score_:.3f}")

# Check how many parameter combinations were rejected
n_failed = np.isnan(tuner.results_["test_score"]).sum()
n_total = len(tuner.results_["test_score"])
print(f"  Rejected {n_failed}/{n_total} parameter combinations")
