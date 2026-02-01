# cluster-tuner

[![CI](https://github.com/ndgigliotti/cluster-tuner/actions/workflows/ci.yml/badge.svg)](https://github.com/ndgigliotti/cluster-tuner/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/cluster-tuner.svg)](https://pypi.org/project/cluster-tuner/)
[![Codecov](https://codecov.io/gh/ndgigliotti/cluster-tuner/branch/main/graph/badge.svg)](https://codecov.io/gh/ndgigliotti/cluster-tuner)

A GridSearchCV-like hyperparameter tuner for clustering algorithms.

## Installation

```bash
pip install cluster-tuner
```

**Requirements:** Python >= 3.10, scikit-learn >= 1.6

## Purpose

This project provides a simple, scikit-learn-compatible hyperparameter tuning tool for clustering. It's intended for situations where predicting clusters for new data points is a low priority. Many clustering algorithms in scikit-learn are **transductive**, meaning they are not designed to be applied to new observations. Even when using an **inductive** algorithm like KMeans, you might not need to predict clusters for new data—or prediction might be a lower priority than finding the best clusters.

Since scikit-learn's `GridSearchCV` uses cross-validation and is designed for inductive models, an alternative tool is necessary.

## `ClusterTuner`

The `ClusterTuner` class is a hyperparameter search tool for clustering algorithms. It fits one model per hyperparameter combination and selects the best. The implementation is derived from scikit-learn's `GridSearchCV`, but without cross-validation. It works with clustering-specific scorers and doesn't always require a target variable, since metrics like silhouette, Calinski-Harabasz, and Davies-Bouldin are designed for unsupervised evaluation.

The interface is largely the same as `GridSearchCV`. Results are stored in the `results_` attribute (`cv_results_` also works as an alias for compatibility).

### Basic Usage

```python
from sklearn.cluster import DBSCAN
from cluster_tuner import ClusterTuner

tuner = ClusterTuner(
    DBSCAN(),
    param_grid={'eps': [0.3, 0.5, 0.7], 'min_samples': [5, 10]},
    scoring='silhouette',
)
tuner.fit(X)

print(tuner.best_params_)
print(tuner.best_score_)
labels = tuner.labels_

# Access detailed results (single-metric uses 'test_score')
print(tuner.results_['test_score'])
```

### Key Parameters

- **`scoring`**: Metric name (string), callable, or list/dict for multi-metric evaluation.
- **`refit`** (default=True): Whether to refit the best estimator on the full dataset. For multi-metric, must be a string specifying which metric to use.
- **`max_noise`** (default=0.1): Maximum allowed ratio of noise points (label=-1). Fits exceeding this threshold receive `error_score`.
- **`min_cluster_size`** (default=3): Minimum allowed size for the smallest cluster. Fits with smaller clusters receive `error_score`.
- **`error_score`** (default=np.nan): Value to assign when a fit fails or violates constraints. Use `'raise'` to raise exceptions instead.
- **`n_jobs`**: Number of parallel jobs (-1 for all CPUs).

### Multi-Metric Scoring

Evaluate multiple metrics simultaneously using a list, tuple, or dict:

```python
tuner = ClusterTuner(
    DBSCAN(),
    param_grid={'eps': [0.3, 0.5, 0.7]},
    scoring=['silhouette', 'calinski_harabasz', 'neg_davies_bouldin'],
    refit='silhouette',  # Required: which metric to use for selecting best
)
tuner.fit(X)

# Results use 'test_' prefix for each metric
print(tuner.results_['test_silhouette'])
print(tuner.results_['test_calinski_harabasz'])
print(tuner.results_['test_neg_davies_bouldin'])
```

### Supervised Scoring

When ground truth labels are available, use supervised metrics:

```python
from sklearn.cluster import KMeans

tuner = ClusterTuner(
    KMeans(n_init='auto'),
    param_grid={'n_clusters': [2, 3, 4, 5]},
    scoring='adjusted_rand',
)
tuner.fit(X, y=y_true)  # Pass ground truth labels

print(tuner.best_score_)  # Adjusted Rand Index
```

### Pipeline Support

`ClusterTuner` works with scikit-learn pipelines:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    KMeans(n_init='auto'),
)

tuner = ClusterTuner(
    pipe,
    param_grid={'kmeans__n_clusters': [2, 3, 4, 5]},
    scoring='silhouette',
)
tuner.fit(X)
```

## Scorers

You can use `ClusterTuner` by passing the string name of a clustering metric, e.g., `'silhouette'`, `'calinski_harabasz'`, or `'adjusted_rand'` (the `_score` suffix is optional).

### Recognized Scorer Names

**Unsupervised metrics** (no ground truth required):
- `'silhouette'` / `'silhouette_score'`
- `'silhouette_euclidean'` / `'silhouette_score_euclidean'`
- `'silhouette_cosine'` / `'silhouette_score_cosine'`
- `'neg_davies_bouldin'` / `'neg_davies_bouldin_score'`
- `'calinski_harabasz'` / `'calinski_harabasz_score'`

**Supervised metrics** (require ground truth labels `y`):
- `'mutual_info'` / `'mutual_info_score'`
- `'normalized_mutual_info'` / `'normalized_mutual_info_score'`
- `'adjusted_mutual_info'` / `'adjusted_mutual_info_score'`
- `'rand'` / `'rand_score'`
- `'adjusted_rand'` / `'adjusted_rand_score'`
- `'completeness'` / `'completeness_score'`
- `'fowlkes_mallows'` / `'fowlkes_mallows_score'`
- `'homogeneity'` / `'homogeneity_score'`
- `'v_measure'` / `'v_measure_score'`

### Naming Convention

Following sklearn's convention, metrics where **lower is better** use a `neg_` prefix. The score is negated internally so that higher values always indicate better clustering:
- `'neg_davies_bouldin'` — Davies-Bouldin index (lower raw values = better separation)

### Custom Scorers

Create custom scorers using `make_scorer`:

```python
from cluster_tuner import make_scorer

# Unsupervised scorer: score_func(X, labels)
def my_metric(X, labels):
    return some_score

scorer = make_scorer(my_metric, ground_truth=False)

# Supervised scorer: score_func(y_true, labels)
def my_supervised_metric(y_true, labels):
    return some_score

scorer = make_scorer(my_supervised_metric, ground_truth=True)

tuner = ClusterTuner(estimator, param_grid, scoring=scorer)
```

## Caveats

### Comparing Clustering Algorithms

Consider your dataset and goals before comparing clustering algorithms. A higher score doesn't necessarily mean a better choice—different algorithms have [different benefits, drawbacks, and use cases](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods).

## Credits

Most of the credit goes to the scikit-learn developers for the engineering behind the search estimators.
