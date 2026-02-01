# cluster-tuner

A GridSearchCV-like hyperparameter tuner for clustering algorithms.

## Installation

```bash
pip install cluster-tuner
```

## Purpose

This project provides a simple, Scikit-Learn-compatible, hyperparameter tuning tool for clustering. It's intended for situations where predicting clusters for new data points is a low priority. Many clustering algorithms in Scikit-Learn are **transductive**, meaning that they are not designed to be applied to new observations. Even if using an **inductive** clustering algorithm like K-Means, you might not have any desire to predict clusters for new observations. Or, even if you do have such a desire, prediction might be a lower priority than finding the best clusters in the data.

Since Scikit-Learn's `GridSearchCV` uses cross-validation, and is designed to optimize inductive machine learning models, an alternative tool is necessary.

## `ClusterTuner`

The `ClusterTuner` class is a hyperparameter search tool for tuning clustering algorithms. It simply fits one model per hyperparameter combination and selects the best. It's a spin-off of `GridSearchCV`, and the implementation is derived from Scikit-Learn. The only difference is that it doesn't use cross-validation and is designed to work with special clustering scorers. It's not always necessary to provide a target variable, since clustering metrics such as silhouette, Calinski-Harabasz, and Davies-Bouldin are designed for unsupervised clustering.

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
```

### Key Parameters

- **`max_noise`** (default=0.1): Maximum allowed ratio of noise points (label=-1). Fits exceeding this threshold receive `error_score` instead.
- **`min_cluster_size`** (default=3): Minimum allowed size for the smallest cluster. Fits with smaller clusters receive `error_score` instead.

These parameters help filter out degenerate clustering solutions where most points are noise or clusters are too small to be meaningful.

### Multi-Metric Scoring

You can evaluate multiple metrics simultaneously using a list, tuple, or dict:

```python
tuner = ClusterTuner(
    DBSCAN(),
    param_grid={'eps': [0.3, 0.5, 0.7]},
    scoring=['silhouette', 'calinski_harabasz', 'neg_davies_bouldin'],
    refit='silhouette',  # Required: which metric to use for selecting best
)
tuner.fit(X)

# Results include all metrics (prefixed with 'test_')
print(tuner.results_['test_silhouette'])
print(tuner.results_['test_calinski_harabasz'])
print(tuner.results_['test_neg_davies_bouldin'])
```

## Transductive Clustering Scorers

You can use `ClusterTuner` by passing the string name of a Scikit-Learn clustering metric, e.g. 'silhouette', 'calinski_harabasz', or 'rand_score' (the '_score' suffix is optional). You can also create a special scorer for transductive clustering using `scorer.make_scorer` on any score function with the signature `score_func(labels_true, labels_fit)` or `score_func(X, labels_fit)`.


### Recognized Scorer Names

Note that the `_score` suffix is always optional (e.g., `'silhouette'` and `'silhouette_score'` both work).

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

#### Naming Convention

Following sklearn's convention, metrics where **lower is better** use a `neg_` prefix. The score is negated internally so that higher values always indicate better clustering. This applies to:
- `'neg_davies_bouldin'` — Davies-Bouldin index (lower raw values = better separation)

The old name `'davies_bouldin'` still works for backwards compatibility but `'neg_davies_bouldin'` is preferred.

## Caveats

### Comparing Clustering Algorithms

It's important to consider your dataset and goals before comparing clustering algorithms in a grid search. Just because one algorithm gets a higher score than another does not necessarily make it a better choice. Different clustering algorithms have [different benefits, drawbacks, and use cases.](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods)

## Credits

Most of the credit goes to the developers of Scikit-Learn for the engineering behind the search estimators. It's not very hard to spam a bunch of models with different hyperparameters, but it's hard to do it in a robust way with a friendly interface and wide compatibility.
