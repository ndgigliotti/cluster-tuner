from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import metrics
from sklearn import preprocessing as pp
from sklearn.base import BaseEstimator
from sklearn.metrics._scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_consistent_length, check_is_fitted


def _passthrough_scorer(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None = None,
) -> float:
    """Call estimator.score directly (passthrough scorer)."""
    if y is None:
        return estimator.score(X)
    return estimator.score(X, y)


def _get_labels(estimator: BaseEstimator | Pipeline) -> NDArray[np.intp]:
    """Gets the cluster labels from an estimator or pipeline."""
    if isinstance(estimator, Pipeline):
        check_is_fitted(estimator._final_estimator, ["labels_"])
        labels = estimator._final_estimator.labels_
    else:
        check_is_fitted(estimator, ["labels_"])
        labels = estimator.labels_
    return np.array(labels)


def _noise_ratio(labels: ArrayLike, noise_label: int = -1) -> float:
    """Calculate the ratio of noise points in labels."""
    labels = np.asarray(labels)
    return float((labels == noise_label).mean())


def _smallest_clust_size(labels: ArrayLike, noise_label: int = -1) -> int:
    """Find the size of the smallest non-noise cluster."""
    labels = pp.LabelEncoder().fit_transform(labels[labels != noise_label])
    sizes = np.bincount(labels)
    if sizes.size == 0:
        smallest_clust_size = -1
    else:
        smallest_clust_size = int(sizes.min())
    return smallest_clust_size


def _remove_noise_cluster(
    *arrays: ArrayLike,
    labels: ArrayLike,
    noise_label: int = -1,
) -> tuple[NDArray[Any], ...]:
    """Removes the noise cluster found in `labels` (if any) from all `arrays`."""
    is_noise = labels == noise_label
    result: list[NDArray[Any]] = []
    for arr in arrays:
        result.append(arr[~is_noise].copy())
    check_consistent_length(*result)
    return tuple(result)


def _cached_call(cache: Any, method: Any, *args: Any, **kwargs: Any) -> None:
    """Dummy cached call for clustering (no caching needed)."""
    return None


class _LabelScorerSupervised(_BaseScorer):
    """Scorer for clustering metrics that require ground truth labels."""

    def _score(
        self,
        method_caller: Callable[..., Any],
        estimator: BaseEstimator,
        X: ArrayLike,
        labels_true: ArrayLike,
        **kwargs: Any,
    ) -> float:
        """Evaluate estimator labels relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Method caller for caching (ignored for clustering since we use labels_).

        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Does nothing, since estimator should already have `labels_`.
            Here for API compatibility.

        labels_true : array-like
            Ground truth target values for cluster labels.

        **kwargs : additional arguments
            Additional parameters (ignored).

        Returns
        -------
        score : float
            Score function applied to cluster labels.
        """
        labels = _get_labels(estimator)
        labels_true, labels = _remove_noise_cluster(labels_true, labels, labels=labels)
        return self._sign * self._score_func(labels_true, labels, **self._kwargs)

    def __call__(
        self,
        estimator: BaseEstimator,
        X: ArrayLike,
        y_true: ArrayLike | None = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate estimator labels relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Does nothing, since estimator should already have `labels_`.
            Here for API compatibility.

        y_true : array-like
            Ground truth target values for cluster labels.

        **kwargs : additional arguments
            Additional parameters passed to _score.

        Returns
        -------
        score : float
            Score function applied to cluster labels.
        """
        return self._score(
            partial(_cached_call, None),
            estimator,
            X,
            y_true,
            **kwargs,
        )


class _LabelScorerUnsupervised(_BaseScorer):
    """Scorer for clustering metrics that don't require ground truth."""

    def _score(
        self,
        method_caller: Callable[..., Any],
        estimator: BaseEstimator,
        X: ArrayLike,
        labels_true: ArrayLike | None = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate cluster labels on X.

        Parameters
        ----------
        method_caller : callable
            Method caller for caching (ignored for clustering since we use labels_).

        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Data that will be used to evaluate cluster labels.

        labels_true: array-like
            Does nothing. Here for API compatibility.

        **kwargs : additional arguments
            Additional parameters (ignored).

        Returns
        -------
        score : float
            Score function applied to cluster labels.
        """
        labels = _get_labels(estimator)
        if isinstance(estimator, Pipeline):
            X = estimator[:-1].transform(X)
        X, labels = _remove_noise_cluster(X, labels, labels=labels)
        return self._sign * self._score_func(X, labels, **self._kwargs)

    def __call__(
        self,
        estimator: BaseEstimator,
        X: ArrayLike,
        y_true: ArrayLike | None = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Data that will be used to evaluate cluster labels.

        y_true: array-like
            Does nothing. Here for API compatibility.

        **kwargs : additional arguments
            Additional parameters passed to _score.

        Returns
        -------
        score : float
            Score function applied cluster labels.
        """
        return self._score(
            partial(_cached_call, None),
            estimator,
            X,
            y_true,
            **kwargs,
        )


class _MultimetricScorer:
    """Scorer that wraps multiple scorers."""

    def __init__(
        self, *, scorers: dict[str, Callable[..., float]], raise_exc: bool = True
    ):
        self._scorers = scorers
        self._raise_exc = raise_exc

    def __call__(
        self, estimator: BaseEstimator, *args: Any, **kwargs: Any
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, scorer in self._scorers.items():
            try:
                score = scorer(estimator, *args, **kwargs)
                scores[name] = score
            except Exception:
                if self._raise_exc:
                    raise
                scores[name] = float("nan")
        return scores


def make_scorer(
    score_func: Callable[..., float],
    *,
    response_method: str = "predict",
    ground_truth: bool = True,
    greater_is_better: bool = True,
    **kwargs: Any,
) -> _LabelScorerSupervised | _LabelScorerUnsupervised:
    """Make a clustering scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in
    :class:`~cluster_opt.ClusterOptimizer`
    It takes a score function, such as :func:`~sklearn.metrics.silhouette_score`,
    :func:`~sklearn.metrics.mutual_info_score`, or
    :func:`~sklearn.metrics.adjusted_rand_index`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    response_method : str, default="predict"
        The method to call on the estimator. For clustering scorers, this
        parameter is stored but not used since we access labels_ directly.
        Provided for API compatibility with sklearn.

    ground_truth : bool, default=True
        Whether score_func uses ground truth labels.

    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    # response_method is stored in kwargs for API compatibility but not used
    # since clustering scorers access labels_ directly
    sign = 1 if greater_is_better else -1
    if ground_truth:
        cls = _LabelScorerSupervised
    else:
        cls = _LabelScorerUnsupervised
    return cls(score_func, sign, kwargs)


_davies_bouldin_scorer = make_scorer(
    metrics.davies_bouldin_score, greater_is_better=False, ground_truth=False
)

SCORERS = {
    "silhouette_score": make_scorer(metrics.silhouette_score, ground_truth=False),
    "silhouette_score_euclidean": make_scorer(
        metrics.silhouette_score, ground_truth=False
    ),
    "silhouette_score_cosine": make_scorer(
        metrics.silhouette_score, ground_truth=False, metric="cosine"
    ),
    # Preferred names (sklearn convention: neg_ prefix for lower-is-better)
    "neg_davies_bouldin_score": _davies_bouldin_scorer,
    # Deprecated aliases (kept for backwards compatibility)
    "davies_bouldin_score": _davies_bouldin_scorer,
    "calinski_harabasz_score": make_scorer(
        metrics.calinski_harabasz_score, ground_truth=False
    ),
    "mutual_info_score": make_scorer(metrics.mutual_info_score),
    "normalized_mutual_info_score": make_scorer(metrics.normalized_mutual_info_score),
    "adjusted_mutual_info_score": make_scorer(metrics.adjusted_mutual_info_score),
    "rand_score": make_scorer(metrics.rand_score),
    "adjusted_rand_score": make_scorer(metrics.adjusted_rand_score),
    "completeness_score": make_scorer(metrics.completeness_score),
    "fowlkes_mallows_score": make_scorer(metrics.fowlkes_mallows_score),
    "homogeneity_score": make_scorer(metrics.homogeneity_score),
    "v_measure_score": make_scorer(metrics.v_measure_score),
}
SCORERS.update({k.replace("_score", ""): v for k, v in SCORERS.items()})
SCORERS = MappingProxyType(SCORERS)


def get_scorer(
    scoring: str | Callable[..., float],
) -> Callable[..., float]:
    """Get a clustering scorer from string.

    Parameters
    ----------
    scoring : str or callable
        Scoring method as string. If callable it is returned as is.

    Returns
    -------
    scorer : callable
        The scorer.
    """
    if isinstance(scoring, str):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError(
                f"'{scoring}' is not a valid scoring value. "
                "Use sorted(cluster_tuner.scorer.SCORERS.keys())"
                "to get valid options."
            )
    else:
        scorer = scoring
    return scorer


def check_scoring(
    estimator: BaseEstimator,
    scoring: str
    | Callable[..., float]
    | Iterable[str]
    | dict[str, Callable[..., float]]
    | None = None,
    *,
    raise_exc: bool = True,
) -> Callable[..., float] | _MultimetricScorer:
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    scoring : str, callable, list, tuple, set, dict or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        For evaluating multiple metrics, a list/tuple/set of strings
        or a dict with names as keys and callables as values can be passed.

    raise_exc : bool, default=True
        Whether to raise an exception if scoring fails. If False,
        failed scores will be set to NaN.

    Returns
    -------
    scoring : callable or _MultimetricScorer
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``, or a _MultimetricScorer for
        multi-metric scoring.
    """
    if not hasattr(estimator, "fit"):
        raise TypeError(
            f"estimator should be an estimator implementing "
            f"'fit' method, {estimator!r} was passed"
        )
    if isinstance(scoring, str):
        return get_scorer(scoring)
    elif isinstance(scoring, list | tuple | set | dict):
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)
        return _MultimetricScorer(scorers=scorers, raise_exc=raise_exc)
    elif callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                f"scoring value {scoring!r} looks like it is a metric "
                f"function rather than a scorer. A scorer should "
                f"require an estimator as its first parameter. "
                f"Please use `make_scorer` to convert a metric to a scorer."
            )
        return get_scorer(scoring)
    elif scoring is None:
        if hasattr(estimator, "score"):
            return _passthrough_scorer
        else:
            raise TypeError(
                f"If no scoring is specified, the estimator passed should "
                f"have a 'score' method. The estimator {estimator!r} does not."
            )
    else:
        raise ValueError(
            f"scoring value should either be a callable, string or "
            f"None. {scoring!r} was passed"
        )


def _check_multimetric_scoring(
    estimator: BaseEstimator,
    scoring: list[str] | tuple[str, ...] | set[str] | dict[str, Callable[..., float]],
) -> dict[str, Callable[..., float]]:
    """Check the scoring parameter in cases when multiple metrics are allowed.

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.

    scoring : list, tuple or dict
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        See :ref:`multimetric_grid_search` for an example.

    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    """
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    if isinstance(scoring, list | tuple | set):
        err_msg = (
            "The list/tuple elements must be unique " "strings of predefined scorers. "
        )
        invalid = False
        try:
            keys = set(scoring)
        except TypeError:
            invalid = True
        if invalid:
            raise ValueError(err_msg)

        if len(keys) != len(scoring):
            raise ValueError(
                f"{err_msg} Duplicate elements were found in"
                f" the given list. {scoring!r}"
            )
        elif len(keys) > 0:
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    raise ValueError(
                        f"{err_msg} Non-string types were found "
                        f"in the given list. Got {scoring!r}"
                    )
            scorers = {
                scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring
            }
        else:
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(
                "Non-string types were found in the keys of "
                f"the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        scorers = {
            key: check_scoring(estimator, scoring=scorer)
            for key, scorer in scoring.items()
        }
    else:
        raise ValueError(err_msg_generic)
    return scorers
