from __future__ import annotations

import numbers
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import partial
from traceback import format_exc
from typing import Any, Literal

import joblib
import numpy as np
from joblib import delayed
from numpy.ma import MaskedArray
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils._tags import get_tags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from cluster_tuner.scorer import (
    _check_multimetric_scoring,
    _get_labels,
    _MultimetricScorer,
    _noise_ratio,
    _smallest_clust_size,
    check_scoring,
)

# Type aliases
ParamGrid = dict[str, Sequence[Any]] | list[dict[str, Sequence[Any]]]
ScorerType = (
    Callable[..., float]
    | dict[str, Callable[..., float]]
    | list[str]
    | tuple[str, ...]
    | str
    | None
)
ErrorScoreType = Literal["raise"] | float


def _check_fit_params(
    X: ArrayLike,
    fit_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and prepare fit parameters.

    Parameters
    ----------
    X : array-like
        The input data (used to validate sample-aligned parameters).
    fit_params : dict or None
        Parameters to pass to fit.

    Returns
    -------
    fit_params : dict
        Validated fit parameters.
    """
    if fit_params is None:
        return {}
    return fit_params


def _check_param_grid(param_grid: dict | list[dict]) -> None:
    """Validate parameter grid format.

    Parameters
    ----------
    param_grid : dict or list of dicts
        Each dict maps parameter names to lists of values.

    Raises
    ------
    ValueError
        If param_grid is empty or has invalid structure.
    TypeError
        If parameter values are not iterable (excluding strings).
    """
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError(
                    f"Parameter array for {name!r} should be one-dimensional, "
                    f"got array with shape {v.shape}"
                )
            if isinstance(v, str) or not hasattr(v, "__iter__"):
                raise TypeError(
                    f"Parameter grid for parameter {name!r} needs to be a list "
                    f"or a numpy array, but got {type(v).__name__!r} instead. "
                    f"Single values should be wrapped in a list."
                )
            if len(v) == 0:
                raise ValueError(
                    f"Parameter grid for parameter {name!r} is empty. "
                    f"Provide at least one value to search over."
                )


def _aggregate_score_dicts(
    scores: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Aggregate a list of score dicts into a dict of arrays.

    Transforms ``[{'a': 0.1, 'b': 0.2}, {'a': 0.3, 'b': 0.4}]``
    into ``{'a': array([0.1, 0.3]), 'b': array([0.2, 0.4])}``.

    Parameters
    ----------
    scores : list of dict
        List of score dictionaries, each mapping scorer names to float values.

    Returns
    -------
    aggregated : dict of ndarray
        Dictionary mapping scorer names to arrays of scores.
    """
    return {key: np.asarray([score[key] for score in scores]) for key in scores[0]}


def _insert_error_scores(
    results: list[dict[str, Any]],
    error_score: float | int,
) -> None:
    """Insert error_score into results for failed fits (in-place).

    Parameters
    ----------
    results : list of dict
        List of result dictionaries from fitting. Each dict may contain
        a ``fit_failed`` key and a ``scores`` key.
    error_score : float or int
        The error score value to insert for failed fits.
    """
    for result in results:
        if result.get("fit_failed", False):
            scores = result.get("scores")
            if isinstance(scores, dict):
                result["scores"] = {name: error_score for name in scores}
            else:
                result["scores"] = error_score


def _normalize_score_results(
    scores: list[dict[str, float] | float],
    scalar_score_key: str = "score",
) -> dict[str, np.ndarray]:
    """Normalize score results into dict of arrays.

    If scores are dicts (multimetric), aggregate them.
    If scores are scalars, wrap in dict with ``scalar_score_key``.

    Parameters
    ----------
    scores : list of dict or list of float
        List of scores. Each element is either a dict mapping scorer names
        to floats (multimetric) or a single float (single metric).
    scalar_score_key : str, default="score"
        Key to use when wrapping scalar scores in a dict.

    Returns
    -------
    normalized : dict of ndarray
        Dictionary mapping scorer names to arrays of scores.
    """
    if isinstance(scores[0], dict):
        return _aggregate_score_dicts(scores)
    else:
        return {scalar_score_key: np.asarray(scores)}


def _estimator_has(attr: str) -> Callable[[BaseSearch], bool]:
    """Check if the fitted estimator has a specific attribute.

    Used with ``available_if`` decorator to conditionally expose methods.

    Parameters
    ----------
    attr : str
        The attribute name to check for.

    Returns
    -------
    check : callable
        A function that takes a ``BaseSearch`` instance and returns True
        if the estimator has the specified attribute.
    """

    def check(self: BaseSearch) -> bool:
        # Check best_estimator_ first (fitted), then estimator (unfitted)
        if hasattr(self, "best_estimator_"):
            return hasattr(self.best_estimator_, attr)
        return hasattr(self.estimator, attr)

    return check


def _score(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    scorer: Callable[..., float] | dict[str, Callable[..., float]],
    error_score: ErrorScoreType = "raise",
    max_noise: float = 0.1,
    min_cluster_size: int = 3,
) -> float | dict[str, float]:
    """Compute the score(s) of an estimator on a given data set.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator with ``labels_`` attribute.
    X : array-like of shape (n_samples, n_features)
        The data to score.
    y : array-like of shape (n_samples,) or None
        Ground truth labels for supervised metrics, or None for unsupervised.
    scorer : callable or dict of callable
        A single scorer or dict mapping scorer names to scorer callables.
    error_score : 'raise' or numeric, default='raise'
        Value to assign if scoring fails. If 'raise', errors are raised.
    max_noise : float, default=0.1
        Maximum allowed noise ratio. If exceeded, ``error_score`` is used.
    min_cluster_size : int, default=3
        Minimum allowed cluster size. If smallest cluster is smaller,
        ``error_score`` is used.

    Returns
    -------
    scores : float or dict of float
        If ``scorer`` is a dict, returns a dict mapping scorer names to scores.
        Otherwise returns a single float score.
    """
    if isinstance(scorer, dict):
        scorers = scorer
    else:
        scorers = {0: scorer}

    noise_ratio = _noise_ratio(_get_labels(estimator))
    smallest_clust_size = _smallest_clust_size(_get_labels(estimator))
    if noise_ratio > max_noise:
        if error_score == "raise":
            raise RuntimeError(f"Noise ratio {noise_ratio:.2f} > {max_noise:.2f}.")
        else:
            scores = {name: error_score for name in scorers}
            warnings.warn(
                f"Noise ratio {noise_ratio:.2f} > {max_noise:.2f}. "
                f"The score for these parameters will be set to {error_score}.",
                UserWarning,
            )
    elif smallest_clust_size < min_cluster_size:
        if error_score == "raise":
            raise RuntimeError(
                f"Smallest cluster too small: "
                f"{smallest_clust_size:.0f} < {min_cluster_size:.0f}."
            )
        else:
            scores = {name: error_score for name in scorers}
            warnings.warn(
                f"Smallest cluster too small: "
                f"{smallest_clust_size:.0f} < {min_cluster_size:.0f}. "
                f"The score for these parameters will be set to {error_score}.",
                UserWarning,
            )
    else:
        try:
            if y is None:
                scores = {name: scorers[name](estimator, X) for name in scorers}
            else:
                scores = {name: scorers[name](estimator, X, y) for name in scorers}
        except Exception:
            if error_score == "raise":
                raise
            else:
                scores = {name: error_score for name in scorers}
                warnings.warn(
                    f"Scoring failed. The score for these parameters"
                    f" will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )

    for name, score in scores.items():
        if hasattr(score, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                score = score.item()
        if not isinstance(score, numbers.Number):
            raise ValueError(
                f"scoring must return a number, got {score} ({type(score)}) "
                f"instead. (scorer={name})"
            )
        scores[name] = score
    if len(scores) == 1:
        scores = list(scores.values())[0]
    return scores


def _fit_and_score(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    scorer: Callable[..., float] | dict[str, Callable[..., float]],
    verbose: int,
    parameters: dict[str, Any] | None,
    fit_params: dict[str, Any] | None,
    return_parameters: bool = False,
    return_n_samples: bool = False,
    return_times: bool = False,
    return_estimator: bool = False,
    return_noise_ratios: bool = False,
    return_smallest_clust_sizes: bool = False,
    candidate_progress: tuple[int, int] | None = None,
    error_score: ErrorScoreType = np.nan,
    max_noise: float = 0.1,
    min_cluster_size: int = 3,
) -> dict[str, Any]:
    """Fit estimator and compute scores for clustering.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable (ground truth labels for supervised metrics);
        None for unsupervised learning.

    scorer : callable or dict mapping scorer name to callable
        If it is a single callable, the return value for ``scores`` is a single
        float. For a dict, it should map the scorer name to the scorer callable.
        The callable should have signature ``scorer(estimator, X, y)``.

    verbose : int
        The verbosity level.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_parameters : bool, default=False
        Return parameters that have been used for the estimator.

    return_n_samples : bool, default=False
        Whether to return the number of samples.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    return_noise_ratios : bool, default=False
        Whether to return the noise ratio of the clustering.

    return_smallest_clust_sizes : bool, default=False
        Whether to return the size of the smallest cluster.

    candidate_progress : tuple of int, default=None
        A tuple of format (current_candidate_id, total_number_of_candidates).

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    max_noise : float, default=0.1
        Maximum allowed noise ratio. If exceeded, error_score is used.

    min_cluster_size : int, default=3
        Minimum allowed cluster size. If smallest cluster is smaller,
        error_score is used.

    Returns
    -------
    result : dict with the following attributes
        scores : dict of scorer name -> float or float
            Score(s) on the data.
        n_samples : int
            Number of samples (if return_n_samples=True).
        fit_time : float
            Time spent for fitting in seconds (if return_times=True).
        score_time : float
            Time spent for scoring in seconds (if return_times=True).
        parameters : dict or None
            The parameters that have been evaluated (if return_parameters=True).
        estimator : estimator object
            The fitted estimator (if return_estimator=True).
        noise_ratio : float
            Ratio of noise points (if return_noise_ratios=True).
        smallest_clust_size : int
            Size of smallest cluster (if return_smallest_clust_sizes=True).
        fit_failed : bool
            Whether the estimator failed to fit.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if candidate_progress and verbose > 9:
            progress_msg += f"{candidate_progress[0]+1}/" f"{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CAND {progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    result = {}
    try:
        if y is None:
            estimator.fit(X, **fit_params)
        else:
            estimator.fit(X, y, **fit_params)
    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        noise_ratio = np.nan
        smallest_clust_size = -1
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                scores = {name: error_score for name in scorer}
            else:
                scores = error_score
            warnings.warn(
                f"Estimator fit failed. The score for these parameters "
                f"will be set to {error_score}. Details:\n{format_exc()}",
                FitFailedWarning,
            )
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        scores = _score(
            estimator, X, y, scorer, error_score, max_noise, min_cluster_size
        )
        noise_ratio = _noise_ratio(_get_labels(estimator))
        smallest_clust_size = _smallest_clust_size(_get_labels(estimator))
        score_time = time.time() - start_time - fit_time

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CAND {progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and isinstance(scores, dict):
            for scorer_name in sorted(scores):
                result_msg += f" {scorer_name}: ("
                result_msg += f"score={scores[scorer_name]:.3f})"
        result_msg += f" noise_ratio={noise_ratio:.2f}"
        result_msg += f" smallest_clust_size={min_cluster_size:.0f}"
        result_msg += f" total_time={joblib.logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["scores"] = scores
    if return_n_samples:
        if isinstance(estimator, Pipeline):
            result["n_samples"] = estimator[:-1].transform(X).shape[0]
        else:
            result["n_samples"] = X.shape[0]
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    if return_noise_ratios:
        result["noise_ratio"] = noise_ratio
    if return_smallest_clust_sizes:
        result["smallest_clust_size"] = smallest_clust_size
    return result


class BaseSearch(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for hyperparameter search.

    This class provides the common interface and functionality for
    hyperparameter search estimators. It should not be instantiated directly;
    use derived classes like :class:`ClusterTuner` instead.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "scoring": [callable, list, tuple, dict, str, None],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str, callable],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
    }

    @abstractmethod
    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        scoring: ScorerType = None,
        n_jobs: int | None = None,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
        verbose: int = 0,
        pre_dispatch: int | str = "2*n_jobs",
        error_score: ErrorScoreType = np.nan,
        max_noise: float = 0.1,
        min_cluster_size: int = 3,
    ) -> None:
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.max_noise = max_noise
        self.min_cluster_size = min_cluster_size

    def __sklearn_tags__(self):
        """Return sklearn tags for the estimator.

        Returns
        -------
        tags : Tags
            Sklearn tags object with estimator type and input tags inherited
            from the wrapped estimator.
        """
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.estimator_type = sub_estimator_tags.estimator_type
        tags.input_tags.pairwise = sub_estimator_tags.input_tags.pairwise
        return tags

    def score(self, X: ArrayLike, y: ArrayLike | None = None) -> float:
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            The score of the best estimator on the given data.
        """
        self._check_is_fitted("score")
        if self.scorer_ is None:
            raise ValueError(
                f"No score function explicitly defined, "
                f"and the estimator doesn't provide one {self.best_estimator_}"
            )
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y)

        # callable
        score = self.scorer_(self.best_estimator_, X, y)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X: ArrayLike) -> NDArray[Any]:
        """Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Scores for each sample.
        """
        self._check_is_fitted("score_samples")
        return self.best_estimator_.score_samples(X)

    def _check_is_fitted(self, method_name: str) -> None:
        """Check if the search estimator is fitted.

        Parameters
        ----------
        method_name : str
            Name of the method requiring the estimator to be fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted or ``refit=False``.
        """
        if not self.refit:
            raise NotFittedError(
                f"This {type(self).__name__} instance was initialized "
                f"with refit=False. {method_name} is "
                f"available only after refitting on the best "
                f"parameters. You can refit an estimator "
                f"manually using the ``best_params_`` attribute"
            )
        else:
            check_is_fitted(self)

    @available_if(_estimator_has("predict"))
    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels or values.
        """
        self._check_is_fitted("predict")
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X: ArrayLike) -> NDArray[Any]:
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        self._check_is_fitted("predict_proba")
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: ArrayLike) -> NDArray[Any]:
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Predicted class log-probabilities.
        """
        self._check_is_fitted("predict_log_proba")
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X: ArrayLike) -> NDArray[Any]:
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values.
        """
        self._check_is_fitted("decision_function")
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_new)
            Transformed data.
        """
        self._check_is_fitted("transform")
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, X: ArrayLike) -> NDArray[Any]:
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in the original feature space.
        """
        self._check_is_fitted("inverse_transform")
        return self.best_estimator_.inverse_transform(X)

    @property
    def n_features_in_(self) -> int:
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the search estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                f"{self.__class__.__name__} object has no n_features_in_ attribute."
            ) from nfe

        return self.best_estimator_.n_features_in_

    @property
    def classes_(self) -> NDArray[Any]:
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    @property
    def cv_results_(self) -> dict[str, Any]:
        """Alias for ``results_`` for GridSearchCV API compatibility.

        Returns
        -------
        cv_results_ : dict
            Same as ``results_``. Provided for compatibility with code
            that expects the GridSearchCV interface.
        """
        return self.results_

    def _run_search(
        self,
        evaluate_candidates: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    ) -> None:
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        This method, implemented in sub-classes, makes it possible to
        customize the the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.
        For example, Successive Halving is implemented by calling
        `evaluate_candidates` multiples times (once per iteration of the SH
        process), each time passing a different set of candidates with `X`
        and `y` of increasing sizes.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  SucessiveHaling estimators). By default, the original `cv`
                  parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`

            It returns a dict of all results so far, formatted like
            ``cv_results_``.

            Important note (relevant whether the default cv is used or not):
            in randomized splitters, and unless the random_state parameter of
            cv was set to an int, calling cv.split() multiple times will
            yield different splits. Since cv.split() is called in
            evaluate_candidates, this means that candidates will be evaluated
            on different splits each time evaluate_candidates is called. This
            might be a methodological issue depending on the search strategy
            that you're implementing. To prevent randomized splitters from
            being used, you may use _split._yields_constant_splits()

        Examples
        --------

        ::

            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        raise NotImplementedError("_run_search not implemented.")

    def _check_refit_for_multimetric(
        self,
        scores: dict[str, Callable[..., float]],
    ) -> None:
        """Check that ``refit`` is compatible with ``scores``.

        Parameters
        ----------
        scores : dict
            Dictionary mapping scorer names to scorer callables.

        Raises
        ------
        ValueError
            If ``refit`` is not False, not a valid scorer name, and not callable.
        """
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if (
            self.refit is not False
            and not valid_refit_dict
            and not callable(self.refit)
        ):
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Select the best index based on refit strategy."""
        if callable(refit):
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")
            if best_index < 0 or best_index >= len(results["params"]):
                raise IndexError("best_index_ index out of range")
        else:
            best_index = results[f"rank_{refit_metric}"].argmin()
        return best_index

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        **fit_params: Any,
    ) -> BaseSearch:
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        # estimator = self.estimator
        refit_metric = "test_score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = f"test_{self.refit}"

        # Handle _MultimetricScorer
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        # X, y = indexable(X, y)
        fit_params = _check_fit_params(X, fit_params)

        base_estimator = clone(self.estimator)

        fit_and_score = delayed(
            partial(
                _fit_and_score,
                scorer=scorers,
                fit_params=fit_params,
                return_n_samples=True,
                return_times=True,
                return_noise_ratios=True,
                return_smallest_clust_sizes=True,
                return_parameters=False,
                error_score=self.error_score,
                max_noise=self.max_noise,
                min_cluster_size=self.min_cluster_size,
                verbose=self.verbose,
            )
        )

        results = {}
        with joblib.Parallel(
            n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch
        ) as workers:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, more_results=None):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(f"Fitting {n_candidates} candidates.")

                out = workers(
                    fit_and_score(
                        clone(base_estimator),
                        X,
                        y,
                        parameters=parameters,
                        candidate_progress=(cand_idx, n_candidates),
                    )
                    for cand_idx, parameters in enumerate(candidate_params)
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. Were there no candidates?"
                    )
                elif len(out) != n_candidates:
                    raise ValueError("Received fewer results than candidates.")

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_score = all_out[0]["scores"]
            self.multimetric_ = isinstance(first_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_score)
                refit_metric = f"test_{self.refit}"

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "test_score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                self.best_score_ = results[refit_metric][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        self.n_splits_ = 1
        self.results_ = results

        return self

    def _format_results(
        self,
        candidate_params: list[dict[str, Any]],
        out: list[dict[str, Any]],
        more_results: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)

        results = dict(more_results or {})

        def _store(key_name: str, array: ArrayLike, rank: bool = False) -> None:
            """A small helper to store the scores/times to the results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64)

            results[key_name] = array

            if np.any(~np.isfinite(array)):
                warnings.warn(
                    f"One or more of the {key_name} scores " f"are non-finite: {array}",
                    category=UserWarning,
                )

            if rank:
                results[f"rank_{key_name}"] = np.asarray(
                    rankdata(-array, method="min"), dtype=np.int32
                )

        _store("noise_ratio", out["noise_ratio"])
        _store("smallest_clust_size", out["smallest_clust_size"])
        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `f"param_{name}"` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results[f"param_{name}"][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        scores_dict = _normalize_score_results(out["scores"])

        for scorer_name in scores_dict:
            _store(
                f"test_{scorer_name}",
                scores_dict[scorer_name],
                rank=True,
            )

        return results


class ClusterTuner(BaseSearch):
    """Exhaustive search over specified parameter values for a clustering estimator.

    ClusterTuner implements a `fit` and a `score` method. It attains the
    `labels_` attribute after optimizing hyperparameters if `refit=True`.
    It also implements "predict", "transform" and "inverse_transform"
    if they are implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by a simple grid-search over a parameter grid. There is no cross-validation;
    one fit is performed on the full data per entry in the grid.

    Parameters
    ----------
    estimator : clustering estimator.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, default=None
        A single str or a callable to evaluate the fit on the data.

        NOTE that when using custom scorers, each scorer should return a single
        value. Consider using `scorer.make_scorer` on a function with the signature
        score_func(labels_true, labels_fit) or score_func(X, labels_fit).

        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a `joblib.parallel_backend` context.
        ``-1`` means using all processors.

    pre_dispatch : int, or str, default=n_jobs
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``ClusterTuner`` instance.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Attributes
    ----------
    results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score on the full data. Not available
        if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Score of the best_estimator on the full dataset.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results.

    best_index_ : int
        The index (of the ``results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used to evaluate clustering quality and choose
        the best parameters.

    n_splits_ : int
        Always 1. Provided for API compatibility with GridSearchCV.
        No cross-validation is performed; each fit uses the full dataset.

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    Notes
    -----
    The parameters selected are those that maximize the clustering score
    on the full dataset.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    scorer.make_scorer : Make a scorer from a performance metric.

    """

    _required_parameters: list[str] = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: ParamGrid,
        *,
        scoring: ScorerType = None,
        n_jobs: int | None = None,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
        verbose: int = 0,
        pre_dispatch: int | str = "2*n_jobs",
        error_score: ErrorScoreType = np.nan,
        max_noise: float = 0.1,
        min_cluster_size: int = 3,
    ) -> None:
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )
        self.param_grid = param_grid
        _check_param_grid(param_grid)
        self.max_noise = max_noise
        self.min_cluster_size = min_cluster_size

    def _run_search(
        self,
        evaluate_candidates: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    ) -> None:
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    @property
    def labels_(self) -> NDArray[np.intp]:
        check_is_fitted(self, "best_estimator_")
        return _get_labels(self.best_estimator_)
