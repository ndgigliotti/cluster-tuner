"""Comprehensive test suite for cluster-optimizer."""

import numpy as np
import pandas as pd
import pytest
from sklearn import cluster, datasets, decomposition
from sklearn import preprocessing as prep
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from cluster_tuner import ClusterTuner, make_scorer, SCORERS
from cluster_tuner.scorer import (
    _get_labels,
    _noise_ratio,
    _smallest_clust_size,
    _remove_noise_cluster,
    _LabelScorerSupervised,
    _LabelScorerUnsupervised,
    get_scorer,
    check_scoring,
    check_multimetric_scoring,
    _passthrough_scorer,
)
from cluster_tuner.search import (
    _check_fit_params,
    _check_param_grid,
    _aggregate_score_dicts,
    _insert_error_scores,
    _normalize_score_results,
    _estimator_has,
    _score,
    _fit_and_score,
)


# =============================================================================
# Test fixtures and helper classes
# =============================================================================


class ErrorClusterer(cluster.DBSCAN):
    """Clusterer that always raises an error on fit."""

    def fit(self, X, y=None):
        raise RuntimeError("This is a drill.")


class AllNoiseClusterer(ClusterMixin, BaseEstimator):
    """Clusterer that assigns all points to noise."""

    def __init__(self, dummy=None):
        self.dummy = dummy

    def fit(self, X, y=None):
        self.labels_ = np.full(len(X), -1)
        return self


class SingleClusterClusterer(ClusterMixin, BaseEstimator):
    """Clusterer that assigns all points to one cluster."""

    def fit(self, X, y=None):
        self.labels_ = np.zeros(len(X), dtype=int)
        return self


class TinyClusterClusterer(ClusterMixin, BaseEstimator):
    """Clusterer that creates one tiny cluster (size 1) and one larger cluster."""

    def fit(self, X, y=None):
        self.labels_ = np.zeros(len(X), dtype=int)
        self.labels_[0] = 1  # One point in cluster 1
        return self


class HighNoiseClusterer(ClusterMixin, BaseEstimator):
    """Clusterer that assigns most points to noise."""

    def __init__(self, noise_ratio=0.5):
        self.noise_ratio = noise_ratio

    def fit(self, X, y=None):
        n = len(X)
        n_noise = int(n * self.noise_ratio)
        self.labels_ = np.zeros(n, dtype=int)
        self.labels_[:n_noise] = -1
        return self


class PredictableClusterer(ClusterMixin, BaseEstimator):
    """Clusterer with predict and transform methods."""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.labels_ = np.arange(len(X)) % self.n_clusters
        self._X = X
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def transform(self, X):
        return X

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


class ScorableClusterer(ClusterMixin, BaseEstimator):
    """Clusterer with a score method."""

    def fit(self, X, y=None):
        self.labels_ = np.zeros(len(X), dtype=int)
        return self

    def score(self, X, y=None):
        return 0.8


@pytest.fixture
def iris_data():
    """Load iris dataset."""
    return datasets.load_iris(return_X_y=True)


@pytest.fixture
def blob_data():
    """Generate blob data."""
    return datasets.make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)


@pytest.fixture
def fitted_dbscan(iris_data):
    """Return a fitted DBSCAN estimator."""
    X, _ = iris_data
    est = cluster.DBSCAN(eps=0.5)
    est.fit(X)
    return est


@pytest.fixture
def fitted_kmeans(iris_data):
    """Return a fitted KMeans estimator."""
    X, _ = iris_data
    est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
    est.fit(X)
    return est


# =============================================================================
# scorer.py: Helper function tests
# =============================================================================


class TestNoiseRatio:
    """Tests for _noise_ratio function."""

    def test_no_noise(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        assert _noise_ratio(labels) == 0.0

    def test_all_noise(self):
        labels = np.array([-1, -1, -1, -1])
        assert _noise_ratio(labels) == 1.0

    def test_partial_noise(self):
        labels = np.array([0, -1, 0, -1])
        assert _noise_ratio(labels) == 0.5

    def test_custom_noise_label(self):
        labels = np.array([0, 99, 0, 99])
        assert _noise_ratio(labels, noise_label=99) == 0.5

    def test_single_point(self):
        assert _noise_ratio(np.array([0])) == 0.0
        assert _noise_ratio(np.array([-1])) == 1.0

    def test_list_input(self):
        """Should work with lists, not just arrays."""
        assert _noise_ratio([0, -1, 0, -1]) == 0.5


class TestSmallestClustSize:
    """Tests for _smallest_clust_size function."""

    def test_uniform_clusters(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert _smallest_clust_size(labels) == 2

    def test_varying_sizes(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 2])
        assert _smallest_clust_size(labels) == 1

    def test_with_noise(self):
        labels = np.array([-1, 0, 0, 1, 1, 1, -1])
        assert _smallest_clust_size(labels) == 2

    def test_all_noise(self):
        labels = np.array([-1, -1, -1])
        assert _smallest_clust_size(labels) == -1

    def test_single_cluster(self):
        labels = np.array([0, 0, 0])
        assert _smallest_clust_size(labels) == 3

    def test_custom_noise_label(self):
        labels = np.array([0, 0, 99, 1, 99])
        assert _smallest_clust_size(labels, noise_label=99) == 1


class TestRemoveNoiseCluster:
    """Tests for _remove_noise_cluster function."""

    def test_remove_noise_single_array(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, -1, 1, -1])
        (result,) = _remove_noise_cluster(X, labels=labels)
        expected = np.array([[1, 2], [5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_remove_noise_multiple_arrays(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([10, 20, 30, 40])
        labels = np.array([0, -1, 0, -1])
        X_out, y_out = _remove_noise_cluster(X, y, labels=labels)
        np.testing.assert_array_equal(X_out, [[1], [3]])
        np.testing.assert_array_equal(y_out, [10, 30])

    def test_no_noise(self):
        X = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1])
        (result,) = _remove_noise_cluster(X, labels=labels)
        np.testing.assert_array_equal(result, X)

    def test_all_noise(self):
        X = np.array([[1, 2], [3, 4]])
        labels = np.array([-1, -1])
        (result,) = _remove_noise_cluster(X, labels=labels)
        assert len(result) == 0


class TestGetLabels:
    """Tests for _get_labels function."""

    def test_basic_estimator(self, fitted_dbscan):
        labels = _get_labels(fitted_dbscan)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 150  # iris dataset

    def test_pipeline(self, iris_data):
        X, _ = iris_data
        pipe = make_pipeline(prep.StandardScaler(), cluster.KMeans(n_clusters=3, n_init="auto"))
        pipe.fit(X)
        labels = _get_labels(pipe)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 150

    def test_not_fitted(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(NotFittedError):
            _get_labels(est)


# =============================================================================
# scorer.py: Scorer class tests
# =============================================================================


class TestLabelScorerUnsupervised:
    """Tests for _LabelScorerUnsupervised."""

    def test_silhouette_scorer(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorer = SCORERS["silhouette"]
        score = scorer(est, X)
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_calinski_harabasz(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorer = SCORERS["calinski_harabasz"]
        score = scorer(est, X)
        assert isinstance(score, float)
        assert score > 0

    def test_davies_bouldin(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        # davies_bouldin is a loss (lower is better), scorer sign-flips it
        scorer = SCORERS["davies_bouldin"]
        score = scorer(est, X)
        assert isinstance(score, float)

    def test_with_noise_removal(self, iris_data):
        X, _ = iris_data
        est = cluster.DBSCAN(eps=0.5)
        est.fit(X)
        # DBSCAN may produce noise; scorer should handle it
        scorer = SCORERS["silhouette"]
        score = scorer(est, X)
        assert isinstance(score, float)


class TestLabelScorerSupervised:
    """Tests for _LabelScorerSupervised."""

    def test_adjusted_rand(self, iris_data):
        X, y_true = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorer = SCORERS["adjusted_rand"]
        score = scorer(est, X, y_true)
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_mutual_info(self, iris_data):
        X, y_true = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorer = SCORERS["mutual_info"]
        score = scorer(est, X, y_true)
        assert isinstance(score, float)
        assert score >= 0

    def test_normalized_mutual_info(self, iris_data):
        X, y_true = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorer = SCORERS["normalized_mutual_info"]
        score = scorer(est, X, y_true)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_homogeneity(self, iris_data):
        X, y_true = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorer = SCORERS["homogeneity"]
        score = scorer(est, X, y_true)
        assert isinstance(score, float)
        assert 0 <= score <= 1


# =============================================================================
# scorer.py: make_scorer and get_scorer tests
# =============================================================================


class TestMakeScorer:
    """Tests for make_scorer function."""

    def test_unsupervised_scorer(self, iris_data):
        X, _ = iris_data

        def custom_metric(X, labels):
            return len(np.unique(labels))

        scorer = make_scorer(custom_metric, ground_truth=False)
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        score = scorer(est, X)
        assert score == 3

    def test_supervised_scorer(self, iris_data):
        X, y_true = iris_data

        def custom_metric(y_true, y_pred):
            return np.sum(y_true == y_pred) / len(y_true)

        scorer = make_scorer(custom_metric, ground_truth=True)
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        score = scorer(est, X, y_true)
        assert isinstance(score, float)

    def test_greater_is_better_false(self, iris_data):
        X, _ = iris_data

        def loss_func(X, labels):
            return 10.0  # Fixed loss

        scorer_loss = make_scorer(loss_func, ground_truth=False, greater_is_better=False)
        scorer_score = make_scorer(loss_func, ground_truth=False, greater_is_better=True)

        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)

        assert scorer_loss(est, X) == -10.0
        assert scorer_score(est, X) == 10.0

    def test_with_kwargs(self, iris_data):
        X, _ = iris_data
        from sklearn.metrics import silhouette_score

        scorer = make_scorer(silhouette_score, ground_truth=False, metric="manhattan")
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        score = scorer(est, X)
        assert isinstance(score, float)


class TestGetScorer:
    """Tests for get_scorer function."""

    def test_string_scorer(self):
        scorer = get_scorer("silhouette")
        assert callable(scorer)

    def test_string_with_suffix(self):
        scorer1 = get_scorer("silhouette")
        scorer2 = get_scorer("silhouette_score")
        # Both should work and be equivalent
        assert scorer1 is scorer2

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="is not a valid scoring value"):
            get_scorer("invalid_scorer_name")

    def test_callable_passthrough(self):
        def my_scorer(est, X, y=None):
            return 1.0

        result = get_scorer(my_scorer)
        assert result is my_scorer


class TestCheckScoring:
    """Tests for check_scoring function."""

    def test_string_scoring(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        scorer = check_scoring(est, "silhouette")
        assert callable(scorer)

    def test_callable_scoring(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, n_init="auto")

        def custom(estimator, X, y=None):
            return 1.0

        scorer = check_scoring(est, custom)
        assert scorer is custom

    def test_none_with_score_method(self):
        est = ScorableClusterer()
        scorer = check_scoring(est, None)
        assert scorer is _passthrough_scorer

    def test_none_without_score_method(self):
        # Use DBSCAN which doesn't have a score method (unlike KMeans which does)
        est = cluster.DBSCAN()
        with pytest.raises(TypeError, match="have a 'score' method"):
            check_scoring(est, None)

    def test_invalid_estimator(self):
        with pytest.raises(TypeError, match="implementing 'fit' method"):
            check_scoring("not_an_estimator", "silhouette")

    def test_metric_instead_of_scorer_warning(self):
        from sklearn import metrics

        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="looks like it is a metric"):
            check_scoring(est, metrics.silhouette_score)

    def test_iterable_scoring_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="cross_validate"):
            check_scoring(est, ["silhouette", "calinski_harabasz"])


class TestCheckMultimetricScoring:
    """Tests for check_multimetric_scoring function."""

    def test_list_of_strings(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        scorers = check_multimetric_scoring(est, ["silhouette", "calinski_harabasz"])
        assert isinstance(scorers, dict)
        assert "silhouette" in scorers
        assert "calinski_harabasz" in scorers

    def test_tuple_of_strings(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        scorers = check_multimetric_scoring(est, ("silhouette", "calinski_harabasz"))
        assert isinstance(scorers, dict)
        assert len(scorers) == 2

    def test_dict_of_scorers(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        input_scorers = {
            "sil": SCORERS["silhouette"],
            "cal": SCORERS["calinski_harabasz"],
        }
        scorers = check_multimetric_scoring(est, input_scorers)
        assert "sil" in scorers
        assert "cal" in scorers

    def test_empty_list_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="Empty list"):
            check_multimetric_scoring(est, [])

    def test_empty_dict_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="empty dict"):
            check_multimetric_scoring(est, {})

    def test_duplicate_strings_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="Duplicate"):
            check_multimetric_scoring(est, ["silhouette", "silhouette"])

    def test_callable_in_list_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="callables"):
            check_multimetric_scoring(est, [SCORERS["silhouette"], "calinski_harabasz"])

    def test_non_string_in_list_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="Non-string"):
            check_multimetric_scoring(est, ["silhouette", 123])


class TestSCORERSDict:
    """Tests for the SCORERS dictionary."""

    def test_expected_scorers_present(self):
        expected = [
            "silhouette",
            "silhouette_score",
            "davies_bouldin",
            "davies_bouldin_score",
            "calinski_harabasz",
            "calinski_harabasz_score",
            "mutual_info",
            "mutual_info_score",
            "adjusted_rand",
            "adjusted_rand_score",
            "homogeneity",
            "homogeneity_score",
            "completeness",
            "completeness_score",
            "v_measure",
            "v_measure_score",
        ]
        for name in expected:
            assert name in SCORERS, f"Missing scorer: {name}"

    def test_scorers_are_callable(self):
        for name, scorer in SCORERS.items():
            assert callable(scorer), f"Scorer {name} is not callable"

    def test_immutable(self):
        """SCORERS should be immutable (MappingProxyType)."""
        with pytest.raises(TypeError):
            SCORERS["new_scorer"] = lambda: None


# =============================================================================
# search.py: Private function tests
# =============================================================================


class TestCheckFitParams:
    """Tests for _check_fit_params function."""

    def test_none_input(self):
        result = _check_fit_params(np.array([[1, 2]]), None)
        assert result == {}

    def test_dict_input(self):
        params = {"sample_weight": np.array([1, 2])}
        result = _check_fit_params(np.array([[1, 2], [3, 4]]), params)
        assert result == params


class TestCheckParamGrid:
    """Tests for _check_param_grid function."""

    def test_valid_dict(self):
        grid = {"eps": [0.1, 0.2], "min_samples": [3, 5]}
        _check_param_grid(grid)  # Should not raise

    def test_valid_list_of_dicts(self):
        grid = [{"eps": [0.1, 0.2]}, {"min_samples": [3, 5]}]
        _check_param_grid(grid)  # Should not raise

    def test_non_iterable_value(self):
        grid = {"eps": 0.5}  # Should be [0.5]
        with pytest.raises(TypeError, match="needs to be a list"):
            _check_param_grid(grid)

    def test_string_value(self):
        grid = {"metric": "euclidean"}  # Should be ["euclidean"]
        with pytest.raises(TypeError, match="needs to be a list"):
            _check_param_grid(grid)

    def test_empty_list_value(self):
        grid = {"eps": []}
        with pytest.raises(ValueError, match="is empty"):
            _check_param_grid(grid)

    def test_2d_array_value(self):
        grid = {"weights": np.array([[1, 2], [3, 4]])}
        with pytest.raises(ValueError, match="one-dimensional"):
            _check_param_grid(grid)

    def test_numpy_array_value(self):
        grid = {"eps": np.array([0.1, 0.2, 0.3])}
        _check_param_grid(grid)  # Should not raise


class TestAggregateScoreDicts:
    """Tests for _aggregate_score_dicts function."""

    def test_basic_aggregation(self):
        scores = [{"a": 0.1, "b": 0.2}, {"a": 0.3, "b": 0.4}]
        result = _aggregate_score_dicts(scores)
        np.testing.assert_array_almost_equal(result["a"], [0.1, 0.3])
        np.testing.assert_array_almost_equal(result["b"], [0.2, 0.4])

    def test_single_score(self):
        scores = [{"score": 0.5}]
        result = _aggregate_score_dicts(scores)
        np.testing.assert_array_almost_equal(result["score"], [0.5])


class TestInsertErrorScores:
    """Tests for _insert_error_scores function."""

    def test_single_metric_failed(self):
        results = [
            {"scores": 0.5, "fit_failed": False},
            {"scores": 0.0, "fit_failed": True},
        ]
        _insert_error_scores(results, error_score=-1)
        assert results[0]["scores"] == 0.5
        assert results[1]["scores"] == -1

    def test_multi_metric_failed(self):
        results = [
            {"scores": {"a": 0.5, "b": 0.6}, "fit_failed": False},
            {"scores": {"a": 0.0, "b": 0.0}, "fit_failed": True},
        ]
        _insert_error_scores(results, error_score=-1)
        assert results[0]["scores"] == {"a": 0.5, "b": 0.6}
        assert results[1]["scores"] == {"a": -1, "b": -1}


class TestNormalizeScoreResults:
    """Tests for _normalize_score_results function."""

    def test_dict_scores(self):
        scores = [{"a": 0.1, "b": 0.2}, {"a": 0.3, "b": 0.4}]
        result = _normalize_score_results(scores)
        assert "a" in result
        assert "b" in result

    def test_scalar_scores(self):
        scores = [0.1, 0.2, 0.3]
        result = _normalize_score_results(scores)
        assert "score" in result
        np.testing.assert_array_almost_equal(result["score"], [0.1, 0.2, 0.3])

    def test_custom_key(self):
        scores = [0.1, 0.2]
        result = _normalize_score_results(scores, scalar_score_key="custom")
        assert "custom" in result


class TestEstimatorHas:
    """Tests for _estimator_has helper."""

    def test_unfitted_with_attribute(self):
        est = PredictableClusterer()
        search = ClusterTuner(est, {"n_clusters": [2, 3]}, scoring="silhouette")
        check = _estimator_has("predict")
        assert check(search) is True

    def test_unfitted_without_attribute(self):
        est = cluster.DBSCAN()
        search = ClusterTuner(est, {"eps": [0.5]}, scoring="silhouette")
        check = _estimator_has("predict")
        assert check(search) is False


# =============================================================================
# search.py: _score and _fit_and_score tests
# =============================================================================


class TestScoreFunction:
    """Tests for _score function."""

    @pytest.mark.filterwarnings("ignore:Noise ratio")
    def test_high_noise_error_score(self, iris_data):
        X, _ = iris_data
        est = HighNoiseClusterer(noise_ratio=0.5)
        est.fit(X)
        scorer = SCORERS["silhouette"]
        score = _score(est, X, None, scorer, error_score=-1, max_noise=0.1)
        assert score == -1

    def test_high_noise_raise(self, iris_data):
        X, _ = iris_data
        est = HighNoiseClusterer(noise_ratio=0.5)
        est.fit(X)
        scorer = SCORERS["silhouette"]
        with pytest.raises(RuntimeError, match="Noise ratio"):
            _score(est, X, None, scorer, error_score="raise", max_noise=0.1)

    @pytest.mark.filterwarnings("ignore:Smallest cluster")
    def test_small_cluster_error_score(self, iris_data):
        X, _ = iris_data
        est = TinyClusterClusterer()
        est.fit(X)
        scorer = SCORERS["silhouette"]
        score = _score(est, X, None, scorer, error_score=-1, min_cluster_size=5)
        assert score == -1

    def test_small_cluster_raise(self, iris_data):
        X, _ = iris_data
        est = TinyClusterClusterer()
        est.fit(X)
        scorer = SCORERS["silhouette"]
        with pytest.raises(RuntimeError, match="Smallest cluster"):
            _score(est, X, None, scorer, error_score="raise", min_cluster_size=5)

    def test_multi_scorer(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)
        scorers = {
            "sil": SCORERS["silhouette"],
            "cal": SCORERS["calinski_harabasz"],
        }
        scores = _score(est, X, None, scorers, max_noise=1.0, min_cluster_size=1)
        assert isinstance(scores, dict)
        assert "sil" in scores
        assert "cal" in scores


class TestFitAndScore:
    """Tests for _fit_and_score function."""

    def test_basic_fit_and_score(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        scorer = SCORERS["silhouette"]
        result = _fit_and_score(
            est,
            X,
            None,
            scorer,
            verbose=0,
            parameters=None,
            fit_params=None,
            return_times=True,
            max_noise=1.0,
            min_cluster_size=1,
        )
        assert "scores" in result
        assert "fit_time" in result
        assert "score_time" in result
        assert result["fit_failed"] is False

    def test_with_parameters(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_init="auto")
        scorer = SCORERS["silhouette"]
        result = _fit_and_score(
            est,
            X,
            None,
            scorer,
            verbose=0,
            parameters={"n_clusters": 4, "random_state": 42},
            fit_params=None,
            max_noise=1.0,
            min_cluster_size=1,
        )
        assert result["fit_failed"] is False

    @pytest.mark.filterwarnings("ignore:Estimator fit failed")
    def test_fit_error_handling(self, iris_data):
        X, _ = iris_data
        est = ErrorClusterer()
        scorer = SCORERS["silhouette"]
        result = _fit_and_score(
            est,
            X,
            None,
            scorer,
            verbose=0,
            parameters=None,
            fit_params=None,
            error_score=-1,
        )
        assert result["fit_failed"] is True
        assert result["scores"] == -1

    def test_return_options(self, iris_data):
        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        scorer = SCORERS["silhouette"]
        result = _fit_and_score(
            est,
            X,
            None,
            scorer,
            verbose=0,
            parameters={"n_clusters": 3},
            fit_params=None,
            return_parameters=True,
            return_n_samples=True,
            return_estimator=True,
            return_noise_ratios=True,
            return_smallest_clust_sizes=True,
            max_noise=1.0,
            min_cluster_size=1,
        )
        assert result["parameters"] == {"n_clusters": 3}
        assert result["n_samples"] == 150
        assert "estimator" in result
        assert "noise_ratio" in result
        assert "smallest_clust_size" in result


# =============================================================================
# ClusterTuner: Basic tests
# =============================================================================


class TestClusterTunerBasic:
    """Basic tests for ClusterTuner."""

    @pytest.mark.filterwarnings("ignore:Estimator fit failed", "ignore:One or more")
    def test_fit_error(self):
        X, _ = datasets.make_blobs(n_samples=100, n_features=2, random_state=325)
        grid = {"eps": np.arange(0.25, 3.25, 0.25), "min_samples": [5, 20, 50]}
        search = ClusterTuner(
            ErrorClusterer(), grid, scoring="silhouette", error_score=-1, refit=False
        )
        search.fit(X)
        results = pd.DataFrame(search.results_)
        assert results["noise_ratio"].isna().all()
        assert np.all(results["score"] == -1)

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
    def test_singular_metric(self):
        df, _ = datasets.load_iris(return_X_y=True, as_frame=True)
        grid = {"eps": np.arange(0.25, 3.25, 0.25), "min_samples": [5, 20, 50]}
        search = ClusterTuner(
            cluster.DBSCAN(), grid, scoring="silhouette", error_score=-1
        )
        search.fit(df)
        check_is_fitted(
            search,
            [
                "results_",
                "best_score_",
                "best_index_",
                "best_estimator_",
                "best_params_",
                "labels_",
            ],
        )
        assert round(search.best_score_, 1) == 0.7
        assert search.best_params_["eps"] == 0.75
        assert search.best_params_["min_samples"] == 20
        assert sorted(search.results_.keys()) == sorted(
            [
                "fit_time",
                "param_eps",
                "param_min_samples",
                "params",
                "rank_score",
                "score",
                "score_time",
                "noise_ratio",
                "smallest_clust_size",
            ]
        )

    @pytest.mark.filterwarnings(
        "ignore:Scoring failed",
        "ignore:One or more",
        "ignore:Noise ratio",
    )
    def test_multi_metric(self):
        df, _ = datasets.load_iris(return_X_y=True, as_frame=True)
        grid = {"eps": np.arange(0.25, 3.25, 0.25), "min_samples": [5, 20, 50]}
        search = ClusterTuner(
            cluster.DBSCAN(),
            grid,
            scoring=["silhouette", "calinski_harabasz", "davies_bouldin_score"],
            refit="silhouette",
            error_score=-1,
        )
        search.fit(df)
        check_is_fitted(
            search,
            [
                "results_",
                "best_score_",
                "best_index_",
                "best_estimator_",
                "best_params_",
                "labels_",
            ],
        )
        assert round(search.best_score_, 1) == 0.7
        assert search.best_params_["eps"] == 0.75
        assert search.best_params_["min_samples"] == 20
        assert sorted(search.results_.keys()) == sorted(
            [
                "fit_time",
                "param_eps",
                "param_min_samples",
                "params",
                "rank_silhouette",
                "rank_davies_bouldin_score",
                "davies_bouldin_score",
                "rank_calinski_harabasz",
                "calinski_harabasz",
                "silhouette",
                "score_time",
                "noise_ratio",
                "smallest_clust_size",
            ]
        )

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
    def test_pipeline(self):
        text = [
            pd.DataFrame.__doc__,
            pd.Series.__doc__,
            prep.Binarizer.__doc__,
            prep.MultiLabelBinarizer.__doc__,
            prep.OneHotEncoder.__doc__,
            prep.OrdinalEncoder.__doc__,
            prep.FunctionTransformer.__doc__,
            prep.StandardScaler.__doc__,
            prep.RobustScaler.__doc__,
            prep.MinMaxScaler.__doc__,
            prep.PowerTransformer.__doc__,
            prep.PolynomialFeatures.__doc__,
            prep.SplineTransformer.__doc__,
            prep.QuantileTransformer.__doc__,
        ]
        text = [y for x in text for y in x.split("\n")]
        grid = {
            "kmeans__n_clusters": np.arange(3, 10),
        }
        pipe = make_pipeline(
            TfidfVectorizer(),
            decomposition.TruncatedSVD(random_state=864),
            cluster.KMeans(random_state=6, n_init="auto"),
        )
        search = ClusterTuner(pipe, grid, scoring="silhouette", error_score=-1)
        search.fit(text)
        check_is_fitted(
            search,
            [
                "results_",
                "best_score_",
                "best_index_",
                "best_estimator_",
                "best_params_",
                "labels_",
            ],
        )
        assert 0.5 < search.best_score_
        assert sorted(search.results_.keys()) == sorted(
            [
                "fit_time",
                "param_kmeans__n_clusters",
                "params",
                "rank_score",
                "score",
                "score_time",
                "noise_ratio",
                "smallest_clust_size",
            ]
        )


# =============================================================================
# ClusterTuner: Constraint tests
# =============================================================================


class TestClusterTunerConstraints:
    """Tests for max_noise and min_cluster_size constraints."""

    @pytest.mark.filterwarnings("ignore:Noise ratio", "ignore:One or more")
    def test_max_noise_rejection(self):
        """High noise fits should be rejected when max_noise is set."""
        X, _ = datasets.make_blobs(n_samples=100, centers=2, random_state=42)
        # DBSCAN with high eps will find few points, eps=0.1 will classify most as noise
        grid = {"eps": [0.1, 0.5, 1.0], "min_samples": [2]}
        search = ClusterTuner(
            cluster.DBSCAN(),
            grid,
            scoring="silhouette",
            max_noise=0.05,  # Very strict noise threshold
            error_score=-1,
        )
        search.fit(X)
        # Some fits should have been rejected due to high noise
        results = pd.DataFrame(search.results_)
        assert (results["score"] == -1).any()

    @pytest.mark.filterwarnings("ignore:Smallest cluster", "ignore:One or more")
    def test_min_cluster_size_rejection(self, iris_data):
        """Tiny clusters should be rejected when min_cluster_size is set."""
        X, _ = iris_data
        # Use a clusterer that might produce tiny clusters
        grid = {"n_clusters": [2, 3, 4, 5, 10, 20]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
            min_cluster_size=50,  # Strict minimum
            error_score=-1,
        )
        search.fit(X)
        # High n_clusters should be rejected due to small cluster sizes
        results = pd.DataFrame(search.results_)
        # With 150 samples and n_clusters=20, avg cluster size is 7.5
        high_k_scores = results[results["param_n_clusters"] >= 10]["score"]
        assert (high_k_scores == -1).all()

    def test_default_constraints(self, iris_data):
        """Default constraints should be max_noise=0.1 and min_cluster_size=3."""
        X, _ = iris_data
        search = ClusterTuner(
            cluster.KMeans(n_clusters=3, n_init="auto"),
            {"n_clusters": [3]},
            scoring="silhouette",
        )
        assert search.max_noise == 0.1
        assert search.min_cluster_size == 3


# =============================================================================
# ClusterTuner: Refit variants
# =============================================================================


class TestClusterTunerRefit:
    """Tests for different refit options."""

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
    def test_refit_false(self, iris_data):
        X, _ = iris_data
        grid = {"eps": [0.5, 0.75, 1.0], "min_samples": [5, 10]}
        search = ClusterTuner(
            cluster.DBSCAN(), grid, scoring="silhouette", refit=False, error_score=-1
        )
        search.fit(X)
        assert hasattr(search, "results_")
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_index_")
        assert not hasattr(search, "best_estimator_")
        with pytest.raises(NotFittedError):
            _ = search.labels_

    @pytest.mark.filterwarnings(
        "ignore:Scoring failed", "ignore:Noise ratio", "ignore:One or more"
    )
    def test_refit_callable(self, iris_data):
        X, _ = iris_data
        grid = {"eps": [0.5, 0.75, 1.0], "min_samples": [5, 10]}

        def select_lowest_noise(results):
            """Select the result with lowest noise ratio."""
            noise_ratios = results["noise_ratio"]
            # Replace NaN with inf so they're ranked last
            noise_ratios = np.where(np.isnan(noise_ratios), np.inf, noise_ratios)
            return np.argmin(noise_ratios)

        search = ClusterTuner(
            cluster.DBSCAN(),
            grid,
            scoring="silhouette",
            refit=select_lowest_noise,
            error_score=-1,
        )
        search.fit(X)
        assert hasattr(search, "best_estimator_")
        assert hasattr(search, "best_index_")
        # best_score_ is not available when refit is callable
        assert not hasattr(search, "best_score_")

    @pytest.mark.filterwarnings(
        "ignore:Scoring failed", "ignore:Noise ratio", "ignore:One or more"
    )
    def test_refit_string_multimetric(self, iris_data):
        X, _ = iris_data
        grid = {"eps": [0.5, 0.75], "min_samples": [5]}
        search = ClusterTuner(
            cluster.DBSCAN(),
            grid,
            scoring=["silhouette", "calinski_harabasz"],
            refit="calinski_harabasz",
            error_score=-1,
        )
        search.fit(X)
        assert hasattr(search, "best_estimator_")
        # best_score_ should be based on calinski_harabasz
        assert search.best_score_ == search.results_["calinski_harabasz"][search.best_index_]


# =============================================================================
# ClusterTuner: Delegated methods
# =============================================================================


class TestClusterTunerDelegatedMethods:
    """Tests for delegated methods like predict, transform."""

    def test_predict(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3, 4]}
        search = ClusterTuner(
            PredictableClusterer(),
            grid,
            scoring="silhouette",
            max_noise=1.0,
            min_cluster_size=1,
        )
        search.fit(X)
        predictions = search.predict(X)
        assert len(predictions) == len(X)

    def test_transform(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3, 4]}
        search = ClusterTuner(
            PredictableClusterer(),
            grid,
            scoring="silhouette",
            max_noise=1.0,
            min_cluster_size=1,
        )
        search.fit(X)
        transformed = search.transform(X)
        assert transformed.shape == X.shape

    def test_no_predict_without_refit(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            PredictableClusterer(),
            grid,
            scoring="silhouette",
            refit=False,
            max_noise=1.0,
            min_cluster_size=1,
        )
        search.fit(X)
        with pytest.raises(NotFittedError):
            search.predict(X)

    def test_score_method(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X)
        score = search.score(X)
        assert isinstance(score, float)


# =============================================================================
# ClusterTuner: Supervised scorers
# =============================================================================


class TestClusterTunerSupervised:
    """Tests for supervised scoring with ground truth labels."""

    def test_adjusted_rand_with_y(self, iris_data):
        X, y = iris_data
        grid = {"n_clusters": [2, 3, 4]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="adjusted_rand",
        )
        search.fit(X, y)
        assert hasattr(search, "best_score_")
        # Adjusted rand should be positive for good clustering
        assert search.best_score_ > 0

    def test_multi_supervised_metrics(self, iris_data):
        X, y = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring=["adjusted_rand", "mutual_info", "homogeneity"],
            refit="adjusted_rand",
        )
        search.fit(X, y)
        assert "adjusted_rand" in search.results_
        assert "mutual_info" in search.results_
        assert "homogeneity" in search.results_


# =============================================================================
# ClusterTuner: Edge cases
# =============================================================================


class TestClusterTunerEdgeCases:
    """Edge case tests."""

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:One or more")
    def test_all_noise_clusterer(self, blob_data):
        X, _ = blob_data
        grid = {"dummy": [None]}  # Dummy parameter for AllNoiseClusterer
        search = ClusterTuner(
            AllNoiseClusterer(),
            grid,
            scoring="silhouette",
            error_score=-1,
            max_noise=1.0,  # Allow all noise
            min_cluster_size=1,
        )
        search.fit(X)
        # Should complete but have error scores due to all noise
        assert search.results_["score"][0] == -1

    def test_single_parameter_combination(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X)
        assert len(search.results_["params"]) == 1

    def test_list_of_grids(self, iris_data):
        X, _ = iris_data
        grid = [
            {"n_clusters": [2, 3]},
            {"n_clusters": [4, 5]},
        ]
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X)
        assert len(search.results_["params"]) == 4

    def test_dataframe_input(self, iris_data):
        X, _ = iris_data
        X_df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X_df)
        assert hasattr(search, "best_estimator_")


# =============================================================================
# ClusterTuner: Parallelization
# =============================================================================


class TestClusterTunerParallel:
    """Tests for parallel execution."""

    def test_n_jobs(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [2, 3, 4, 5]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
            n_jobs=2,
        )
        search.fit(X)
        assert len(search.results_["params"]) == 4

    def test_n_jobs_all(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [2, 3, 4]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
            n_jobs=-1,
        )
        search.fit(X)
        assert hasattr(search, "best_estimator_")


# =============================================================================
# ClusterTuner: sklearn compatibility
# =============================================================================


class TestClusterTunerSklearnCompat:
    """Tests for sklearn compatibility features."""

    def test_labels_property(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X)
        labels = search.labels_
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 3

    def test_n_features_in(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X)
        assert search.n_features_in_ == 4

    def test_n_features_in_not_fitted(self):
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(n_init="auto"),
            grid,
            scoring="silhouette",
        )
        with pytest.raises(AttributeError):
            _ = search.n_features_in_

    def test_estimator_type(self):
        grid = {"n_clusters": [3]}
        # Use a custom clusterer that explicitly has _estimator_type
        search = ClusterTuner(
            PredictableClusterer(),
            grid,
            scoring="silhouette",
            max_noise=1.0,
            min_cluster_size=1,
        )
        assert search._estimator_type == "clusterer"

    def test_sklearn_tags(self):
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(n_init="auto"),
            grid,
            scoring="silhouette",
        )
        tags = search.__sklearn_tags__()
        assert tags is not None


# =============================================================================
# ClusterTuner: Verbose output
# =============================================================================


class TestClusterTunerVerbose:
    """Tests for verbose output."""

    def test_verbose_output(self, iris_data, capsys):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
            verbose=1,
        )
        search.fit(X)
        captured = capsys.readouterr()
        assert "Fitting" in captured.out


# =============================================================================
# ClusterTuner: Error handling
# =============================================================================


class TestClusterTunerErrors:
    """Tests for error handling."""

    def test_error_score_raise(self, iris_data):
        X, _ = iris_data
        grid = {"eps": [0.5]}
        search = ClusterTuner(
            ErrorClusterer(),
            grid,
            scoring="silhouette",
            error_score="raise",
        )
        with pytest.raises(RuntimeError, match="This is a drill"):
            search.fit(X)

    def test_invalid_error_score(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(n_init="auto"),
            grid,
            scoring="silhouette",
            error_score="invalid",
        )
        with pytest.raises(ValueError, match="error_score must be"):
            search.fit(X)

    def test_invalid_refit_multimetric(self, iris_data):
        X, _ = iris_data
        grid = {"n_clusters": [3]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring=["silhouette", "calinski_harabasz"],
            refit=True,  # Should be a string when multimetric
        )
        with pytest.raises(ValueError, match="refit must be set to a scorer key"):
            search.fit(X)

    def test_empty_param_grid(self):
        with pytest.raises(ValueError):
            ClusterTuner(
                cluster.KMeans(n_init="auto"),
                {"n_clusters": []},
                scoring="silhouette",
            )


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests covering full workflows."""

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
    def test_full_dbscan_workflow(self, iris_data):
        """Complete workflow with DBSCAN."""
        X, y = iris_data
        grid = {
            "eps": np.arange(0.3, 1.0, 0.1),
            "min_samples": [3, 5, 10],
        }
        search = ClusterTuner(
            cluster.DBSCAN(),
            grid,
            scoring=["silhouette", "adjusted_rand"],
            refit="silhouette",
            error_score=-1,
            max_noise=0.3,
            min_cluster_size=5,
        )
        search.fit(X, y)

        # Check all expected attributes
        assert hasattr(search, "best_estimator_")
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert hasattr(search, "best_index_")
        assert hasattr(search, "results_")
        assert hasattr(search, "labels_")
        assert hasattr(search, "refit_time_")
        assert search.multimetric_ is True

        # Results should be a proper dict
        results = search.results_
        assert "silhouette" in results
        assert "adjusted_rand" in results
        assert "noise_ratio" in results
        assert "smallest_clust_size" in results

    def test_full_kmeans_workflow(self, iris_data):
        """Complete workflow with KMeans."""
        X, y = iris_data
        grid = {"n_clusters": [2, 3, 4, 5]}
        search = ClusterTuner(
            cluster.KMeans(random_state=42, n_init="auto"),
            grid,
            scoring="silhouette",
        )
        search.fit(X)

        # Verify scoring works
        score = search.score(X)
        assert isinstance(score, float)
        assert score == search.best_score_

        # Verify best_params leads to best_score
        results_df = pd.DataFrame(search.results_)
        best_row = results_df.iloc[search.best_index_]
        assert best_row["score"] == search.best_score_

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
    def test_pipeline_workflow(self):
        """Complete workflow with Pipeline."""
        X, _ = datasets.make_blobs(n_samples=200, n_features=10, centers=4, random_state=42)
        pipe = make_pipeline(
            prep.StandardScaler(),
            decomposition.PCA(n_components=3, random_state=42),
            cluster.DBSCAN(),
        )
        grid = {
            "dbscan__eps": [0.3, 0.5, 0.7],
            "dbscan__min_samples": [3, 5],
        }
        search = ClusterTuner(
            pipe,
            grid,
            scoring="silhouette",
            error_score=-1,
        )
        search.fit(X)
        assert hasattr(search, "best_estimator_")
        assert isinstance(search.best_estimator_, Pipeline)
