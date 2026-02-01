"""Test suite for cluster_tuner.scorer module."""

import numpy as np
import pytest
from sklearn import cluster, datasets
from sklearn import preprocessing as prep
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline

from cluster_tuner import SCORERS, make_scorer
from cluster_tuner.scorer import (
    _check_multimetric_scoring,
    _get_labels,
    _noise_ratio,
    _passthrough_scorer,
    _remove_noise_cluster,
    _smallest_clust_size,
    check_scoring,
    get_scorer,
)

# =============================================================================
# Helper classes
# =============================================================================


class AllNoiseClusterer(ClusterMixin, BaseEstimator):
    """Clusterer that assigns all points to noise."""

    def __init__(self, dummy=None):
        self.dummy = dummy

    def fit(self, X, y=None):
        self.labels_ = np.full(len(X), -1)
        return self


class ScorableClusterer(ClusterMixin, BaseEstimator):
    """Clusterer with a score method."""

    def fit(self, X, y=None):
        self.labels_ = np.zeros(len(X), dtype=int)
        return self

    def score(self, X, y=None):
        return 0.8


# =============================================================================
# Fixtures
# =============================================================================


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
# Helper function tests
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
        pipe = make_pipeline(
            prep.StandardScaler(), cluster.KMeans(n_clusters=3, n_init="auto")
        )
        pipe.fit(X)
        labels = _get_labels(pipe)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 150

    def test_not_fitted(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(NotFittedError):
            _get_labels(est)


# =============================================================================
# Scorer class tests
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
# make_scorer and get_scorer tests
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

        scorer_loss = make_scorer(
            loss_func, ground_truth=False, greater_is_better=False
        )
        scorer_score = make_scorer(
            loss_func, ground_truth=False, greater_is_better=True
        )

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

    def test_iterable_scoring_returns_multimetric(self):
        from cluster_tuner.scorer import _MultimetricScorer

        est = cluster.KMeans(n_clusters=3, n_init="auto")
        scorer = check_scoring(est, ["silhouette", "calinski_harabasz"])
        assert isinstance(scorer, _MultimetricScorer)


class TestCheckMultimetricScoringPrivate:
    """Tests for _check_multimetric_scoring function."""

    def test_list_of_strings(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        scorers = _check_multimetric_scoring(est, ["silhouette", "calinski_harabasz"])
        assert isinstance(scorers, dict)
        assert "silhouette" in scorers
        assert "calinski_harabasz" in scorers

    def test_tuple_of_strings(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        scorers = _check_multimetric_scoring(est, ("silhouette", "calinski_harabasz"))
        assert isinstance(scorers, dict)
        assert len(scorers) == 2

    def test_dict_of_scorers(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        input_scorers = {
            "sil": SCORERS["silhouette"],
            "cal": SCORERS["calinski_harabasz"],
        }
        scorers = _check_multimetric_scoring(est, input_scorers)
        assert "sil" in scorers
        assert "cal" in scorers

    def test_empty_list_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="Empty list"):
            _check_multimetric_scoring(est, [])

    def test_empty_dict_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="empty dict"):
            _check_multimetric_scoring(est, {})

    def test_duplicate_strings_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="Duplicate"):
            _check_multimetric_scoring(est, ["silhouette", "silhouette"])

    def test_callable_in_list_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="callables"):
            _check_multimetric_scoring(
                est, [SCORERS["silhouette"], "calinski_harabasz"]
            )

    def test_non_string_in_list_error(self):
        est = cluster.KMeans(n_clusters=3, n_init="auto")
        with pytest.raises(ValueError, match="Non-string"):
            _check_multimetric_scoring(est, ["silhouette", 123])


class TestMultimetricScorer:
    """Tests for _MultimetricScorer class."""

    def test_basic_multimetric(self, iris_data):
        from cluster_tuner.scorer import _MultimetricScorer

        X, _ = iris_data
        est = cluster.KMeans(n_clusters=3, random_state=42, n_init="auto")
        est.fit(X)

        scorer = _MultimetricScorer(
            scorers={"sil": SCORERS["silhouette"], "cal": SCORERS["calinski_harabasz"]},
            raise_exc=True,
        )
        scores = scorer(est, X)
        assert isinstance(scores, dict)
        assert "sil" in scores
        assert "cal" in scores

    def test_multimetric_error_handling(self, iris_data):
        from cluster_tuner.scorer import _MultimetricScorer

        X, _ = iris_data
        est = AllNoiseClusterer()
        est.fit(X)

        def bad_scorer(est, X, y=None):
            raise ValueError("Scoring failed")

        scorer = _MultimetricScorer(scorers={"bad": bad_scorer}, raise_exc=False)
        scores = scorer(est, X)
        assert np.isnan(scores["bad"])


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
