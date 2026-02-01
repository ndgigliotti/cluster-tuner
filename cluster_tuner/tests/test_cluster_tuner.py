"""Comprehensive test suite for ClusterTuner."""

import numpy as np
import pandas as pd
import pytest
from sklearn import cluster, datasets, decomposition
from sklearn import preprocessing as prep
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

from cluster_tuner import SCORERS, ClusterTuner
from cluster_tuner.search import (
    _aggregate_score_dicts,
    _check_fit_params,
    _check_param_grid,
    _estimator_has,
    _fit_and_score,
    _insert_error_scores,
    _normalize_score_results,
    _score,
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
        assert np.all(results["test_score"] == -1)

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
                "rank_test_score",
                "test_score",
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
                "rank_test_silhouette",
                "rank_test_davies_bouldin_score",
                "test_davies_bouldin_score",
                "rank_test_calinski_harabasz",
                "test_calinski_harabasz",
                "test_silhouette",
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
                "rank_test_score",
                "test_score",
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
        assert (results["test_score"] == -1).any()

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
        high_k_scores = results[results["param_n_clusters"] >= 10]["test_score"]
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
        assert (
            search.best_score_
            == search.results_["test_calinski_harabasz"][search.best_index_]
        )


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
        assert "test_adjusted_rand" in search.results_
        assert "test_mutual_info" in search.results_
        assert "test_homogeneity" in search.results_


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
        assert search.results_["test_score"][0] == -1

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
        # Use a custom clusterer - estimator type is now accessed via __sklearn_tags__
        search = ClusterTuner(
            PredictableClusterer(),
            grid,
            scoring="silhouette",
            max_noise=1.0,
            min_cluster_size=1,
        )
        # _estimator_type property is deprecated in sklearn 1.8+
        # Use __sklearn_tags__() instead
        tags = search.__sklearn_tags__()
        assert tags.estimator_type == "clusterer"

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
        assert "test_silhouette" in results
        assert "test_adjusted_rand" in results
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
        assert best_row["test_score"] == search.best_score_

    @pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
    def test_pipeline_workflow(self):
        """Complete workflow with Pipeline."""
        X, _ = datasets.make_blobs(
            n_samples=200, n_features=10, centers=4, random_state=42
        )
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
