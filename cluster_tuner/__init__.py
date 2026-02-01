"""cluster-tuner: Hyperparameter tuning for clustering algorithms."""

from cluster_tuner.scorer import SCORERS, make_scorer
from cluster_tuner.search import ClusterTuner

__version__ = "0.1.0"
__all__ = ["ClusterTuner", "make_scorer", "SCORERS"]

# Backwards compatibility alias
ClusterOptimizer = ClusterTuner
