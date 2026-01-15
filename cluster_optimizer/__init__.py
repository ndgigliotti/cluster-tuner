"""cluster-optimizer: Hyperparameter optimization for clustering algorithms."""

from cluster_optimizer.scorer import SCORERS, make_scorer
from cluster_optimizer.search import ClusterOptimizer

__version__ = "0.1.0"
__all__ = ["ClusterOptimizer", "make_scorer", "SCORERS"]
