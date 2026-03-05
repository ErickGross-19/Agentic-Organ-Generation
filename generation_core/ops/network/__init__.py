"""
Network-level operations for vascular network manipulation.

This module provides operations for cleaning, merging, and analyzing
vascular networks at the graph level.
"""

from .cleanup import (
    snap_nodes,
    merge_duplicate_nodes,
    prune_short_segments,
    cleanup_network,
    NetworkCleanupPolicy,
)
from .merge import (
    merge_networks,
    NetworkMergePolicy,
)
from .metrics import (
    compute_network_metrics,
    NetworkMetrics,
)

__all__ = [
    "snap_nodes",
    "merge_duplicate_nodes",
    "prune_short_segments",
    "cleanup_network",
    "NetworkCleanupPolicy",
    "merge_networks",
    "NetworkMergePolicy",
    "compute_network_metrics",
    "NetworkMetrics",
]
