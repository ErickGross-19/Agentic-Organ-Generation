"""
Geometry core utilities for efficient boolean operations.

Provides tree reduction and parallel processing for combining manifold geometries.
These functions are critical for performance when unioning hundreds of segments.
"""

from __future__ import annotations
from typing import List, Optional, Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    m3d = None
    HAS_MANIFOLD = False

# Type alias for manifold objects
Manifold = TypeVar('Manifold')


def union_pair(pair: tuple) -> Manifold:
    """
    Union a pair of manifolds - designed for parallel execution.

    Args:
        pair: Tuple of two manifold objects (a, b)

    Returns:
        Combined manifold (a + b)
    """
    a, b = pair
    return a + b


def tree_union(manifolds: List[Manifold]) -> Optional[Manifold]:
    """
    Union manifolds using tree reduction - O(log n) depth instead of O(n).

    This is much faster than sequential union for large numbers of objects
    because boolean operations are roughly O(n) in complexity, so:
    - Sequential: O(n^2) total work
    - Tree reduction: O(n log n) total work

    Args:
        manifolds: List of manifold objects to combine

    Returns:
        Single combined manifold, or None if input is empty

    Example:
        # Instead of: result = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
        # Does:       ((m1+m2) + (m3+m4)) + ((m5+m6) + (m7+m8))
    """
    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]

    current = list(manifolds)
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                next_level.append(current[i] + current[i + 1])
            else:
                next_level.append(current[i])
        current = next_level

    return current[0]


def tree_union_parallel(
    manifolds: List[Manifold],
    executor: Optional[ThreadPoolExecutor] = None
) -> Optional[Manifold]:
    """
    Union manifolds using parallel tree reduction.

    Uses ThreadPoolExecutor for parallel union operations at each level.
    Only parallelizes when there are enough pairs to make it worthwhile.

    Args:
        manifolds: List of manifold objects to combine
        executor: Optional ThreadPoolExecutor to reuse. If None, creates one.

    Returns:
        Single combined manifold, or None if input is empty

    Note:
        Parallelization overhead means this is only faster for 5+ pairs
        at each level. For smaller sets, falls back to sequential.
    """
    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]

    own_executor = executor is None
    if own_executor:
        executor = ThreadPoolExecutor(max_workers=mp.cpu_count())

    try:
        current = list(manifolds)
        while len(current) > 1:
            # Create pairs for parallel union
            pairs = []
            unpaired = None
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    pairs.append((current[i], current[i + 1]))
                else:
                    unpaired = current[i]

            # Parallel union of all pairs (only if worth it)
            if len(pairs) > 4:
                results = list(executor.map(union_pair, pairs))
            else:
                results = [a + b for a, b in pairs]

            if unpaired is not None:
                results.append(unpaired)

            current = results

        return current[0]
    finally:
        if own_executor:
            executor.shutdown(wait=False)


def batch_union(
    manifolds: List[Manifold],
    batch_size: int = 50,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[Manifold]:
    """
    Union manifolds in batches using parallel tree reduction.

    Balances memory usage with speed by processing in chunks.
    Each batch uses sequential tree union, then batches are
    combined using parallel tree union.

    Args:
        manifolds: List of manifold objects to combine
        batch_size: Number of manifolds per batch (default 50)
        progress_callback: Optional callback for progress updates

    Returns:
        Single combined manifold, or None if input is empty

    Example:
        channels = [make_cyl(...) for _ in range(500)]
        combined = batch_union(channels, batch_size=50)
    """
    if not manifolds:
        return None
    if len(manifolds) <= batch_size:
        return tree_union_parallel(manifolds)

    # Process in batches with parallel tree union
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        batches = []
        total_batches = (len(manifolds) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(manifolds), batch_size)):
            batch = manifolds[i:i + batch_size]
            batches.append(tree_union(batch))  # Sequential within batch

            if progress_callback:
                progress_callback(
                    f"Processed batch {batch_idx + 1}/{total_batches}"
                )

        # Parallel union of batches
        return tree_union_parallel(batches, executor)


def check_manifold_available() -> bool:
    """Check if manifold3d library is available."""
    return HAS_MANIFOLD


def get_manifold_module():
    """
    Get the manifold3d module.

    Returns:
        The manifold3d module

    Raises:
        ImportError: If manifold3d is not installed
    """
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )
    return m3d
