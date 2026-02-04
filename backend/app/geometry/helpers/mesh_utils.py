"""
Enhanced mesh operations for combining manifold geometries.

Provides optimized boolean union operations including tree reduction
and parallel processing for efficient combination of many geometries.
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


def _get_manifold():
    """Get manifold3d module, raising ImportError if not available."""
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )
    return m3d


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
    Union manifolds using m3d.Manifold.batch_boolean for fastest performance.

    Uses manifold3d's native batch boolean operation which is optimized
    for combining multiple geometries efficiently.

    Args:
        manifolds: List of manifold objects to combine

    Returns:
        Single combined manifold, or None if input is empty

    Example:
        >>> spheres = [m3d.Manifold.sphere(1).translate([i*2, 0, 0]) for i in range(10)]
        >>> combined = tree_union(spheres)
    """
    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]

    m3d = _get_manifold()
    return m3d.Manifold.batch_boolean(manifolds, m3d.OpType.Add)


def tree_union_sequential(manifolds: List[Manifold]) -> Optional[Manifold]:
    """
    Union manifolds using tree reduction - O(log n) depth instead of O(n).

    This is the fallback method when batch_boolean is not available.
    Uses pairwise reduction to minimize the complexity of intermediate
    boolean operations.

    This is much faster than sequential union for large numbers of objects
    because boolean operations are roughly O(n) in complexity, so:
    - Sequential: O(n^2) total work
    - Tree reduction: O(n log n) total work

    Args:
        manifolds: List of manifold objects to combine

    Returns:
        Single combined manifold, or None if input is empty

    Example:
        >>> # Instead of: result = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
        >>> # Does:       ((m1+m2) + (m3+m4)) + ((m5+m6) + (m7+m8))
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
    Union manifolds in batches using tree reduction.

    Balances memory usage with speed by processing in chunks.
    Each batch uses tree_union (batch_boolean), then batches are
    combined using parallel tree union.

    Args:
        manifolds: List of manifold objects to combine
        batch_size: Number of manifolds per batch (default 50)
        progress_callback: Optional callback for progress updates

    Returns:
        Single combined manifold, or None if input is empty

    Example:
        >>> channels = [make_channel(...) for _ in range(500)]
        >>> combined = batch_union(channels, batch_size=50)
    """
    if not manifolds:
        return None
    if len(manifolds) <= batch_size:
        return tree_union(manifolds)

    # Process in batches with tree union
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        batches = []
        total_batches = (len(manifolds) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(manifolds), batch_size)):
            batch = manifolds[i:i + batch_size]
            batches.append(tree_union(batch))  # Use batch_boolean within batch

            if progress_callback:
                progress_callback(
                    f"Processed batch {batch_idx + 1}/{total_batches}"
                )

        # Parallel union of batches
        return tree_union_parallel(batches, executor)


def subtract_all(
    base: Manifold,
    subtractors: List[Manifold],
    batch_size: int = 50
) -> Manifold:
    """
    Subtract multiple manifolds from a base manifold efficiently.

    Unions all subtractors first, then performs a single subtraction.
    More efficient than sequential subtractions.

    Args:
        base: Base manifold to subtract from
        subtractors: List of manifolds to subtract
        batch_size: Batch size for combining subtractors

    Returns:
        Base manifold with all subtractors removed

    Example:
        >>> solid = m3d.Manifold.cube([10, 10, 10])
        >>> holes = [m3d.Manifold.sphere(0.5).translate([x, y, z]) for ...]
        >>> result = subtract_all(solid, holes)
    """
    if not subtractors:
        return base

    combined_subtractor = batch_union(subtractors, batch_size)
    return base - combined_subtractor


def intersect_all(manifolds: List[Manifold]) -> Optional[Manifold]:
    """
    Compute intersection of all manifolds in the list.

    Uses tree reduction for efficiency.

    Args:
        manifolds: List of manifold objects to intersect

    Returns:
        Intersection of all manifolds, or None if input is empty
    """
    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]

    m3d = _get_manifold()
    return m3d.Manifold.batch_boolean(manifolds, m3d.OpType.Intersect)


def difference_all(manifolds: List[Manifold]) -> Optional[Manifold]:
    """
    Compute sequential difference: m[0] - m[1] - m[2] - ...

    Different from subtract_all which subtracts union(m[1:]) from m[0].
    This computes m[0] - m[1], then (result) - m[2], etc.

    Args:
        manifolds: List where first element is base, rest are subtracted

    Returns:
        Result of sequential subtraction, or None if input is empty
    """
    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]

    result = manifolds[0]
    for i in range(1, len(manifolds)):
        result = result - manifolds[i]

    return result
