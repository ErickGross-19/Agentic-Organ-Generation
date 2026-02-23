"""
GPU-accelerated nearest-neighbor and spatial query helpers for space colonization.

All functions gracefully fall back to CPU (scipy cKDTree) when GPU libraries
(faiss-gpu, cupy) are not available.  The public API mirrors the scipy signatures
so callers can swap in transparently.

Phase 2 of the SC performance optimization.
"""

import logging
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple

logger = logging.getLogger(__name__)

_FAISS_GPU_AVAILABLE: bool = False
_CUPY_AVAILABLE: bool = False

try:
    import faiss  # type: ignore
    if faiss.get_num_gpus() > 0:
        _FAISS_GPU_AVAILABLE = True
        logger.info("FAISS GPU detected — GPU-accelerated NN enabled")
    else:
        logger.debug("FAISS found but no GPU — using CPU fallback")
except ImportError:
    logger.debug("FAISS not installed — using scipy cKDTree fallback")

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
    logger.debug("CuPy detected — GPU direction averaging enabled")
except ImportError:
    logger.debug("CuPy not installed — using numpy fallback")


def gpu_available() -> bool:
    """Return True if any GPU acceleration is available."""
    return _FAISS_GPU_AVAILABLE or _CUPY_AVAILABLE


def nearest_neighbor(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors in *database* for each point in *queries*.

    Parameters
    ----------
    queries : (Q, 3) float64
    database : (D, 3) float64
    k : int

    Returns
    -------
    distances : (Q,) float64  — Euclidean distances to nearest neighbor
    indices   : (Q,) int      — index into *database*
    """
    if len(database) == 0 or len(queries) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp)

    if _FAISS_GPU_AVAILABLE and len(queries) >= 256:
        return _faiss_gpu_knn(queries, database, k)

    tree = cKDTree(database)
    distances, indices = tree.query(queries, k=k)
    if k == 1:
        return distances.ravel(), indices.ravel()
    return distances, indices


def range_search(
    queries: np.ndarray,
    database: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    For each query point, determine whether *any* database point is within *radius*.

    Parameters
    ----------
    queries : (Q, 3) float64
    database : (D, 3) float64
    radius : float

    Returns
    -------
    has_neighbor : (Q,) bool
    """
    if len(database) == 0 or len(queries) == 0:
        return np.zeros(len(queries), dtype=bool)

    if _FAISS_GPU_AVAILABLE and len(queries) >= 256:
        return _faiss_gpu_range(queries, database, radius)

    tree = cKDTree(database)
    results = tree.query_ball_point(queries, radius)
    return np.array([len(r) > 0 for r in results], dtype=bool)


def vectorized_direction_average(
    attracted_positions: np.ndarray,
    node_pos: np.ndarray,
) -> np.ndarray:
    """
    Compute the normalized average attraction direction.

    Uses CuPy on GPU when available and arrays are large enough.

    Parameters
    ----------
    attracted_positions : (A, 3) float64
    node_pos : (3,) float64

    Returns
    -------
    avg_dir : (3,) float64 — unit vector, or zeros if degenerate
    """
    if _CUPY_AVAILABLE and len(attracted_positions) >= 1024:
        return _cupy_direction_avg(attracted_positions, node_pos)

    raw = attracted_positions - node_pos
    norms = np.linalg.norm(raw, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return np.zeros(3, dtype=np.float64)
    unit_dirs = raw[valid] / norms[valid, np.newaxis]
    avg = unit_dirs.sum(axis=0)
    mag = np.linalg.norm(avg)
    if mag < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return avg / mag


# ---------------------------------------------------------------------------
# Internal GPU implementations
# ---------------------------------------------------------------------------

def _faiss_gpu_knn(
    queries: np.ndarray,
    database: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS GPU k-NN search."""
    import faiss  # type: ignore

    d = database.shape[1]
    db32 = np.ascontiguousarray(database, dtype=np.float32)
    q32 = np.ascontiguousarray(queries, dtype=np.float32)

    res = faiss.StandardGpuResources()
    res.setTempMemory(64 * 1024 * 1024)

    index_flat = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(db32)

    sq_distances, indices = gpu_index.search(q32, k)

    if k == 1:
        distances = np.sqrt(np.maximum(sq_distances[:, 0], 0.0)).astype(np.float64)
        return distances, indices[:, 0].astype(np.intp)
    distances = np.sqrt(np.maximum(sq_distances, 0.0)).astype(np.float64)
    return distances, indices.astype(np.intp)


def _faiss_gpu_range(
    queries: np.ndarray,
    database: np.ndarray,
    radius: float,
) -> np.ndarray:
    """FAISS GPU range search — returns boolean mask."""
    import faiss  # type: ignore

    d = database.shape[1]
    db32 = np.ascontiguousarray(database, dtype=np.float32)
    q32 = np.ascontiguousarray(queries, dtype=np.float32)

    res = faiss.StandardGpuResources()
    res.setTempMemory(64 * 1024 * 1024)

    index_flat = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(db32)

    sq_distances, _ = gpu_index.search(q32, 1)
    has_neighbor = sq_distances[:, 0] <= radius * radius
    return has_neighbor.astype(bool)


def _cupy_direction_avg(
    attracted_positions: np.ndarray,
    node_pos: np.ndarray,
) -> np.ndarray:
    """CuPy GPU direction averaging."""
    import cupy as cp  # type: ignore

    pos_gpu = cp.asarray(attracted_positions)
    npos_gpu = cp.asarray(node_pos)
    raw = pos_gpu - npos_gpu
    norms = cp.linalg.norm(raw, axis=1)
    valid = norms > 1e-10
    if not cp.any(valid):
        return np.zeros(3, dtype=np.float64)
    unit_dirs = raw[valid] / norms[valid, cp.newaxis]
    avg = unit_dirs.sum(axis=0)
    mag = float(cp.linalg.norm(avg))
    if mag < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return cp.asnumpy(avg / mag).astype(np.float64)
