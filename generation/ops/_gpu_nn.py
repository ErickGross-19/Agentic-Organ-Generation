"""GPU-accelerated nearest-neighbor and spatial query helpers for space colonization.

All functions gracefully fall back to CPU (scipy cKDTree) when GPU libraries
(PyTorch CUDA) are not available.

Phase 2 of the SC performance optimization.

Notes
-----
This implementation uses PyTorch's `torch.cdist` on CUDA for k=1 nearest-neighbor
and for radius checks (via nearest distance). This is O(Q*D) and is therefore
only beneficial for moderate sizes; it is implemented with chunking to avoid
large intermediate allocations.

GPU activates only when query count >= _GPU_MIN_QUERIES (default 2000) to avoid
CPU↔GPU transfer overhead dominating on small datasets. A one-time warmup call
is issued on first use to eliminate PyTorch JIT/CUDA init latency from timing.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE: bool = False
_TORCH_CUDA_AVAILABLE: bool = False
_GPU_WARMED_UP: bool = False
_GPU_MIN_QUERIES: int = 2000
_GPU_MIN_DIR_AVG: int = 1024

_cached_db_tensor = None
_cached_db_id: Optional[int] = None

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
    _TORCH_CUDA_AVAILABLE = bool(torch.cuda.is_available())

    if _TORCH_CUDA_AVAILABLE:
        logger.info("PyTorch CUDA detected — GPU-accelerated NN enabled")
    else:
        logger.debug("PyTorch installed but CUDA unavailable — using CPU fallback")
except Exception as exc:
    logger.debug("PyTorch unavailable (%s: %s) — using scipy cKDTree fallback", type(exc).__name__, exc)


def _warmup_gpu() -> None:
    """One-time CUDA warmup to eliminate JIT latency from first real call."""
    global _GPU_WARMED_UP
    if _GPU_WARMED_UP or not _TORCH_CUDA_AVAILABLE:
        return
    try:
        import torch  # type: ignore
        a = torch.randn(4, 3, device="cuda")
        b = torch.randn(4, 3, device="cuda")
        _ = torch.cdist(a, b)
        torch.cuda.synchronize()
        _GPU_WARMED_UP = True
        logger.debug("GPU warmup complete")
    except Exception:
        _GPU_WARMED_UP = True


def invalidate_gpu_cache() -> None:
    """Clear cached GPU database tensor (call when database changes)."""
    global _cached_db_tensor, _cached_db_id
    _cached_db_tensor = None
    _cached_db_id = None


def gpu_available() -> bool:
    """Return True if any GPU acceleration is available."""

    return _TORCH_CUDA_AVAILABLE


def nearest_neighbor(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors in *database* for each point in *queries*.

    Parameters
    ----------
    queries : (Q, 3) float64
    database : (D, 3) float64
    k : int

    Returns
    -------
    distances : (Q,) float64  — Euclidean distances to nearest neighbor
    indices   : (Q,) int      — index into *database*

    Notes
    -----
    GPU path supports only k=1. For k>1, falls back to CPU.
    """

    if len(database) == 0 or len(queries) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp)

    if _TORCH_CUDA_AVAILABLE and k == 1 and len(queries) >= _GPU_MIN_QUERIES:
        _warmup_gpu()
        return _torch_gpu_knn1(queries, database)

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
    """For each query point, determine whether *any* database point is within *radius*.

    Returns
    -------
    has_neighbor : (Q,) bool
    """

    if len(database) == 0 or len(queries) == 0:
        return np.zeros(len(queries), dtype=bool)

    if _TORCH_CUDA_AVAILABLE and len(queries) >= _GPU_MIN_QUERIES:
        _warmup_gpu()
        distances, _ = _torch_gpu_knn1(queries, database)
        return (distances <= radius).astype(bool)

    tree = cKDTree(database)
    results = tree.query_ball_point(queries, radius)
    return np.array([len(r) > 0 for r in results], dtype=bool)


def vectorized_direction_average(
    attracted_positions: np.ndarray,
    node_pos: np.ndarray,
) -> np.ndarray:
    """Compute the normalized average attraction direction.

    Uses PyTorch on GPU when available and arrays are large enough.

    Parameters
    ----------
    attracted_positions : (A, 3) float64
    node_pos : (3,) float64

    Returns
    -------
    avg_dir : (3,) float64 — unit vector, or zeros if degenerate
    """

    if _TORCH_CUDA_AVAILABLE and len(attracted_positions) >= _GPU_MIN_DIR_AVG:
        _warmup_gpu()
        return _torch_gpu_direction_avg(attracted_positions, node_pos)

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


def _choose_query_chunk_size(
    q_len: int,
    db_len: int,
    max_elements: int = 20_000_000,
) -> int:
    """Choose a chunk size that bounds the intermediate distance matrix."""

    if q_len <= 0:
        return 0
    if db_len <= 0:
        return q_len

    chunk = int(max_elements // max(db_len, 1))
    chunk = max(chunk, 256)
    chunk = min(chunk, q_len)
    return chunk


def _torch_gpu_knn1(
    queries: np.ndarray,
    database: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """PyTorch CUDA k-NN (k=1) search using chunked cdist.

    Caches the database tensor on GPU when the same database array is reused
    across calls (e.g. NN query followed by range_search in the same step).
    """

    import torch  # type: ignore
    global _cached_db_tensor, _cached_db_id

    q32 = np.ascontiguousarray(queries, dtype=np.float32)

    device = torch.device("cuda")

    db_data_id = id(database)
    if _cached_db_id == db_data_id and _cached_db_tensor is not None:
        db_t = _cached_db_tensor
    else:
        db32 = np.ascontiguousarray(database, dtype=np.float32)
        db_t = torch.from_numpy(db32).to(device)
        _cached_db_tensor = db_t
        _cached_db_id = db_data_id

    out_dist = np.empty(len(q32), dtype=np.float64)
    out_idx = np.empty(len(q32), dtype=np.intp)

    chunk = _choose_query_chunk_size(len(q32), len(db_t))

    with torch.no_grad():
        for start in range(0, len(q32), chunk):
            end = min(start + chunk, len(q32))
            q_t = torch.from_numpy(q32[start:end]).to(device)

            d = torch.cdist(q_t, db_t)
            min_dist, min_idx = torch.min(d, dim=1)

            out_dist[start:end] = min_dist.cpu().numpy().astype(np.float64)
            out_idx[start:end] = min_idx.cpu().numpy().astype(np.intp)

    return out_dist, out_idx


def _torch_gpu_direction_avg(
    attracted_positions: np.ndarray,
    node_pos: np.ndarray,
) -> np.ndarray:
    """PyTorch CUDA direction averaging."""

    import torch  # type: ignore

    device = torch.device("cuda")

    pos_t = torch.from_numpy(
        np.ascontiguousarray(attracted_positions, dtype=np.float32)
    ).to(device)
    node_t = torch.from_numpy(
        np.ascontiguousarray(node_pos, dtype=np.float32)
    ).to(device)

    raw = pos_t - node_t
    norms = torch.linalg.norm(raw, dim=1)
    valid = norms > 1e-10
    if not bool(valid.any()):
        return np.zeros(3, dtype=np.float64)

    unit_dirs = raw[valid] / norms[valid].unsqueeze(1)
    avg = unit_dirs.sum(dim=0)
    mag = float(torch.linalg.norm(avg))
    if mag < 1e-10:
        return np.zeros(3, dtype=np.float64)

    out = (avg / mag).cpu().numpy().astype(np.float64)
    return out
