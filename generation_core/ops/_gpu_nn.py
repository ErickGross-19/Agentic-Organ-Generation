"""GPU-accelerated nearest-neighbor and spatial query helpers for space colonization.

All functions gracefully fall back to CPU (scipy cKDTree) when GPU libraries
(PyTorch CUDA) are not available.

Phases 2-4 of the SC performance optimization:
- Phase 2: Batch collision pre-filter (midpoint distance check)
- Phase 3: Batch growth direction computation
- Phase 4: PersistentGPUIndex for NN + kill queries with persistent tissue on GPU

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
from typing import Optional, Tuple, List, Dict

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


def batch_collision_prefilter(
    candidate_starts: np.ndarray,
    candidate_ends: np.ndarray,
    candidate_radii: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_radii: np.ndarray,
    buffer: float = 0.0,
) -> np.ndarray:
    """Batch midpoint-distance collision pre-filter for multiple candidates.

    For each candidate segment, computes the minimum distance from its midpoint
    to all existing segment midpoints. If that distance is less than the sum of
    half-lengths + radii + buffer, the candidate *might* collide (needs narrow-phase).
    Otherwise, it's guaranteed clear.

    Parameters
    ----------
    candidate_starts : (C, 3) float64
    candidate_ends : (C, 3) float64
    candidate_radii : (C,) float64
    segment_starts : (S, 3) float64
    segment_ends : (S, 3) float64
    segment_radii : (S,) float64
    buffer : float

    Returns
    -------
    might_collide : (C,) bool — True if candidate needs narrow-phase check
    """
    n_cand = len(candidate_starts)
    n_seg = len(segment_starts)

    if n_cand == 0 or n_seg == 0:
        return np.zeros(n_cand, dtype=bool)

    cand_mids = (candidate_starts + candidate_ends) * 0.5
    cand_half_len = np.linalg.norm(candidate_ends - candidate_starts, axis=1) * 0.5
    seg_mids = (segment_starts + segment_ends) * 0.5
    seg_half_len = np.linalg.norm(segment_ends - segment_starts, axis=1) * 0.5

    if _TORCH_CUDA_AVAILABLE and n_cand * n_seg >= 10000:
        _warmup_gpu()
        return _torch_batch_collision_prefilter(
            cand_mids, cand_half_len, candidate_radii,
            seg_mids, seg_half_len, segment_radii, buffer,
        )

    dists = np.linalg.norm(cand_mids[:, np.newaxis, :] - seg_mids[np.newaxis, :, :], axis=2)
    thresholds = (cand_half_len[:, np.newaxis] + seg_half_len[np.newaxis, :]
                  + candidate_radii[:, np.newaxis] + segment_radii[np.newaxis, :] + buffer)
    return np.any(dists < thresholds, axis=1)


def _torch_batch_collision_prefilter(
    cand_mids: np.ndarray,
    cand_half_len: np.ndarray,
    cand_radii: np.ndarray,
    seg_mids: np.ndarray,
    seg_half_len: np.ndarray,
    seg_radii: np.ndarray,
    buffer: float,
) -> np.ndarray:
    """GPU-accelerated batch collision pre-filter using torch.cdist."""
    import torch  # type: ignore
    device = torch.device("cuda")

    cm_t = torch.from_numpy(np.ascontiguousarray(cand_mids, dtype=np.float32)).to(device)
    sm_t = torch.from_numpy(np.ascontiguousarray(seg_mids, dtype=np.float32)).to(device)
    chl_t = torch.from_numpy(np.ascontiguousarray(cand_half_len, dtype=np.float32)).to(device)
    shl_t = torch.from_numpy(np.ascontiguousarray(seg_half_len, dtype=np.float32)).to(device)
    cr_t = torch.from_numpy(np.ascontiguousarray(cand_radii, dtype=np.float32)).to(device)
    sr_t = torch.from_numpy(np.ascontiguousarray(seg_radii, dtype=np.float32)).to(device)

    with torch.no_grad():
        chunk = _choose_query_chunk_size(len(cm_t), len(sm_t))
        result = np.zeros(len(cm_t), dtype=bool)
        for start in range(0, len(cm_t), chunk):
            end = min(start + chunk, len(cm_t))
            dists = torch.cdist(cm_t[start:end], sm_t)
            thresh = (chl_t[start:end].unsqueeze(1) + shl_t.unsqueeze(0)
                      + cr_t[start:end].unsqueeze(1) + sr_t.unsqueeze(0) + buffer)
            result[start:end] = torch.any(dists < thresh, dim=1).cpu().numpy()

    return result


def batch_direction_average(
    tip_positions: np.ndarray,
    attracted_positions_list: List[np.ndarray],
    preferred_direction: Optional[np.ndarray] = None,
    directional_bias: float = 0.0,
    prev_directions: Optional[np.ndarray] = None,
    smoothing_weight: float = 0.0,
) -> np.ndarray:
    """Compute normalized average attraction directions for multiple tips at once.

    Parameters
    ----------
    tip_positions : (T, 3) float64
    attracted_positions_list : list of (A_i, 3) arrays, one per tip
    preferred_direction : (3,) float64, optional
    directional_bias : float
    prev_directions : (T, 3) float64, optional
    smoothing_weight : float

    Returns
    -------
    directions : (T, 3) float64 — unit vectors, zeros for degenerate cases
    """
    n_tips = len(tip_positions)
    directions = np.zeros((n_tips, 3), dtype=np.float64)

    for i in range(n_tips):
        attracted = attracted_positions_list[i]
        if len(attracted) == 0:
            continue
        raw = attracted - tip_positions[i]
        norms = np.linalg.norm(raw, axis=1)
        valid = norms > 1e-10
        if not np.any(valid):
            continue
        unit_dirs = raw[valid] / norms[valid, np.newaxis]
        avg = unit_dirs.sum(axis=0)
        mag = np.linalg.norm(avg)
        if mag < 1e-10:
            continue
        avg = avg / mag

        if preferred_direction is not None and directional_bias > 0:
            w_prev = smoothing_weight if prev_directions is not None else 0.0
            if prev_directions is not None:
                blended = (1 - directional_bias - w_prev) * avg + directional_bias * preferred_direction + w_prev * prev_directions[i]
            else:
                blended = (1 - directional_bias) * avg + directional_bias * preferred_direction
            blended_norm = np.linalg.norm(blended)
            if blended_norm > 1e-10:
                avg = blended / blended_norm

        directions[i] = avg

    return directions


class PersistentGPUIndex:
    """Keeps tissue points persistently on GPU to avoid repeated CPU→GPU transfers.

    Maintains an active mask on GPU so killed attractors are excluded without
    re-uploading the full array each step.
    """

    def __init__(self, tissue_points: np.ndarray):
        self._n = len(tissue_points)
        self._tissue_np = np.ascontiguousarray(tissue_points, dtype=np.float64)
        self._tissue_gpu = None
        self._active_mask_gpu = None
        self._on_gpu = False

        if _TORCH_CUDA_AVAILABLE and self._n >= _GPU_MIN_QUERIES:
            try:
                import torch  # type: ignore
                _warmup_gpu()
                self._tissue_gpu = torch.from_numpy(
                    self._tissue_np.astype(np.float32)
                ).to(torch.device("cuda"))
                self._active_mask_gpu = torch.ones(self._n, dtype=torch.bool, device="cuda")
                self._on_gpu = True
                logger.debug("PersistentGPUIndex: %d tissue points uploaded to GPU", self._n)
            except Exception:
                self._on_gpu = False

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu

    def nn_query(
        self,
        active_indices: np.ndarray,
        database: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbor query for active tissue points against a database (tip positions).

        Parameters
        ----------
        active_indices : (A,) int — indices of active tissue points
        database : (D, 3) float64 — tip positions

        Returns
        -------
        distances : (A,) float64
        indices : (A,) int — index into database
        """
        if len(active_indices) == 0 or len(database) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp)

        if self._on_gpu and len(active_indices) >= _GPU_MIN_QUERIES:
            import torch  # type: ignore
            device = torch.device("cuda")
            db_t = torch.from_numpy(
                np.ascontiguousarray(database, dtype=np.float32)
            ).to(device)
            idx_t = torch.from_numpy(active_indices.astype(np.int64)).to(device)
            active_pts = self._tissue_gpu[idx_t]

            chunk = _choose_query_chunk_size(len(active_pts), len(db_t))
            out_dist = np.empty(len(active_pts), dtype=np.float64)
            out_idx = np.empty(len(active_pts), dtype=np.intp)

            with torch.no_grad():
                for start in range(0, len(active_pts), chunk):
                    end = min(start + chunk, len(active_pts))
                    d = torch.cdist(active_pts[start:end], db_t)
                    min_dist, min_idx = torch.min(d, dim=1)
                    out_dist[start:end] = min_dist.cpu().numpy().astype(np.float64)
                    out_idx[start:end] = min_idx.cpu().numpy().astype(np.intp)

            return out_dist, out_idx

        active_positions = self._tissue_np[active_indices]
        return nearest_neighbor(active_positions, database, k=1)

    def kill_within_radius(
        self,
        active_indices: np.ndarray,
        all_node_positions: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Find which active tissue points are within kill_radius of any node.

        Parameters
        ----------
        active_indices : (A,) int
        all_node_positions : (N, 3) float64
        radius : float

        Returns
        -------
        kill_mask : (A,) bool — True for tissue points that should be killed
        """
        if len(active_indices) == 0 or len(all_node_positions) == 0:
            return np.zeros(len(active_indices), dtype=bool)

        if self._on_gpu and len(active_indices) >= _GPU_MIN_QUERIES:
            import torch  # type: ignore
            device = torch.device("cuda")
            nodes_t = torch.from_numpy(
                np.ascontiguousarray(all_node_positions, dtype=np.float32)
            ).to(device)
            idx_t = torch.from_numpy(active_indices.astype(np.int64)).to(device)
            active_pts = self._tissue_gpu[idx_t]

            chunk = _choose_query_chunk_size(len(active_pts), len(nodes_t))
            kill_mask = np.zeros(len(active_pts), dtype=bool)

            with torch.no_grad():
                for start in range(0, len(active_pts), chunk):
                    end = min(start + chunk, len(active_pts))
                    d = torch.cdist(active_pts[start:end], nodes_t)
                    min_dist, _ = torch.min(d, dim=1)
                    kill_mask[start:end] = (min_dist <= radius).cpu().numpy()

            return kill_mask

        active_positions = self._tissue_np[active_indices]
        return range_search(active_positions, all_node_positions, radius)
