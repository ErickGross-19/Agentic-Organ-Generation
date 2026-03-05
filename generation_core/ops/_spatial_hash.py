"""
Fixed-grid spatial hash for fast range queries with a known radius.

Phase 3b of the SC performance optimization.

When the query radius is known in advance (e.g. kill_radius), a spatial hash
with cell_size = radius provides O(1) amortized lookups per query instead of
O(log N) from a KD-tree.  For dense point clouds (> ~5k points) this is
measurably faster than scipy.cKDTree.query_ball_point.
"""

import numpy as np
from typing import Optional


class SpatialHash:
    """
    3-D spatial hash mapping (ix, iy, iz) cell indices to point indices.

    Build once with ``build()``, then query many times with
    ``has_neighbor_mask()``.
    """

    __slots__ = ("_cell_size", "_inv_cell", "_table", "_built")

    def __init__(self, cell_size: float):
        self._cell_size = cell_size
        self._inv_cell = 1.0 / cell_size
        self._table: dict = {}
        self._built = False

    def build(self, points: np.ndarray) -> "SpatialHash":
        """
        Insert *points* (N, 3) into the hash grid.

        Returns self for chaining.
        """
        self._table.clear()
        if len(points) == 0:
            self._built = True
            return self

        cells = np.floor(points * self._inv_cell).astype(np.int64)
        for idx in range(len(cells)):
            key = (int(cells[idx, 0]), int(cells[idx, 1]), int(cells[idx, 2]))
            if key in self._table:
                self._table[key].append(idx)
            else:
                self._table[key] = [idx]

        self._built = True
        return self

    def has_neighbor_mask(
        self,
        queries: np.ndarray,
        points: np.ndarray,
        radius: float,
        radius_sq: Optional[float] = None,
    ) -> np.ndarray:
        """
        For each query, return True if any inserted point is within *radius*.

        Parameters
        ----------
        queries : (Q, 3) float64
        points  : (N, 3) float64  — the same array passed to ``build()``
        radius  : float
        radius_sq : float, optional — pre-computed radius**2

        Returns
        -------
        mask : (Q,) bool
        """
        if not self._built:
            raise RuntimeError("Call build() before querying")

        if len(queries) == 0:
            return np.empty(0, dtype=bool)

        if not self._table:
            return np.zeros(len(queries), dtype=bool)

        r2 = radius_sq if radius_sq is not None else radius * radius
        mask = np.zeros(len(queries), dtype=bool)

        q_cells = np.floor(queries * self._inv_cell).astype(np.int64)

        for qi in range(len(q_cells)):
            cx, cy, cz = int(q_cells[qi, 0]), int(q_cells[qi, 1]), int(q_cells[qi, 2])
            qx, qy, qz = queries[qi, 0], queries[qi, 1], queries[qi, 2]
            found = False
            for dx in (-1, 0, 1):
                if found:
                    break
                for dy in (-1, 0, 1):
                    if found:
                        break
                    for dz in (-1, 0, 1):
                        key = (cx + dx, cy + dy, cz + dz)
                        bucket = self._table.get(key)
                        if bucket is None:
                            continue
                        for pidx in bucket:
                            ddx = points[pidx, 0] - qx
                            ddy = points[pidx, 1] - qy
                            ddz = points[pidx, 2] - qz
                            if ddx * ddx + ddy * ddy + ddz * ddz <= r2:
                                found = True
                                break
            mask[qi] = found

        return mask
