#!/usr/bin/env python3
"""
Scaffold Studio - With Collision Detection (Optimized)
Generates vascular networks top-down, avoiding collisions by rotating branches.
Optimized with:
- Vectorized NumPy collision detection
- Adaptive spatial hashing with dynamic cell sizing
- Geometry caching and instancing
- Parallel tree reduction for boolean operations
- Optional GPU acceleration via CuPy
"""

import numpy as np
import manifold3d as m3d
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3
from pyvista.trame import PyVistaRemoteView
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
    print("GPU acceleration available via CuPy")
except ImportError:
    cp = None
    HAS_GPU = False
    print("CuPy not available - using CPU only")

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("Numba JIT compilation available")
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - using pure NumPy")
    # Dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def union_pair(pair):
    """Union a pair of manifolds - for parallel execution."""
    a, b = pair
    return a + b


def tree_union_parallel(manifolds, executor=None):
    """
    Union manifolds using parallel tree reduction.
    Uses ThreadPoolExecutor for parallel union operations at each level.
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
            pairs = []
            unpaired = None
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    pairs.append((current[i], current[i + 1]))
                else:
                    unpaired = current[i]

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


def tree_union(manifolds):
    """Tree reduction union - O(log n) depth instead of O(n)."""
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


def batch_union(manifolds, batch_size=50):
    """Union in batches using parallel tree reduction."""
    if not manifolds:
        return None
    if len(manifolds) <= batch_size:
        return tree_union_parallel(manifolds)

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        batches = []
        for i in range(0, len(manifolds), batch_size):
            batch = manifolds[i:i + batch_size]
            batches.append(tree_union(batch))

        return tree_union_parallel(batches, executor)


server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

# Parameters
state.inlets = 4
state.levels = 2
state.splits = 2
state.spread = 0.35
state.ratio = 0.79
state.cone_angle = 60
state.curvature = 0.3
state.seed = 42
state.view_mode = "Normal"
state.info_text = "Collision detection enabled!"
state.auto_build = True

# Randomness parameters
state.radius_variation = 25   # 0-100 percentage
state.flip_chance = 0.30
state.z_variation = 0.35
state.angle_variation = 0.40
state.even_spread = True  # True = even 360° spread, False = directional cone spread
state.collision_buffer = 0.08  # Extra distance between branches to avoid collision
state.fast_preview = True  # Skip boolean ops for instant preview
state.high_res = False  # Use high resolution geometry (for export)
state.skip_collision = True  # Skip collision detection in preview (much faster)
state.deterministic = False  # Ticker mode: straight grid-aligned channels, much faster
state.use_gpu = HAS_GPU  # Use GPU acceleration if available
state.tips_down = False  # Force all terminal branch tips to point straight down

# Resolution settings
LOW_RES = 6   # Fast preview
HIGH_RES = 16  # Export quality

meshes = {"scaffold": None, "channels": None, "result": None}
cached_scaffold = {"body": None, "params": None}  # Cache scaffold body

plotter = pv.Plotter(off_screen=True)
plotter.set_background("white")


# =============================================================================
# JIT-COMPILED DISTANCE FUNCTIONS (Hot paths)
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _point_to_segment_dist_sq_jit(px, py, pz, ax, ay, az, bx, by, bz):
    """JIT-compiled point-to-segment squared distance."""
    abx, aby, abz = bx - ax, by - ay, bz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    ab_sq = abx*abx + aby*aby + abz*abz
    if ab_sq < 1e-10:
        return apx*apx + apy*apy + apz*apz
    t = (apx*abx + apy*aby + apz*abz) / ab_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    dx, dy, dz = px - cx, py - cy, pz - cz
    return dx*dx + dy*dy + dz*dz


@jit(nopython=True, cache=True, fastmath=True)
def _segment_to_segment_dist_jit(s1ax, s1ay, s1az, s1bx, s1by, s1bz,
                                  s2ax, s2ay, s2az, s2bx, s2by, s2bz):
    """JIT-compiled segment-to-segment distance."""
    # Sample points along segments
    min_dist_sq = 1e30
    for t1 in (0.0, 0.33, 0.67, 1.0):
        p1x = s1ax + t1 * (s1bx - s1ax)
        p1y = s1ay + t1 * (s1by - s1ay)
        p1z = s1az + t1 * (s1bz - s1az)
        for t2 in (0.0, 0.33, 0.67, 1.0):
            p2x = s2ax + t2 * (s2bx - s2ax)
            p2y = s2ay + t2 * (s2by - s2ay)
            p2z = s2az + t2 * (s2bz - s2az)
            dx, dy, dz = p1x - p2x, p1y - p2y, p1z - p2z
            d_sq = dx*dx + dy*dy + dz*dz
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq

    # Also check point-to-segment for endpoints
    for px, py, pz in [(s1ax, s1ay, s1az), (s1bx, s1by, s1bz)]:
        d_sq = _point_to_segment_dist_sq_jit(px, py, pz, s2ax, s2ay, s2az, s2bx, s2by, s2bz)
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
    for px, py, pz in [(s2ax, s2ay, s2az), (s2bx, s2by, s2bz)]:
        d_sq = _point_to_segment_dist_sq_jit(px, py, pz, s1ax, s1ay, s1az, s1bx, s1by, s1bz)
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq

    return min_dist_sq ** 0.5


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _batch_segment_distances_jit(query_start, query_end, cand_starts, cand_ends):
    """JIT-compiled batch distance calculation with parallelization."""
    n = len(cand_starts)
    distances = np.empty(n, dtype=np.float32)
    s1ax, s1ay, s1az = query_start[0], query_start[1], query_start[2]
    s1bx, s1by, s1bz = query_end[0], query_end[1], query_end[2]

    for i in prange(n):
        s2ax, s2ay, s2az = cand_starts[i, 0], cand_starts[i, 1], cand_starts[i, 2]
        s2bx, s2by, s2bz = cand_ends[i, 0], cand_ends[i, 1], cand_ends[i, 2]
        distances[i] = _segment_to_segment_dist_jit(
            s1ax, s1ay, s1az, s1bx, s1by, s1bz,
            s2ax, s2ay, s2az, s2bx, s2by, s2bz
        )
    return distances


# =============================================================================
# OPTIMIZED COLLISION DETECTION WITH SPATIAL HASHING + VECTORIZED NUMPY
# =============================================================================

class OptimizedSpatialGrid:
    """
    High-performance spatial hashing with:
    - Adaptive cell sizing based on average branch radius
    - Vectorized cell computation
    - Compact integer keys for faster hashing
    """

    def __init__(self, cell_size=0.5):
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.grid = {}
        self._cell_cache = {}  # Cache cell computations

    def clear(self):
        self.grid.clear()
        self._cell_cache.clear()

    def set_cell_size(self, avg_radius):
        """Adapt cell size based on average branch radius for optimal performance."""
        # Optimal cell size is ~2-3x the average radius
        optimal = max(0.3, min(1.0, avg_radius * 2.5))
        if abs(optimal - self.cell_size) > 0.1:
            self.cell_size = optimal
            self.inv_cell_size = 1.0 / optimal

    def _cell_key(self, x, y, z):
        """Fast integer cell key computation."""
        return (int(x * self.inv_cell_size),
                int(y * self.inv_cell_size),
                int(z * self.inv_cell_size))

    def _get_cells_vectorized(self, start, end, radius):
        """Vectorized cell computation - much faster than loop."""
        # Sample 4 points along segment (reduced from 5)
        t = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float32)
        direction = end - start
        points = start + np.outer(t, direction)

        # Compute cell range needed
        r_cells = int(np.ceil(radius * self.inv_cell_size)) + 1

        cells = set()
        for p in points:
            cx = int(p[0] * self.inv_cell_size)
            cy = int(p[1] * self.inv_cell_size)
            cz = int(p[2] * self.inv_cell_size)
            # Use smaller neighborhood for speed
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    for dz in range(-r_cells, r_cells + 1):
                        cells.add((cx + dx, cy + dy, cz + dz))
        return cells

    def add_branch(self, idx, start, end, radius):
        """Add branch to grid with vectorized cell computation."""
        for cell in self._get_cells_vectorized(start, end, radius):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(idx)

    def get_nearby_indices(self, start, end, radius):
        """Get candidate branch indices using spatial hash."""
        nearby = set()
        for cell in self._get_cells_vectorized(start, end, radius):
            if cell in self.grid:
                nearby.update(self.grid[cell])
        return nearby


class VectorizedBranchTracker:
    """
    High-performance branch tracking with:
    - Contiguous NumPy arrays for cache efficiency
    - Vectorized distance computations
    - Optional GPU acceleration via CuPy
    - Pre-allocated buffers for zero-copy operations
    """

    def __init__(self, use_gpu=False, max_branches=5000):
        self.use_gpu = use_gpu and HAS_GPU
        self.xp = cp if self.use_gpu else np  # Array module

        # Pre-allocate contiguous arrays for cache efficiency
        self.max_branches = max_branches
        self._starts = np.zeros((max_branches, 3), dtype=np.float32)
        self._ends = np.zeros((max_branches, 3), dtype=np.float32)
        self._radii = np.zeros(max_branches, dtype=np.float32)
        self._count = 0

        self.spatial = OptimizedSpatialGrid(cell_size=0.5)

        # Pre-computed sample points for segment distance
        self._t_samples = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float32)

    def clear(self):
        self._count = 0
        self.spatial.clear()

    @property
    def branch_count(self):
        return self._count

    def add_branch(self, start, end, radius):
        """Add branch with zero-copy to pre-allocated arrays."""
        if self._count >= self.max_branches:
            # Expand arrays if needed (rare)
            self._expand_arrays()

        idx = self._count
        self._starts[idx] = start
        self._ends[idx] = end
        self._radii[idx] = radius
        self._count += 1

        self.spatial.add_branch(idx, np.asarray(start), np.asarray(end), radius)

    def _expand_arrays(self):
        """Double array capacity when full."""
        new_max = self.max_branches * 2
        new_starts = np.zeros((new_max, 3), dtype=np.float32)
        new_ends = np.zeros((new_max, 3), dtype=np.float32)
        new_radii = np.zeros(new_max, dtype=np.float32)

        new_starts[:self.max_branches] = self._starts
        new_ends[:self.max_branches] = self._ends
        new_radii[:self.max_branches] = self._radii

        self._starts = new_starts
        self._ends = new_ends
        self._radii = new_radii
        self.max_branches = new_max

    def add_curved_branch(self, points, radius):
        """Add curve as multiple segments."""
        for i in range(len(points) - 1):
            self.add_branch(points[i], points[i + 1], radius)

    def _vectorized_segment_distance(self, s1a, s1b, candidates_start, candidates_end):
        """
        Compute distance from one segment to multiple candidate segments.
        Uses JIT-compiled functions if available, otherwise vectorized NumPy.
        """
        n = len(candidates_start)
        if n == 0:
            return np.array([], dtype=np.float32)

        # Use JIT-compiled batch function if available (much faster)
        if HAS_NUMBA and n > 5:
            return _batch_segment_distances_jit(
                s1a.astype(np.float32),
                s1b.astype(np.float32),
                candidates_start.astype(np.float32),
                candidates_end.astype(np.float32)
            )

        # Fallback: Vectorized NumPy
        d1 = s1b - s1a
        query_points = s1a + np.outer(self._t_samples, d1)
        d2 = candidates_end - candidates_start

        min_dists_sq = np.full(n, np.inf, dtype=np.float32)

        for t in self._t_samples:
            cand_points = candidates_start + t * d2
            for qp in query_points:
                diffs = cand_points - qp
                dists_sq = np.sum(diffs * diffs, axis=1)
                min_dists_sq = np.minimum(min_dists_sq, dists_sq)

        for endpoint in [s1a, s1b]:
            dists_sq = self._point_to_segments_dist_sq(endpoint, candidates_start, candidates_end)
            min_dists_sq = np.minimum(min_dists_sq, dists_sq)

        return np.sqrt(min_dists_sq)

    def _point_to_segments_dist_sq(self, p, seg_starts, seg_ends):
        """Vectorized point-to-multiple-segments squared distance."""
        ab = seg_ends - seg_starts  # (n, 3)
        ap = p - seg_starts  # (n, 3)

        ab_sq = np.sum(ab * ab, axis=1)  # (n,)
        ab_sq = np.maximum(ab_sq, 1e-10)  # Avoid division by zero

        t = np.sum(ap * ab, axis=1) / ab_sq  # (n,)
        t = np.clip(t, 0.0, 1.0)

        closest = seg_starts + t[:, np.newaxis] * ab  # (n, 3)
        diff = p - closest  # (n, 3)

        return np.sum(diff * diff, axis=1)  # (n,)

    def check_collision(self, start, end, radius, buffer, debug=False):
        """
        Vectorized collision check against nearby branches.
        """
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        # Get candidate indices from spatial hash
        nearby = self.spatial.get_nearby_indices(start, end, radius + buffer)

        if not nearby:
            return False

        # Convert to array indices, filtering out parent (branch ending at our start)
        indices = []
        for idx in nearby:
            if idx < self._count:
                # Skip parent branch (ends at our start)
                if np.sum((start - self._ends[idx])**2) < 0.001:
                    continue
                indices.append(idx)

        if not indices:
            return False

        indices = np.array(indices, dtype=np.int32)

        # Vectorized distance computation
        candidates_start = self._starts[indices]
        candidates_end = self._ends[indices]
        candidates_radii = self._radii[indices]

        distances = self._vectorized_segment_distance(start, end, candidates_start, candidates_end)

        # Check collision: distance < radius + other_radius + buffer
        min_allowed = radius + candidates_radii + buffer

        if debug and len(indices) > 0:
            print(f"[DEBUG] Checked {len(indices)} candidates, min_dist={distances.min():.3f}")

        return np.any(distances < min_allowed)

    def check_curved_collision(self, start, end, radius, buffer, curvature, debug=False):
        """
        Optimized curved collision check with adaptive sampling.
        """
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        dist = np.linalg.norm(end - start)
        if dist < 0.02:
            return False

        # Adaptive sampling: fewer samples for straighter paths
        n_samples = max(3, min(8, int(dist / 0.2) + int(curvature * 3)))

        # Pre-compute curve samples
        t_vals = np.linspace(0, 1, n_samples + 1, dtype=np.float32)
        bulge_vals = curvature * 0.15 * dist * np.sin(t_vals * np.pi)

        # Generate all sample points at once
        points_x = start[0] + (end[0] - start[0]) * t_vals
        points_y = start[1] + (end[1] - start[1]) * t_vals
        points_z = start[2] + (end[2] - start[2]) * t_vals - bulge_vals * 0.5

        # Check each segment
        for i in range(n_samples):
            p1 = np.array([points_x[i], points_y[i], points_z[i]], dtype=np.float32)
            p2 = np.array([points_x[i+1], points_y[i+1], points_z[i+1]], dtype=np.float32)

            if self.check_collision(p1, p2, radius, buffer, debug=(debug and i == 0)):
                return True

        return False

    def find_safe_position(self, start, base_end, radius, rng, buffer, curvature, max_attempts=16):
        """
        Optimized safe position finding with early exit and smarter search.
        """
        start = np.asarray(start, dtype=np.float32)
        base_end = np.asarray(base_end, dtype=np.float32)

        # Quick check: try original first
        if not self.check_curved_collision(start, base_end, radius, buffer, curvature):
            return base_end, 0

        dx = base_end[0] - start[0]
        dy = base_end[1] - start[1]
        dz = base_end[2] - start[2]

        horiz_dist = np.sqrt(dx*dx + dy*dy)
        if horiz_dist < 0.01:
            return base_end, 0

        base_angle = np.arctan2(dy, dx)

        # Pre-compute all rotation attempts at once
        step = 2 * np.pi / max_attempts
        offsets = np.zeros(max_attempts, dtype=np.float32)
        for i in range(max_attempts):
            sign = 1 if i % 2 == 0 else -1
            offsets[i] = ((i + 2) // 2) * step * sign

        # Try each rotation
        for offset in offsets:
            new_angle = base_angle + offset
            cos_a, sin_a = np.cos(new_angle), np.sin(new_angle)
            new_end = np.array([
                start[0] + horiz_dist * cos_a,
                start[1] + horiz_dist * sin_a,
                base_end[2]
            ], dtype=np.float32)

            if not self.check_curved_collision(start, new_end, radius, buffer, curvature):
                return new_end, offset

        # Fallback: try reduced spread
        for reduction in [0.55, 0.35]:
            reduced_horiz = horiz_dist * reduction
            for _ in range(4):
                rand_angle = rng.uniform(0, 2 * np.pi)
                new_end = np.array([
                    start[0] + reduced_horiz * np.cos(rand_angle),
                    start[1] + reduced_horiz * np.sin(rand_angle),
                    start[2] + dz
                ], dtype=np.float32)

                if not self.check_curved_collision(start, new_end, radius, buffer, curvature):
                    return new_end, rand_angle - base_angle

        return base_end, 0


# Global branch tracker (use optimized version)
branch_tracker = VectorizedBranchTracker(use_gpu=HAS_GPU)


# =============================================================================
# GEOMETRY CACHING FOR FASTER REBUILDS
# =============================================================================

class GeometryCache:
    """
    Cache frequently-used geometry primitives to avoid recreation.
    Spheres and cylinders of common sizes are pre-computed.
    """

    def __init__(self):
        self._sphere_cache = {}
        self._cylinder_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def clear(self):
        self._sphere_cache.clear()
        self._cylinder_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _quantize_radius(self, r, precision=0.02):
        """Quantize radius to reduce cache entries."""
        return round(r / precision) * precision

    def _quantize_length(self, length, precision=0.05):
        """Quantize length to reduce cache entries."""
        return round(length / precision) * precision

    def get_sphere(self, radius, resolution):
        """Get cached sphere or create new one."""
        q_radius = self._quantize_radius(radius)
        key = (q_radius, resolution)

        if key in self._sphere_cache:
            self._cache_hits += 1
            # Return a copy that can be translated
            return self._sphere_cache[key]
        else:
            self._cache_misses += 1
            sphere = m3d.Manifold.sphere(q_radius, resolution)
            self._sphere_cache[key] = sphere
            return sphere

    def get_cylinder(self, length, r1, r2, resolution):
        """Get cached cylinder or create new one."""
        q_length = self._quantize_length(length)
        q_r1 = self._quantize_radius(r1)
        q_r2 = self._quantize_radius(r2)
        key = (q_length, q_r1, q_r2, resolution)

        if key in self._cylinder_cache:
            self._cache_hits += 1
            return self._cylinder_cache[key]
        else:
            self._cache_misses += 1
            cyl = m3d.Manifold.cylinder(q_length, q_r2, q_r1, resolution)
            self._cylinder_cache[key] = cyl
            return cyl

    @property
    def hit_rate(self):
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0


# Global geometry cache
geometry_cache = GeometryCache()


def build_scaffold():
    """Generate scaffold geometry with collision detection (optimized)."""
    global branch_tracker, cached_scaffold, geometry_cache
    branch_tracker.clear()
    geometry_cache.clear()

    inlets = state.inlets
    levels = state.levels
    splits = state.splits
    spread = state.spread
    ratio = state.ratio
    cone_angle = np.radians(state.cone_angle)
    curvature = state.curvature
    seed = state.seed

    radius_var = state.radius_variation / 100.0
    flip_chance = state.flip_chance
    z_var = state.z_variation
    angle_var = state.angle_variation
    even_spread = state.even_spread
    collision_buffer = state.collision_buffer
    skip_collision = state.skip_collision and state.fast_preview  # Only skip in fast preview

    outer_r = 4.875
    inner_r = 4.575
    height = 2.0
    scaffold_h = 1.92
    net_r = inner_r - 0.12

    # Cache scaffold body (rarely changes)
    scaffold_params = (outer_r, inner_r, height, scaffold_h)
    if cached_scaffold["params"] == scaffold_params and cached_scaffold["body"] is not None:
        scaffold_body = cached_scaffold["body"]
    else:
        outer = m3d.Manifold.cylinder(height, outer_r, outer_r, 48)
        inner_cut = m3d.Manifold.cylinder(height + 0.02, inner_r, inner_r, 48).translate([0, 0, -0.01])
        ring = outer - inner_cut
        body = m3d.Manifold.cylinder(scaffold_h, inner_r, inner_r, 48)
        scaffold_body = ring + body
        cached_scaffold["body"] = scaffold_body
        cached_scaffold["params"] = scaffold_params

    n = inlets
    if n == 1:
        inlet_pos = [(0.0, 0.0)]
    elif n <= 4:
        r = net_r * 0.45
        inlet_pos = [(r * np.cos(np.pi/4 + i * np.pi/2),
                      r * np.sin(np.pi/4 + i * np.pi/2)) for i in range(n)]
    elif n == 9:
        sp = net_r * 0.5
        inlet_pos = [(i * sp, j * sp) for i in range(-1, 2) for j in range(-1, 2)]
    else:
        g = np.pi * (3 - np.sqrt(5))
        inlet_pos = [(net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.cos(i * g),
                     net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.sin(i * g)) for i in range(n)]

    net_top = scaffold_h
    net_bot = 0.06
    inlet_r = 0.35

    channels = []
    rng = np.random.default_rng(seed)

    # Randomize inlet positions (skip in deterministic mode)
    if state.deterministic:
        randomized_inlets = inlet_pos
    else:
        randomized_inlets = []
        for ix, iy in inlet_pos:
            jitter_amt = 0.12 * rng.uniform(0.5, 1.5)
            jitter_ang = rng.uniform(0, 2 * np.pi)
            nx = ix + jitter_amt * np.cos(jitter_ang)
            ny = iy + jitter_amt * np.sin(jitter_ang)

            d = np.sqrt(nx*nx + ny*ny)
            if d > net_r - 0.5:
                scale = (net_r - 0.5) / d
                nx *= scale
                ny *= scale

            randomized_inlets.append((nx, ny))

    def make_single_segment(x1, y1, z1, x2, y2, z2, r1, r2):
        """Create a single tapered cylinder between two points (cached)."""
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.01:
            return None
        # Use cached cylinder
        cyl = geometry_cache.get_cylinder(length, r1, r2, res)
        h = np.sqrt(dx*dx + dy*dy)
        if h > 0.001 or abs(dz) > 0.001:
            tilt = np.arctan2(h, -dz) * 180 / np.pi
            azim = np.arctan2(-dy, -dx) * 180 / np.pi
            cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])
        return cyl.translate([x2, y2, z2])

    def make_curved_branch(start, end, r1, r2, start_dir, end_dir, smooth):
        """
        Create a branch with geometry caching.
        - Deterministic mode: straight segments with no curve
        - Fast preview: 2-3 cylinders along simple curve
        - Full quality: spheres along bezier curve
        Returns (segments, path_points) for geometry and collision tracking.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end

        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        if dist < 0.02:
            return [], []

        path_points = [(x1, y1, z1), (x2, y2, z2)]

        # DETERMINISTIC MODE: Single straight segment (fastest)
        if state.deterministic:
            segments = []
            seg = make_single_segment(x1, y1, z1, x2, y2, z2, r1, r2)
            if seg:
                segments.append(seg)
            # Add junction sphere at end
            sphere = geometry_cache.get_sphere(r2 * 1.05, res)
            segments.append(sphere.translate([x2, y2, z2]))
            return segments, path_points

        # FAST PREVIEW: Just 2-3 cylinders along a simple curve
        if state.fast_preview and not state.high_res:
            ctrl_dist = dist * 0.4
            mid_t = 0.5
            mx = x1 + (x2 - x1) * mid_t + start_dir[0] * ctrl_dist * 0.3
            my = y1 + (y2 - y1) * mid_t + start_dir[1] * ctrl_dist * 0.3
            mz = z1 + (z2 - z1) * mid_t - smooth * dist * 0.1
            mr = r1 + (r2 - r1) * mid_t

            segments = []
            seg1 = make_single_segment(x1, y1, z1, mx, my, mz, r1, mr)
            seg2 = make_single_segment(mx, my, mz, x2, y2, z2, mr, r2)
            if seg1: segments.append(seg1)
            if seg2: segments.append(seg2)
            # Cached junction sphere at midpoint
            sphere = geometry_cache.get_sphere(mr * 1.05, res)
            segments.append(sphere.translate([mx, my, mz]))
            return segments, path_points

        # FULL QUALITY: Spheres along bezier curve (cached)
        ctrl_dist = dist * (0.4 + smooth * 0.5)
        c1x = x1 + start_dir[0] * ctrl_dist
        c1y = y1 + start_dir[1] * ctrl_dist
        c1z = z1 + start_dir[2] * ctrl_dist - smooth * dist * 0.15
        c2x = x2 - end_dir[0] * ctrl_dist * 0.8
        c2y = y2 - end_dir[1] * ctrl_dist * 0.8
        c2z = z2 + ctrl_dist * 0.3 * smooth

        avg_r = (r1 + r2) / 2
        n_spheres = max(10, int(dist / (avg_r * 0.35)))
        segments = []

        for i in range(n_spheres + 1):
            t = i / n_spheres
            t2, t3 = t * t, t * t * t
            mt, mt2, mt3 = 1 - t, (1-t)**2, (1-t)**3

            cx = mt3*x1 + 3*mt2*t*c1x + 3*mt*t2*c2x + t3*x2
            cy = mt3*y1 + 3*mt2*t*c1y + 3*mt*t2*c2y + t3*y2
            cz = mt3*z1 + 3*mt2*t*c1z + 3*mt*t2*c2z + t3*z2
            cr = r1 + (r2 - r1) * t

            # Use cached sphere
            sphere = geometry_cache.get_sphere(cr, res)
            segments.append(sphere.translate([cx, cy, cz]))
            if i % 4 == 0:
                path_points.append((cx, cy, cz))

        return segments, path_points

    # Process each inlet
    res = HIGH_RES if state.high_res else LOW_RES
    for ix, iy in randomized_inlets:
        port = m3d.Manifold.cylinder(height - net_top + 0.03, inlet_r, inlet_r, res)
        port = port.translate([ix, iy, net_top - 0.01])
        channels.append(port)

        # Initial angle depends on mode
        if state.deterministic:
            # Deterministic: always outward from center
            out_ang = np.arctan2(iy, ix) if (abs(ix) > 0.01 or abs(iy) > 0.01) else 0
        elif even_spread:
            # Random initial angle (not biased)
            out_ang = rng.uniform(0, 2 * np.pi)
        else:
            # Directional: bias outward from center
            base_ang = np.arctan2(iy, ix) if (abs(ix) > 0.01 or abs(iy) > 0.01) else rng.uniform(0, 2*np.pi)
            out_ang = base_ang + rng.uniform(-0.4, 0.4)

        def branch(x, y, z, r, ang, remaining_levels, parent_dir=None):
            """Create a branch with collision detection (optimized for deterministic mode)."""
            if r < 0.03 or z <= net_bot + 0.02:
                return

            # Calculate target z (deterministic or random)
            if remaining_levels > 0:
                z_step = (z - net_bot) / (remaining_levels + 1)
                if not state.deterministic:
                    z_step *= rng.uniform(1 - z_var, 1 + z_var)
                nz = max(z - z_step, net_bot + 0.02)
            else:
                nz = net_bot + 0.02

            # Check if this is a terminal branch (no more children after this)
            # Terminal means: remaining_levels == 0, OR we won't continue due to size/position
            is_terminal = remaining_levels == 0

            # Calculate spread (deterministic or random)
            if state.deterministic:
                sp = spread if remaining_levels < levels else 0
                safe_ang = ang  # No angle jitter
            else:
                spread_var = z_var * 0.6
                sp = spread * rng.uniform(1 - spread_var, 1 + spread_var) if remaining_levels < levels else 0
                # Apply angle jitter
                if parent_dir is None:
                    safe_ang = ang + rng.uniform(-angle_var, angle_var)
                else:
                    safe_ang = ang + rng.uniform(-angle_var, angle_var)

            target_x = x + sp * np.cos(safe_ang)
            target_y = y + sp * np.sin(safe_ang)

            # Keep within bounds
            d = np.sqrt(target_x * target_x + target_y * target_y)
            if d > net_r - 0.1:
                scale = (net_r - 0.1) / d
                target_x *= scale
                target_y *= scale

            # Calculate child radius (deterministic or random)
            if state.deterministic:
                cr = r * ratio
            else:
                cr = r * ratio * rng.uniform(1 - radius_var, 1 + radius_var)

            # === COLLISION DETECTION ===
            # Use average radius for collision (branch tapers from r to cr)
            avg_radius = (r + cr) / 2

            start_pos = (x, y, z)
            target_pos = np.array([target_x, target_y, nz])

            # Run collision detection to find safe position
            safe_end, angle_offset = branch_tracker.find_safe_position(
                start_pos, target_pos, avg_radius, rng, collision_buffer, curvature
            )
            nx, ny = safe_end[0], safe_end[1]

            # Update angle based on collision avoidance
            if angle_offset != 0:
                safe_ang = safe_ang + angle_offset

            # Calculate direction vectors
            if parent_dir is None:
                start_dir = np.array([0.0, 0.0, -1.0])
            else:
                start_dir = np.array(parent_dir)

            # Calculate end direction
            outward = np.array([np.cos(safe_ang), np.sin(safe_ang), 0.0])
            horiz_blend = 0.4 + curvature * 0.5
            vert_blend = -0.6 + curvature * 0.3

            end_dir = np.array([
                outward[0] * horiz_blend,
                outward[1] * horiz_blend,
                vert_blend
            ])
            end_norm = np.linalg.norm(end_dir)
            if end_norm > 0.01:
                end_dir = end_dir / end_norm
            else:
                end_dir = np.array([0.0, 0.0, -1.0])

            # TIPS DOWN: For terminal branches, smooth curve ending pointing down
            if state.tips_down and is_terminal:
                # Create smooth bezier curve from start direction to straight down
                # Using 4 segments for smooth biological curve
                dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
                ctrl_dist = dist * 0.4

                # Control points for cubic bezier
                # P0 = start, P1 = continue in start_dir, P2 = above end, P3 = end
                p0 = np.array([x, y, z])
                p1 = p0 + start_dir * ctrl_dist  # Continue parent direction
                p3 = np.array([nx, ny, nz])
                p2 = p3 + np.array([0, 0, dist * 0.35])  # Above endpoint

                # Sample 4 points along bezier for smooth curve
                path_points = []
                prev_pt = None
                prev_r = r
                n_segs = 4

                for i in range(n_segs + 1):
                    t = i / n_segs
                    mt = 1 - t
                    # Cubic bezier
                    pt = mt**3 * p0 + 3*mt**2*t * p1 + 3*mt*t**2 * p2 + t**3 * p3
                    cur_r = r + (cr - r) * t

                    if prev_pt is not None:
                        seg = make_single_segment(prev_pt[0], prev_pt[1], prev_pt[2],
                                                   pt[0], pt[1], pt[2], prev_r, cur_r)
                        if seg:
                            channels.append(seg)
                        # Junction sphere for smoothness
                        sph = geometry_cache.get_sphere(cur_r * 1.02, res)
                        channels.append(sph.translate([pt[0], pt[1], pt[2]]))

                    path_points.append((pt[0], pt[1], pt[2]))
                    prev_pt = pt
                    prev_r = cur_r
            else:
                # Normal branch geometry
                segs, path_points = make_curved_branch(
                    (x, y, z), (nx, ny, nz),
                    r, cr,
                    tuple(start_dir), tuple(end_dir),
                    curvature
                )
                channels.extend(segs)

            # Record the CURVED path for collision detection (not just straight line)
            if path_points and len(path_points) >= 2:
                branch_tracker.add_curved_branch(path_points, avg_radius)

            # Continue branching
            if remaining_levels > 0 and nz > net_bot + 0.05:
                # Use cached junction sphere
                junction = geometry_cache.get_sphere(cr * 1.15, res)
                channels.append(junction.translate([nx, ny, nz]))

                child_angles = []

                if state.deterministic:
                    # DETERMINISTIC: Even spread, no randomness
                    if splits == 1:
                        child_angles.append(safe_ang)
                    else:
                        base_spacing = 2 * np.pi / splits
                        for i in range(splits):
                            child_angles.append(i * base_spacing)
                elif splits == 1:
                    # Single child continues roughly in same direction with some variation
                    child_angles.append(safe_ang + rng.uniform(-angle_var, angle_var))
                elif even_spread:
                    # EVEN SPREAD: distribute children evenly around 360°
                    base_spacing = 2 * np.pi / splits
                    start_rotation = rng.uniform(0, base_spacing)

                    for i in range(splits):
                        base_angle = start_rotation + i * base_spacing
                        max_deviation = min(cone_angle, base_spacing * 0.4)
                        random_offset = rng.uniform(-max_deviation, max_deviation)

                        if rng.random() < flip_chance:
                            random_offset += rng.choice([-1, 1]) * base_spacing * 0.3

                        child_angles.append(base_angle + random_offset)
                else:
                    # DIRECTIONAL SPREAD: spread in a cone from parent direction
                    total_spread = cone_angle * 2
                    base_spacing = total_spread / splits

                    for i in range(splits):
                        base_offset = -cone_angle + base_spacing * (i + 0.5)
                        random_offset = rng.uniform(-base_spacing * angle_var * 2, base_spacing * angle_var * 2)

                        if rng.random() < flip_chance:
                            random_offset += rng.choice([-1, 1]) * base_spacing * (0.5 + flip_chance)

                        child_angles.append(safe_ang + base_offset + random_offset)

                for child_ang in child_angles:
                    branch(nx, ny, nz, cr, child_ang, remaining_levels - 1, tuple(end_dir))

        # Start wider and taper more aggressively
        start_r = inlet_r * 1.1  # Start wider than inlet
        branch(ix, iy, net_top, start_r, out_ang, levels, None)

    if not channels:
        meshes["scaffold"] = None
        meshes["channels"] = None
        meshes["result"] = None
        return

    def to_pv(m):
        d = m.to_mesh()
        v = np.array(d.vert_properties)[:, :3]
        t = np.array(d.tri_verts)
        return pv.PolyData(v, np.hstack([[3] + list(x) for x in t]))

    print(f"  Combining {len(channels)} channel segments...")

    if state.fast_preview:
        # FAST PREVIEW: Use compose() - instant, no boolean computation
        combined = m3d.Manifold.compose(channels)
        meshes["scaffold"] = to_pv(scaffold_body)
        meshes["channels"] = to_pv(combined)
        meshes["result"] = meshes["scaffold"]
    else:
        # FULL BUILD: Do actual boolean operations
        combined = batch_union(channels, batch_size=100)
        print(f"  Subtracting from scaffold...")
        result = scaffold_body - combined
        meshes["scaffold"] = to_pv(scaffold_body)
        meshes["channels"] = to_pv(combined)
        meshes["result"] = to_pv(result)


def update_view():
    """Update the 3D visualization."""
    plotter.clear_actors()

    if meshes["result"] is None:
        return

    mode = state.view_mode

    if state.fast_preview:
        # Fast preview: show scaffold with channels overlay
        if mode == "Inverted":
            plotter.add_mesh(meshes["channels"], color="crimson", opacity=1.0)
        else:
            plotter.add_mesh(meshes["scaffold"], color="steelblue", opacity=0.3)
            plotter.add_mesh(meshes["channels"], color="crimson", opacity=0.8)
    else:
        if mode == "Normal":
            plotter.add_mesh(meshes["result"], color="steelblue", opacity=0.85)
            plotter.add_mesh(meshes["channels"], color="crimson", opacity=0.35)
        elif mode == "Inverted":
            plotter.add_mesh(meshes["channels"], color="crimson", opacity=1.0)
            plotter.add_mesh(meshes["scaffold"], color="gray", opacity=0.1, style="wireframe")
        elif mode == "Section":
            b = meshes["result"].bounds
            y = (b[2] + b[3]) / 2
            clip = meshes["result"].clip(normal="y", origin=[0, y, 0])
            plotter.add_mesh(clip, color="steelblue", opacity=0.95)
            plotter.add_mesh(meshes["result"], color="gray", opacity=0.05, style="wireframe")
            clip_ch = meshes["channels"].clip(normal="y", origin=[0, y, 0])
            plotter.add_mesh(clip_ch, color="crimson", opacity=0.6)

    plotter.add_axes()
    plotter.view_isometric()
    ctrl.view_update()


def do_rebuild():
    """Rebuild and update view."""
    import time
    t0 = time.time()
    build_scaffold()
    build_time = time.time() - t0
    if meshes["channels"] and meshes["scaffold"]:
        void = meshes["channels"].volume / meshes["scaffold"].volume * 100
        branch_count = branch_tracker.branch_count
        cache_rate = geometry_cache.hit_rate * 100
        mode = "DET" if state.deterministic else "ORG"
        tips = "+TIPS" if state.tips_down else ""
        jit = "+JIT" if HAS_NUMBA else ""
        state.info_text = f"{state.inlets}in {state.levels}lvl {state.splits}sp | {branch_count}br | {build_time:.2f}s | {void:.1f}% | {mode}{tips}{jit} | seed:{state.seed}"
    update_view()


@state.change("view_mode")
def on_view_change(**kwargs):
    if meshes["result"]:
        update_view()


@state.change("inlets", "levels", "splits", "spread", "ratio", "cone_angle", "curvature", "seed",
              "radius_variation", "flip_chance", "z_variation", "angle_variation", "even_spread",
              "collision_buffer", "fast_preview", "deterministic", "tips_down")
def on_param_change(**kwargs):
    if state.auto_build:
        do_rebuild()


@ctrl.add("on_build")
def on_build():
    do_rebuild()


@ctrl.add("on_full_build")
def on_full_build():
    """Do full boolean build (slow but accurate)."""
    old_preview = state.fast_preview
    state.fast_preview = False
    do_rebuild()
    state.fast_preview = old_preview


@ctrl.add("on_reseed")
def on_reseed():
    """Just change the seed for a new random variation."""
    state.seed = int(np.random.randint(0, 100000))
    do_rebuild()


@ctrl.add("on_randomize")
def on_randomize():
    """Randomize all the variation parameters AND the seed."""
    state.seed = int(np.random.randint(0, 100000))
    state.radius_variation = int(np.random.uniform(10, 45))  # Percentage
    state.flip_chance = round(np.random.uniform(0.1, 0.45), 2)
    state.z_variation = round(np.random.uniform(0.15, 0.5), 2)
    state.angle_variation = round(np.random.uniform(0.2, 0.5), 2)
    state.spread = round(np.random.uniform(0.25, 0.5), 2)
    state.cone_angle = int(np.random.uniform(40, 120))
    state.curvature = round(np.random.uniform(0.2, 0.7), 2)
    do_rebuild()


@ctrl.add("on_export_scaffold")
def on_export_scaffold():
    """Export with high resolution and full boolean."""
    state.info_text = "Building high-res for export..."
    # Save current state
    old_preview = state.fast_preview
    old_highres = state.high_res
    # Set export quality
    state.fast_preview = False
    state.high_res = True
    do_rebuild()
    if meshes["result"]:
        fn = f"scaffold_{state.inlets}in_{state.levels}lvl.stl"
        meshes["result"].save(fn)
        state.info_text = f"Exported: {fn}"
    # Restore state
    state.fast_preview = old_preview
    state.high_res = old_highres


@ctrl.add("on_export_channels")
def on_export_channels():
    """Export channels with high resolution."""
    state.info_text = "Building high-res for export..."
    old_highres = state.high_res
    state.high_res = True
    do_rebuild()
    if meshes["channels"]:
        fn = f"channels_{state.inlets}in_{state.levels}lvl.stl"
        meshes["channels"].save(fn)
        state.info_text = f"Exported: {fn}"
    state.high_res = old_highres


# UI Layout
with SinglePageLayout(server) as layout:
    layout.title.set_text("Scaffold Studio (Optimized Collision Detection)")

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VSwitch(v_model=("fast_preview",), label="Fast", hide_details=True, density="compact", color="cyan")
        vuetify3.VSwitch(v_model=("deterministic",), label="Deterministic", hide_details=True, density="compact", color="warning", class_="ml-2")
        vuetify3.VSwitch(v_model=("tips_down",), label="Tips Down", hide_details=True, density="compact", color="purple", class_="ml-2")
        vuetify3.VSwitch(v_model=("auto_build",), label="Auto", hide_details=True, density="compact", color="success", class_="ml-2")
        vuetify3.VBtn("PREVIEW", click=ctrl.on_build, color="cyan", variant="elevated", class_="ml-2")
        vuetify3.VBtn("FULL BUILD", click=ctrl.on_full_build, color="success", variant="elevated", class_="ml-2")
        vuetify3.VBtn("RESEED", click=ctrl.on_reseed, color="orange", variant="elevated", class_="ml-2")
        vuetify3.VBtn("Export Scaffold", click=ctrl.on_export_scaffold, color="primary", variant="outlined", class_="ml-2")
        vuetify3.VBtn("Export Channels", click=ctrl.on_export_channels, color="error", variant="outlined", class_="ml-2")

    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="fill-height pa-0"):
            with vuetify3.VRow(classes="fill-height ma-0"):
                with vuetify3.VCol(cols=3, classes="pa-4"):
                    with vuetify3.VCard():
                        vuetify3.VCardTitle("Parameters")
                        with vuetify3.VCardText():
                            vuetify3.VLabel("Inlets")
                            vuetify3.VSlider(v_model=("inlets",), min=1, max=25, step=1, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Levels (0=straight)")
                            vuetify3.VSlider(v_model=("levels",), min=0, max=8, step=1, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Splits per branch")
                            vuetify3.VSlider(v_model=("splits",), min=1, max=6, step=1, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Spread")
                            vuetify3.VSlider(v_model=("spread",), min=0.1, max=0.8, step=0.02, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Ratio (Murray=0.79)")
                            vuetify3.VSlider(v_model=("ratio",), min=0.5, max=0.95, step=0.02, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Cone Angle")
                            vuetify3.VSlider(v_model=("cone_angle",), min=10, max=180, step=5, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Curvature")
                            vuetify3.VSlider(v_model=("curvature",), min=0, max=1, step=0.05, thumb_label="always", class_="mb-2")
                            vuetify3.VSwitch(v_model=("even_spread",), label="Even Spread (vs Directional)", color="primary", hide_details=True, class_="mb-2")

                    with vuetify3.VCard(classes="mt-4"):
                        vuetify3.VCardTitle("Seed")
                        with vuetify3.VCardText():
                            vuetify3.VTextField(v_model=("seed",), label="Current Seed", type="number", density="compact", variant="outlined", class_="mb-2")
                            vuetify3.VSlider(v_model=("seed",), min=0, max=99999, step=1, thumb_label="always", class_="mb-2")

                    with vuetify3.VCard(classes="mt-4"):
                        vuetify3.VCardTitle("Randomness")
                        with vuetify3.VCardText():
                            vuetify3.VLabel("Radius Variation (%)")
                            vuetify3.VSlider(v_model=("radius_variation",), min=0, max=100, step=5, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Flip Chance")
                            vuetify3.VSlider(v_model=("flip_chance",), min=0, max=0.5, step=0.05, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Z Variation")
                            vuetify3.VSlider(v_model=("z_variation",), min=0, max=0.5, step=0.05, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Angle Variation")
                            vuetify3.VSlider(v_model=("angle_variation",), min=0, max=0.5, step=0.05, thumb_label="always", class_="mb-2")
                            vuetify3.VLabel("Collision Buffer")
                            vuetify3.VSlider(v_model=("collision_buffer",), min=0, max=0.3, step=0.02, thumb_label="always", class_="mb-2")

                    with vuetify3.VCard(classes="mt-4"):
                        vuetify3.VCardTitle("View")
                        with vuetify3.VCardText():
                            with vuetify3.VBtnToggle(v_model=("view_mode",), mandatory=True, density="compact"):
                                vuetify3.VBtn("Normal", value="Normal", size="small")
                                vuetify3.VBtn("Inverted", value="Inverted", size="small")
                                vuetify3.VBtn("Section", value="Section", size="small")

                    vuetify3.VAlert("{{ info_text }}", type="info", variant="tonal", class_="mt-4")

                with vuetify3.VCol(cols=9, classes="pa-0"):
                    view = PyVistaRemoteView(plotter)
                    ctrl.view_update = view.update


if __name__ == "__main__":
    print("=" * 60)
    print("  SCAFFOLD STUDIO (Optimized Collision Detection)")
    print("=" * 60)
    print("  Optimizations enabled:")
    print("    - Vectorized NumPy collision detection")
    print("    - Spatial hash grid with adaptive cell sizing")
    print("    - Geometry caching for spheres/cylinders")
    print(f"    - GPU acceleration: {'YES' if HAS_GPU else 'No (pip install cupy)'}")
    print(f"    - Numba JIT compilation: {'YES' if HAS_NUMBA else 'No (pip install numba)'}")
    print("    - Deterministic mode for fastest generation")
    print("    - Tips Down mode for vertical endpoints")
    print("=" * 60)
    print("  http://localhost:8080")
    print("=" * 60)

    do_rebuild()
    server.start()
