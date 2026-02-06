"""
Vascular perfusion dish geometry generator.

Generates branching tree structures with collision detection to avoid
overlapping branches. Uses spatial hashing and vectorized distance
calculations for efficient collision avoidance.

Key features:
- Collision detection with spatial hashing for efficient lookups
- Vectorized NumPy collision checks
- Adaptive branch positioning when collisions detected
- All features from standard vascular network plus:
  - cone_angle: Control branching cone
  - bottom_height: Where vessels terminate
  - radius_variation, flip_chance, z_variation, angle_variation
  - collision_buffer: Extra spacing between branches
  - even_spread: Distribute evenly vs directional cone
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Callable, Dict, Any, Set
from dataclasses import dataclass, field
import numpy as np

from .core import batch_union, get_manifold_module


# =============================================================================
# SPATIAL GRID FOR COLLISION DETECTION
# =============================================================================

class SpatialGrid:
    """
    Spatial hashing grid for efficient collision queries.
    Uses adaptive cell sizing based on average branch radius.
    """

    def __init__(self, cell_size: float = 0.5):
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}

    def clear(self):
        self.grid.clear()

    def set_cell_size(self, avg_radius: float):
        """Adapt cell size based on average branch radius."""
        optimal = max(0.3, min(1.0, avg_radius * 2.5))
        if abs(optimal - self.cell_size) > 0.1:
            self.cell_size = optimal
            self.inv_cell_size = 1.0 / optimal

    def _cell_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Fast integer cell key computation."""
        return (
            int(x * self.inv_cell_size),
            int(y * self.inv_cell_size),
            int(z * self.inv_cell_size)
        )

    def _get_cells(self, start: np.ndarray, end: np.ndarray, radius: float) -> Set[Tuple[int, int, int]]:
        """Get all cells that a segment passes through."""
        t_samples = np.array([0.0, 0.33, 0.67, 1.0])
        direction = end - start
        points = start + np.outer(t_samples, direction)

        r_cells = int(np.ceil(radius * self.inv_cell_size)) + 1
        cells = set()

        for p in points:
            cx = int(p[0] * self.inv_cell_size)
            cy = int(p[1] * self.inv_cell_size)
            cz = int(p[2] * self.inv_cell_size)
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    for dz in range(-r_cells, r_cells + 1):
                        cells.add((cx + dx, cy + dy, cz + dz))
        return cells

    def add_branch(self, idx: int, start: np.ndarray, end: np.ndarray, radius: float):
        """Add branch to spatial grid."""
        for cell in self._get_cells(start, end, radius):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(idx)

    def get_nearby_indices(self, start: np.ndarray, end: np.ndarray, radius: float) -> Set[int]:
        """Get indices of branches that might collide."""
        nearby = set()
        for cell in self._get_cells(start, end, radius):
            if cell in self.grid:
                nearby.update(self.grid[cell])
        return nearby


# =============================================================================
# BRANCH TRACKER WITH COLLISION DETECTION
# =============================================================================

class BranchTracker:
    """
    Tracks all branches and provides collision detection.
    Uses contiguous NumPy arrays for efficient vectorized operations.
    """

    def __init__(self, max_branches: int = 5000):
        self.max_branches = max_branches
        self._starts = np.zeros((max_branches, 3), dtype=np.float32)
        self._ends = np.zeros((max_branches, 3), dtype=np.float32)
        self._radii = np.zeros(max_branches, dtype=np.float32)
        self._count = 0
        self.spatial = SpatialGrid(cell_size=0.5)
        self._t_samples = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float32)

    def clear(self):
        self._count = 0
        self.spatial.clear()

    @property
    def branch_count(self) -> int:
        return self._count

    def add_branch(self, start: np.ndarray, end: np.ndarray, radius: float):
        """Add a branch segment."""
        if self._count >= self.max_branches:
            self._expand_arrays()

        idx = self._count
        self._starts[idx] = start
        self._ends[idx] = end
        self._radii[idx] = radius
        self._count += 1

        self.spatial.add_branch(idx, start, end, radius)

    def _expand_arrays(self):
        """Double array capacity."""
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

    def add_curved_branch(self, points: List[Tuple[float, float, float]], radius: float):
        """Add a curve as multiple segments."""
        for i in range(len(points) - 1):
            self.add_branch(
                np.array(points[i], dtype=np.float32),
                np.array(points[i + 1], dtype=np.float32),
                radius
            )

    def _point_to_segments_dist_sq(
        self, p: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray
    ) -> np.ndarray:
        """Vectorized point-to-multiple-segments squared distance."""
        ab = seg_ends - seg_starts
        ap = p - seg_starts

        ab_sq = np.sum(ab * ab, axis=1)
        ab_sq = np.maximum(ab_sq, 1e-10)

        t = np.sum(ap * ab, axis=1) / ab_sq
        t = np.clip(t, 0.0, 1.0)

        closest = seg_starts + t[:, np.newaxis] * ab
        diff = p - closest

        return np.sum(diff * diff, axis=1)

    def _segment_distances(
        self, s1a: np.ndarray, s1b: np.ndarray,
        candidates_start: np.ndarray, candidates_end: np.ndarray
    ) -> np.ndarray:
        """Compute distances from one segment to multiple candidates."""
        n = len(candidates_start)
        if n == 0:
            return np.array([], dtype=np.float32)

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

    def check_collision(
        self, start: np.ndarray, end: np.ndarray, radius: float, buffer: float
    ) -> bool:
        """Check if segment collides with existing branches."""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        nearby = self.spatial.get_nearby_indices(start, end, radius + buffer)
        if not nearby:
            return False

        # Filter out parent branch (ends at our start)
        indices = []
        for idx in nearby:
            if idx < self._count:
                if np.sum((start - self._ends[idx])**2) < 0.001:
                    continue
                indices.append(idx)

        if not indices:
            return False

        indices = np.array(indices, dtype=np.int32)
        candidates_start = self._starts[indices]
        candidates_end = self._ends[indices]
        candidates_radii = self._radii[indices]

        distances = self._segment_distances(start, end, candidates_start, candidates_end)
        min_allowed = radius + candidates_radii + buffer

        return np.any(distances < min_allowed)

    def check_curved_collision(
        self, start: np.ndarray, end: np.ndarray, radius: float,
        buffer: float, curvature: float
    ) -> bool:
        """Check collision along a curved path."""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        dist = np.linalg.norm(end - start)
        if dist < 0.02:
            return False

        n_samples = max(3, min(8, int(dist / 0.2) + int(curvature * 3)))
        t_vals = np.linspace(0, 1, n_samples + 1, dtype=np.float32)
        bulge_vals = curvature * 0.15 * dist * np.sin(t_vals * np.pi)

        points_x = start[0] + (end[0] - start[0]) * t_vals
        points_y = start[1] + (end[1] - start[1]) * t_vals
        points_z = start[2] + (end[2] - start[2]) * t_vals - bulge_vals * 0.5

        for i in range(n_samples):
            p1 = np.array([points_x[i], points_y[i], points_z[i]], dtype=np.float32)
            p2 = np.array([points_x[i+1], points_y[i+1], points_z[i+1]], dtype=np.float32)
            if self.check_collision(p1, p2, radius, buffer):
                return True

        return False

    def find_safe_position(
        self, start: np.ndarray, base_end: np.ndarray, radius: float,
        rng: np.random.Generator, buffer: float, curvature: float,
        max_attempts: int = 16
    ) -> Tuple[np.ndarray, float]:
        """Find a position that doesn't collide."""
        start = np.asarray(start, dtype=np.float32)
        base_end = np.asarray(base_end, dtype=np.float32)

        if not self.check_curved_collision(start, base_end, radius, buffer, curvature):
            return base_end, 0.0

        dx = base_end[0] - start[0]
        dy = base_end[1] - start[1]
        dz = base_end[2] - start[2]

        horiz_dist = np.sqrt(dx*dx + dy*dy)
        if horiz_dist < 0.01:
            return base_end, 0.0

        base_angle = np.arctan2(dy, dx)
        step = 2 * np.pi / max_attempts

        for i in range(max_attempts):
            sign = 1 if i % 2 == 0 else -1
            offset = ((i + 2) // 2) * step * sign
            new_angle = base_angle + offset

            new_end = np.array([
                start[0] + horiz_dist * np.cos(new_angle),
                start[1] + horiz_dist * np.sin(new_angle),
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

        return base_end, 0.0


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class VascularPerfusionDishParams:
    """
    Parameters for collision-aware vascular network generation.

    Network Structure:
        inlets: Number of inlet points (1-25)
        levels: Branching depth levels (0-8)
        splits: Number of child branches per split (1-6)
        spread: Horizontal spread factor (0.1-0.8)
        ratio: Child/parent radius ratio (0.5-0.95), Murray's law = 0.79
        cone_angle: Branching cone angle in degrees (10-180)
        curvature: Branch curve intensity (0.0-1.0)
        bottom_height: Height from bottom where vessels terminate (0.02-1.0)

    Variation:
        radius_variation: Random radius variation (0-1)
        flip_chance: Chance of flipping branch direction (0-0.5)
        z_variation: Z-axis variation factor (0-0.5)
        angle_variation: Angle variation factor (0-0.5)
        collision_buffer: Extra buffer for collision detection (0-0.3)

    Behavior:
        even_spread: Use even 360° spread vs directional cone
        deterministic: Use straight grid-aligned channels
        tips_down: Terminal branches point downward
        seed: Random seed for reproducibility

    Geometry:
        resolution: Cylinder/sphere segments (6-32)
        outer_radius: Outer ring radius
        inner_radius: Inner scaffold radius
        height: Total height
        scaffold_height: Active scaffold height
        inlet_radius: Radius of inlet channels
        rim_height: Height of rim around dish edge (0-3mm)
        flip_upside_down: Flip scaffold 180° for export
    """
    # Network structure
    inlets: int = 4
    levels: int = 2
    splits: int = 2
    spread: float = 0.35
    ratio: float = 0.79
    cone_angle: float = 60.0
    curvature: float = 0.3
    bottom_height: float = 0.06

    # Variation
    radius_variation: float = 0.25
    flip_chance: float = 0.30
    z_variation: float = 0.35
    angle_variation: float = 0.40
    collision_buffer: float = 0.08

    # Behavior
    even_spread: bool = True
    deterministic: bool = False
    tips_down: bool = False
    seed: int = 42

    # Geometry
    resolution: int = 12
    outer_radius: float = 4.875
    inner_radius: float = 4.575
    height: float = 2.0
    scaffold_height: float = 1.92
    inlet_radius: float = 0.35
    rim_height: float = 0.2
    flip_upside_down: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'VascularPerfusionDishParams':
        """Create VascularPerfusionDishParams from a dictionary."""
        return cls(
            inlets=params.get('inlets', 4),
            levels=params.get('levels', 2),
            splits=params.get('splits', 2),
            spread=params.get('spread', 0.35),
            ratio=params.get('ratio', 0.79),
            cone_angle=params.get('cone_angle', 60.0),
            curvature=params.get('curvature', 0.3),
            bottom_height=params.get('bottom_height', 0.06),
            radius_variation=params.get('radius_variation', 0.25),
            flip_chance=params.get('flip_chance', 0.30),
            z_variation=params.get('z_variation', 0.35),
            angle_variation=params.get('angle_variation', 0.40),
            collision_buffer=params.get('collision_buffer', 0.08),
            even_spread=params.get('even_spread', True),
            deterministic=params.get('deterministic', False),
            tips_down=params.get('tips_down', False),
            seed=params.get('seed', 42),
            resolution=params.get('resolution', 12),
            outer_radius=params.get('outer_radius', 4.875),
            inner_radius=params.get('inner_radius', 4.575),
            height=params.get('height', 2.0),
            scaffold_height=params.get('scaffold_height', 1.92),
            inlet_radius=params.get('inlet_radius', 0.35),
            rim_height=params.get('rim_height', 0.2),
            flip_upside_down=params.get('flip_upside_down', False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'inlets': self.inlets,
            'levels': self.levels,
            'splits': self.splits,
            'spread': self.spread,
            'ratio': self.ratio,
            'cone_angle': self.cone_angle,
            'curvature': self.curvature,
            'bottom_height': self.bottom_height,
            'radius_variation': self.radius_variation,
            'flip_chance': self.flip_chance,
            'z_variation': self.z_variation,
            'angle_variation': self.angle_variation,
            'collision_buffer': self.collision_buffer,
            'even_spread': self.even_spread,
            'deterministic': self.deterministic,
            'tips_down': self.tips_down,
            'seed': self.seed,
            'resolution': self.resolution,
            'outer_radius': self.outer_radius,
            'inner_radius': self.inner_radius,
            'height': self.height,
            'scaffold_height': self.scaffold_height,
            'inlet_radius': self.inlet_radius,
            'rim_height': self.rim_height,
            'flip_upside_down': self.flip_upside_down,
        }


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _make_single_segment(
    m3d: Any,
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    r1: float, r2: float,
    resolution: int
) -> Optional[Any]:
    """Create a single tapered cylinder between two points."""
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 0.01:
        return None

    cyl = m3d.Manifold.cylinder(length, r2, r1, resolution)

    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, -dz) * 180 / np.pi
        azim = np.arctan2(-dy, -dx) * 180 / np.pi
        cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return cyl.translate([x2, y2, z2])


def _make_curved_branch(
    m3d: Any,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    r1: float, r2: float,
    start_dir: Tuple[float, float, float],
    end_dir: Tuple[float, float, float],
    curvature: float,
    resolution: int,
    deterministic: bool = False
) -> Tuple[List[Any], List[Tuple[float, float, float]]]:
    """
    Create a curved branch with junction spheres.
    Returns (geometry_segments, path_points).
    """
    x1, y1, z1 = start
    x2, y2, z2 = end

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    if dist < 0.02:
        return [], []

    path_points = [(x1, y1, z1), (x2, y2, z2)]
    segments = []

    if deterministic:
        # Straight segment for deterministic mode
        seg = _make_single_segment(m3d, x1, y1, z1, x2, y2, z2, r1, r2, resolution)
        if seg:
            segments.append(seg)
        sphere = m3d.Manifold.sphere(r2 * 1.05, resolution)
        segments.append(sphere.translate([x2, y2, z2]))
        return segments, path_points

    # Curved branch with midpoint
    ctrl_dist = dist * 0.4
    mid_t = 0.5
    mx = x1 + (x2 - x1) * mid_t + start_dir[0] * ctrl_dist * 0.3
    my = y1 + (y2 - y1) * mid_t + start_dir[1] * ctrl_dist * 0.3
    mz = z1 + (z2 - z1) * mid_t - curvature * dist * 0.1
    mr = r1 + (r2 - r1) * mid_t

    seg1 = _make_single_segment(m3d, x1, y1, z1, mx, my, mz, r1, mr, resolution)
    seg2 = _make_single_segment(m3d, mx, my, mz, x2, y2, z2, mr, r2, resolution)
    if seg1:
        segments.append(seg1)
    if seg2:
        segments.append(seg2)

    # Junction sphere at midpoint
    sphere = m3d.Manifold.sphere(mr * 1.05, resolution)
    segments.append(sphere.translate([mx, my, mz]))

    return segments, path_points


def _make_tips_down_branch(
    m3d: Any,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    r1: float, r2: float,
    start_dir: np.ndarray,
    resolution: int
) -> Tuple[List[Any], List[Tuple[float, float, float]]]:
    """Create branch that curves to point straight down at the tip."""
    x, y, z = start
    nx, ny, nz = end

    dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
    ctrl_dist = dist * 0.4

    p0 = np.array([x, y, z])
    p1 = p0 + start_dir * ctrl_dist
    p3 = np.array([nx, ny, nz])
    p2 = p3 + np.array([0, 0, dist * 0.35])

    segments = []
    path_points = []
    prev_pt = None
    prev_r = r1
    n_segs = 4

    for i in range(n_segs + 1):
        t = i / n_segs
        mt = 1 - t
        pt = mt**3 * p0 + 3*mt**2*t * p1 + 3*mt*t**2 * p2 + t**3 * p3
        cur_r = r1 + (r2 - r1) * t

        if prev_pt is not None:
            seg = _make_single_segment(
                m3d, prev_pt[0], prev_pt[1], prev_pt[2],
                pt[0], pt[1], pt[2], prev_r, cur_r, resolution
            )
            if seg:
                segments.append(seg)
            sph = m3d.Manifold.sphere(cur_r * 1.02, resolution)
            segments.append(sph.translate([pt[0], pt[1], pt[2]]))

        path_points.append((pt[0], pt[1], pt[2]))
        prev_pt = pt
        prev_r = cur_r

    return segments, path_points


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_vascular_perfusion_dish(
    params: VascularPerfusionDishParams,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[Any, Optional[Any], Any]:
    """
    Generate a collision-aware vascular network scaffold.

    Args:
        params: VascularPerfusionDishParams configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (scaffold_body, channels, result)
        - scaffold_body: The solid scaffold body
        - channels: Combined channel geometry
        - result: scaffold_body - channels (the final scaffold with channels cut out)
    """
    m3d = get_manifold_module()
    branch_tracker = BranchTracker()

    if progress_callback:
        progress_callback("Initializing scaffold...")

    # Geometry setup
    outer_r = params.outer_radius
    inner_r = params.inner_radius
    scaffold_h = params.scaffold_height
    height = scaffold_h + params.rim_height  # Rim extends above the body
    net_r = inner_r - 0.12
    net_top = scaffold_h
    net_bot = params.bottom_height
    cone_angle = np.radians(params.cone_angle)
    res = params.resolution

    # Build scaffold body
    outer = m3d.Manifold.cylinder(height, outer_r, outer_r, 48)
    inner_cut = m3d.Manifold.cylinder(height + 0.02, inner_r, inner_r, 48).translate([0, 0, -0.01])
    ring = outer - inner_cut
    body = m3d.Manifold.cylinder(scaffold_h, inner_r, inner_r, 48)
    scaffold_body = ring + body

    # Calculate inlet positions
    n = params.inlets
    if n == 1:
        inlet_pos = [(0.0, 0.0)]
    elif n <= 4:
        r = net_r * 0.45
        inlet_pos = [
            (r * np.cos(np.pi/4 + i * np.pi/2), r * np.sin(np.pi/4 + i * np.pi/2))
            for i in range(n)
        ]
    elif n == 9:
        sp = net_r * 0.5
        inlet_pos = [(i * sp, j * sp) for i in range(-1, 2) for j in range(-1, 2)]
    else:
        g = np.pi * (3 - np.sqrt(5))
        inlet_pos = [
            (net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.cos(i * g),
             net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.sin(i * g))
            for i in range(n)
        ]

    rng = np.random.default_rng(params.seed)
    channels: List[Any] = []

    # Randomize inlet positions (unless deterministic)
    if params.deterministic:
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

    if progress_callback:
        progress_callback(f"Generating {len(randomized_inlets)} inlet trees...")

    def branch(
        x: float, y: float, z: float, r: float, ang: float,
        remaining_levels: int, parent_dir: Optional[Tuple[float, float, float]]
    ):
        """Recursive branch generation with collision detection."""
        if r < 0.03 or z <= net_bot + 0.02:
            return

        # Calculate target z
        if remaining_levels > 0:
            z_step = (z - net_bot) / (remaining_levels + 1)
            if not params.deterministic:
                z_step *= rng.uniform(1 - params.z_variation, 1 + params.z_variation)
            nz = max(z - z_step, net_bot + 0.02)
        else:
            nz = net_bot + 0.02

        is_terminal = remaining_levels == 0

        # Calculate spread
        if params.deterministic:
            sp = params.spread if remaining_levels < params.levels else 0
            safe_ang = ang
        else:
            spread_var = params.z_variation * 0.6
            sp = params.spread * rng.uniform(1 - spread_var, 1 + spread_var) if remaining_levels < params.levels else 0
            safe_ang = ang + rng.uniform(-params.angle_variation, params.angle_variation)

        target_x = x + sp * np.cos(safe_ang)
        target_y = y + sp * np.sin(safe_ang)

        # Keep within bounds
        d = np.sqrt(target_x * target_x + target_y * target_y)
        if d > net_r - 0.1:
            scale = (net_r - 0.1) / d
            target_x *= scale
            target_y *= scale

        # Calculate child radius
        if params.deterministic:
            cr = r * params.ratio
        else:
            cr = r * params.ratio * rng.uniform(1 - params.radius_variation, 1 + params.radius_variation)

        # Collision detection
        avg_radius = (r + cr) / 2
        start_pos = np.array([x, y, z], dtype=np.float32)
        target_pos = np.array([target_x, target_y, nz], dtype=np.float32)

        safe_end, angle_offset = branch_tracker.find_safe_position(
            start_pos, target_pos, avg_radius, rng,
            params.collision_buffer, params.curvature
        )
        nx, ny = safe_end[0], safe_end[1]

        if angle_offset != 0:
            safe_ang = safe_ang + angle_offset

        # Direction vectors
        if parent_dir is None:
            start_dir = np.array([0.0, 0.0, -1.0])
        else:
            start_dir = np.array(parent_dir)

        outward = np.array([np.cos(safe_ang), np.sin(safe_ang), 0.0])
        horiz_blend = 0.4 + params.curvature * 0.5
        vert_blend = -0.6 + params.curvature * 0.3

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

        # Generate branch geometry
        if params.tips_down and is_terminal:
            segs, path_points = _make_tips_down_branch(
                m3d, (x, y, z), (nx, ny, nz),
                r, cr, start_dir, res
            )
        else:
            segs, path_points = _make_curved_branch(
                m3d, (x, y, z), (nx, ny, nz),
                r, cr, tuple(start_dir), tuple(end_dir),
                params.curvature, res, params.deterministic
            )

        channels.extend(segs)

        # Record path for collision detection
        if path_points and len(path_points) >= 2:
            branch_tracker.add_curved_branch(path_points, avg_radius)

        # Continue branching
        if remaining_levels > 0 and nz > net_bot + 0.05:
            junction = m3d.Manifold.sphere(cr * 1.15, res)
            channels.append(junction.translate([nx, ny, nz]))

            child_angles = []

            if params.deterministic:
                if params.splits == 1:
                    child_angles.append(safe_ang)
                else:
                    base_spacing = 2 * np.pi / params.splits
                    for i in range(params.splits):
                        child_angles.append(i * base_spacing)
            elif params.splits == 1:
                child_angles.append(safe_ang + rng.uniform(-params.angle_variation, params.angle_variation))
            elif params.even_spread:
                base_spacing = 2 * np.pi / params.splits
                start_rotation = rng.uniform(0, base_spacing)

                for i in range(params.splits):
                    base_angle = start_rotation + i * base_spacing
                    max_deviation = min(cone_angle, base_spacing * 0.4)
                    random_offset = rng.uniform(-max_deviation, max_deviation)

                    if rng.random() < params.flip_chance:
                        random_offset += rng.choice([-1, 1]) * base_spacing * 0.3

                    child_angles.append(base_angle + random_offset)
            else:
                total_spread = cone_angle * 2
                base_spacing = total_spread / params.splits

                for i in range(params.splits):
                    base_offset = -cone_angle + base_spacing * (i + 0.5)
                    random_offset = rng.uniform(
                        -base_spacing * params.angle_variation * 2,
                        base_spacing * params.angle_variation * 2
                    )

                    if rng.random() < params.flip_chance:
                        random_offset += rng.choice([-1, 1]) * base_spacing * (0.5 + params.flip_chance)

                    child_angles.append(safe_ang + base_offset + random_offset)

            for child_ang in child_angles:
                branch(nx, ny, nz, cr, child_ang, remaining_levels - 1, tuple(end_dir))

    # Process each inlet
    for idx, (ix, iy) in enumerate(randomized_inlets):
        if progress_callback and idx % max(1, len(randomized_inlets) // 5) == 0:
            progress_callback(f"Processing inlet {idx + 1}/{len(randomized_inlets)}...")

        # Create inlet port
        port = m3d.Manifold.cylinder(height - net_top + 0.03, params.inlet_radius, params.inlet_radius, res)
        port = port.translate([ix, iy, net_top - 0.01])
        channels.append(port)

        # Determine initial angle
        if params.deterministic:
            out_ang = np.arctan2(iy, ix) if (abs(ix) > 0.01 or abs(iy) > 0.01) else 0
        elif params.even_spread:
            out_ang = rng.uniform(0, 2 * np.pi)
        else:
            base_ang = np.arctan2(iy, ix) if (abs(ix) > 0.01 or abs(iy) > 0.01) else rng.uniform(0, 2*np.pi)
            out_ang = base_ang + rng.uniform(-0.4, 0.4)

        # Start branching
        start_r = params.inlet_radius * 1.1
        branch(ix, iy, net_top, start_r, out_ang, params.levels, None)

    if not channels:
        if params.flip_upside_down:
            scaffold_body = scaffold_body.rotate([180, 0, 0])
        return scaffold_body, None, scaffold_body

    if progress_callback:
        progress_callback(f"Combining {len(channels)} segments...")

    # Combine all channel segments
    combined = batch_union(channels, progress_callback=progress_callback)

    if progress_callback:
        progress_callback("Finalizing scaffold...")

    # Boolean subtract channels from scaffold
    result = scaffold_body - combined

    # Apply flip if requested
    if params.flip_upside_down:
        scaffold_body = scaffold_body.rotate([180, 0, 0])
        combined = combined.rotate([180, 0, 0])
        result = result.rotate([180, 0, 0])

    return scaffold_body, combined, result


def generate_vascular_perfusion_dish_from_dict(
    params: Dict[str, Any],
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[Any, Optional[Any], Any]:
    """
    Convenience function to generate collision vascular network from dictionary params.

    Args:
        params: Dictionary of parameters (see VascularPerfusionDishParams for keys)
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (scaffold_body, channels, result)
    """
    return generate_vascular_perfusion_dish(
        VascularPerfusionDishParams.from_dict(params),
        progress_callback
    )
