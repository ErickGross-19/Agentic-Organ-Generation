"""
Procedural tissue point generation helpers.

Includes Poisson disk sampling and organ-specific distribution generators.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.domain import DomainSpec


def generate_poisson_disk(
    bounds: Tuple[float, ...],
    min_distance: float,
    max_points: int,
    rng: np.random.Generator,
    max_attempts: int = 30,
    domain: Optional["DomainSpec"] = None,
) -> np.ndarray:
    """
    Generate Poisson disk-sampled points in 3D.

    Produces a blue-noise distribution where no two points are closer
    than min_distance.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds[:6]

    cell_size = min_distance / np.sqrt(3)
    nx = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_size)))
    nz = max(1, int(np.ceil((z_max - z_min) / cell_size)))

    grid: Dict[Tuple[int, int, int], int] = {}
    points = []
    active = []

    first = np.array([
        rng.uniform(x_min, x_max),
        rng.uniform(y_min, y_max),
        rng.uniform(z_min, z_max),
    ])

    if domain is not None:
        from ..core.types import Point3D
        for _ in range(100):
            if domain.contains(Point3D(float(first[0]), float(first[1]), float(first[2]))):
                break
            first = np.array([
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                rng.uniform(z_min, z_max),
            ])

    points.append(first)
    active.append(0)
    ci, cj, ck = _point_to_cell(first, x_min, y_min, z_min, cell_size)
    grid[(ci, cj, ck)] = 0

    while active and len(points) < max_points:
        idx = rng.integers(0, len(active))
        base_idx = active[idx]
        base_pt = points[base_idx]

        found = False
        for _ in range(max_attempts):
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            r = rng.uniform(min_distance, 2 * min_distance)
            candidate = base_pt + direction * r

            if (candidate[0] < x_min or candidate[0] > x_max or
                    candidate[1] < y_min or candidate[1] > y_max or
                    candidate[2] < z_min or candidate[2] > z_max):
                continue

            if domain is not None:
                from ..core.types import Point3D
                if not domain.contains(Point3D(float(candidate[0]), float(candidate[1]), float(candidate[2]))):
                    continue

            ci, cj, ck = _point_to_cell(candidate, x_min, y_min, z_min, cell_size)

            too_close = False
            for di in range(-2, 3):
                if too_close:
                    break
                for dj in range(-2, 3):
                    if too_close:
                        break
                    for dk in range(-2, 3):
                        ni, nj, nk = ci + di, cj + dj, ck + dk
                        if (ni, nj, nk) in grid:
                            other = points[grid[(ni, nj, nk)]]
                            if np.linalg.norm(candidate - other) < min_distance:
                                too_close = True
                                break

            if not too_close:
                new_idx = len(points)
                points.append(candidate)
                active.append(new_idx)
                grid[(ci, cj, ck)] = new_idx
                found = True
                break

        if not found:
            active.pop(idx)

    return np.array(points) if points else np.empty((0, 3))


def _point_to_cell(
    point: np.ndarray,
    x_min: float,
    y_min: float,
    z_min: float,
    cell_size: float,
) -> Tuple[int, int, int]:
    return (
        int((point[0] - x_min) / cell_size),
        int((point[1] - y_min) / cell_size),
        int((point[2] - z_min) / cell_size),
    )


def generate_liver_lobule_pattern(
    domain: "DomainSpec",
    n_lobules: int = 50,
    points_per_lobule: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate liver-like hexagonal lobule tissue pattern."""
    rng = np.random.default_rng(seed)
    bounds = domain.get_bounds()

    lobule_centers = generate_poisson_disk(
        bounds,
        min_distance=max(
            (bounds[1] - bounds[0]) / (n_lobules ** (1 / 3) + 1),
            0.001,
        ),
        max_points=n_lobules,
        rng=rng,
        domain=domain,
    )

    all_points = []
    lobule_radius = max(
        (bounds[1] - bounds[0]) / (n_lobules ** (1 / 3) * 2),
        0.0005,
    )

    for center in lobule_centers:
        for _ in range(points_per_lobule * 3):
            if len(all_points) >= n_lobules * points_per_lobule:
                break
            offset = rng.normal(scale=lobule_radius * 0.5, size=3)
            pt = center + offset
            from ..core.types import Point3D
            if domain.contains(Point3D(float(pt[0]), float(pt[1]), float(pt[2]))):
                all_points.append(pt)

    return np.array(all_points) if all_points else np.empty((0, 3))


def generate_lung_bronchiole_pattern(
    domain: "DomainSpec",
    n_clusters: int = 30,
    points_per_cluster: int = 30,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate lung-like clustered airway tissue pattern."""
    rng = np.random.default_rng(seed)
    bounds = domain.get_bounds()

    cluster_centers = generate_poisson_disk(
        bounds,
        min_distance=max(
            (bounds[1] - bounds[0]) / (n_clusters ** (1 / 3) + 1),
            0.001,
        ),
        max_points=n_clusters,
        rng=rng,
        domain=domain,
    )

    all_points = []
    for center in cluster_centers:
        sigma = max((bounds[1] - bounds[0]) / (n_clusters ** (1 / 3) * 3), 0.0005)
        for _ in range(points_per_cluster * 3):
            if len(all_points) >= n_clusters * points_per_cluster:
                break
            offset = rng.normal(scale=sigma, size=3)
            pt = center + offset
            from ..core.types import Point3D
            if domain.contains(Point3D(float(pt[0]), float(pt[1]), float(pt[2]))):
                all_points.append(pt)

    return np.array(all_points) if all_points else np.empty((0, 3))


def generate_kidney_nephron_pattern(
    domain: "DomainSpec",
    n_nephrons: int = 40,
    points_per_nephron: int = 25,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate kidney-like layered nephron tissue pattern."""
    rng = np.random.default_rng(seed)
    bounds = domain.get_bounds()

    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ])

    max_r = min(
        (bounds[1] - bounds[0]) / 2,
        (bounds[3] - bounds[2]) / 2,
        (bounds[5] - bounds[4]) / 2,
    )

    cortex_inner = max_r * 0.5
    cortex_outer = max_r * 0.95

    all_points = []
    for _ in range(n_nephrons * points_per_nephron * 3):
        if len(all_points) >= n_nephrons * points_per_nephron:
            break
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        r = rng.uniform(cortex_inner, cortex_outer)
        pt = center + direction * r
        from ..core.types import Point3D
        if domain.contains(Point3D(float(pt[0]), float(pt[1]), float(pt[2]))):
            all_points.append(pt)

    return np.array(all_points) if all_points else np.empty((0, 3))
