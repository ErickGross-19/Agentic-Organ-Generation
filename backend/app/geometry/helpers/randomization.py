"""
Noise and perturbation functions for organic variation.

Provides utilities for adding controlled randomness to geometric positions,
sizes, and patterns to create more natural-looking scaffolds.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union
import numpy as np


def perturb_positions(
    positions: Union[List[Tuple[float, float]], List[Tuple[float, float, float]]],
    noise_amount: float,
    rng: Optional[np.random.Generator] = None
) -> Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """
    Apply random perturbation to 2D or 3D positions.

    Adds uniformly distributed noise in the range [-noise_amount, +noise_amount]
    to each coordinate.

    Args:
        positions: List of (x, y) or (x, y, z) positions
        noise_amount: Maximum perturbation distance per axis
        rng: Random number generator for reproducibility

    Returns:
        List of perturbed positions with same dimensionality as input

    Example:
        >>> positions = [(0, 0), (1, 0), (0.5, 0.866)]
        >>> perturbed = perturb_positions(positions, noise_amount=0.1)
    """
    if not positions:
        return positions

    if rng is None:
        rng = np.random.default_rng()

    is_3d = len(positions[0]) == 3
    perturbed = []

    for pos in positions:
        if is_3d:
            x, y, z = pos
            new_x = x + (rng.random() - 0.5) * 2 * noise_amount
            new_y = y + (rng.random() - 0.5) * 2 * noise_amount
            new_z = z + (rng.random() - 0.5) * 2 * noise_amount
            perturbed.append((new_x, new_y, new_z))
        else:
            x, y = pos
            new_x = x + (rng.random() - 0.5) * 2 * noise_amount
            new_y = y + (rng.random() - 0.5) * 2 * noise_amount
            perturbed.append((new_x, new_y))

    return perturbed


def perturb_unique_corners(
    unique_corners: List[Tuple[float, float]],
    lobule_centers: List[Tuple[float, float]],
    corner_to_lobules: Dict[int, List[int]],
    radius: float,
    corner_noise: float,
    angle_noise: float,
    stretch_noise: float,
    rng: np.random.Generator
) -> Dict[int, Tuple[float, float]]:
    """
    Apply random perturbations to unique corner positions.

    Designed for honeycomb patterns where corners are shared between
    adjacent hexagonal units. Applies three types of noise:
    - Radial noise: moves corner toward/away from cell centers
    - Tangential noise: moves corner perpendicular to radial direction
    - Global stretch: applies anisotropic stretch to entire pattern

    Args:
        unique_corners: List of (x, y) unique corner positions
        lobule_centers: List of (x, y) cell center positions
        corner_to_lobules: Dict mapping corner index to list of lobule indices
                           that share this corner
        radius: Base radius of hexagonal cells
        corner_noise: Amount of radial/tangential noise (0-1)
        angle_noise: Amount of angular perturbation (0-1)
        stretch_noise: Amount of global stretch deformation (0-1)
        rng: Random number generator

    Returns:
        Dict mapping corner index to perturbed (x, y) position

    Example:
        >>> rng = np.random.default_rng(42)
        >>> perturbed = perturb_unique_corners(
        ...     unique_corners, lobule_centers, corner_to_lobules,
        ...     radius=1.0, corner_noise=0.3, angle_noise=0.2,
        ...     stretch_noise=0.1, rng=rng
        ... )
    """
    perturbed = {}

    if lobule_centers:
        global_cx = np.mean([c[0] for c in lobule_centers])
        global_cy = np.mean([c[1] for c in lobule_centers])
    else:
        global_cx, global_cy = 0, 0

    if stretch_noise > 0:
        stretch_angle = rng.random() * np.pi
        stretch_amount = 1.0 + (rng.random() - 0.5) * stretch_noise * 0.6
    else:
        stretch_angle = 0
        stretch_amount = 1.0

    for corner_idx, (cx, cy) in enumerate(unique_corners):
        lobule_indices = corner_to_lobules[corner_idx]

        avg_lob_x = np.mean([lobule_centers[i][0] for i in lobule_indices])
        avg_lob_y = np.mean([lobule_centers[i][1] for i in lobule_indices])

        dx = cx - avg_lob_x
        dy = cy - avg_lob_y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0.01:
            dx, dy = dx/dist, dy/dist
        else:
            dx, dy = 1, 0

        perp_x, perp_y = -dy, dx

        radial_noise = (rng.random() - 0.5) * 2 * corner_noise * radius * 0.4
        tangent_noise = (rng.random() - 0.5) * 2 * corner_noise * radius * 0.4

        if angle_noise > 0:
            angle_offset = (rng.random() - 0.5) * 2 * angle_noise * np.radians(20)
            cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
            new_dx = dx * cos_a - dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            rot_offset_x = (new_dx - dx) * dist
            rot_offset_y = (new_dy - dy) * dist
        else:
            rot_offset_x, rot_offset_y = 0, 0

        new_x = cx + dx * radial_noise + perp_x * tangent_noise + rot_offset_x
        new_y = cy + dy * radial_noise + perp_y * tangent_noise + rot_offset_y

        if stretch_noise > 0:
            to_corner_x = new_x - global_cx
            to_corner_y = new_y - global_cy

            cos_s, sin_s = np.cos(stretch_angle), np.sin(stretch_angle)
            proj_along = to_corner_x * cos_s + to_corner_y * sin_s
            proj_perp = -to_corner_x * sin_s + to_corner_y * cos_s

            proj_along *= stretch_amount
            proj_perp /= stretch_amount

            new_x = global_cx + proj_along * cos_s - proj_perp * sin_s
            new_y = global_cy + proj_along * sin_s + proj_perp * cos_s

        perturbed[corner_idx] = (new_x, new_y)

    return perturbed


def generate_randomized_z_positions(
    count: int,
    height: float,
    randomness: float = 0.0,
    min_spacing: Optional[float] = None,
    seed: Optional[int] = None
) -> List[float]:
    """
    Generate Z positions with optional randomization.

    Creates evenly spaced positions with optional random perturbation.
    Useful for placing features at different heights along a scaffold.

    Args:
        count: Number of positions to generate
        height: Total height range (positions go from 0 to height)
        randomness: Amount of random perturbation (0-1), as fraction of spacing
        min_spacing: Minimum spacing between positions (optional)
        seed: Random seed for reproducibility

    Returns:
        List of Z positions sorted from lowest to highest

    Example:
        >>> # 5 evenly spaced positions with 10% randomness
        >>> z_positions = generate_randomized_z_positions(5, 10.0, randomness=0.1)
    """
    if count <= 0:
        return []
    if count == 1:
        return [height / 2]

    rng = np.random.default_rng(seed)

    # Generate base evenly-spaced positions
    spacing = height / (count + 1)
    positions = [spacing * (i + 1) for i in range(count)]

    if randomness > 0:
        max_offset = spacing * randomness * 0.5
        positions = [
            pos + (rng.random() - 0.5) * 2 * max_offset
            for pos in positions
        ]

    # Enforce minimum spacing if specified
    if min_spacing is not None and min_spacing > 0:
        positions = sorted(positions)
        adjusted = [positions[0]]
        for i in range(1, len(positions)):
            if positions[i] - adjusted[-1] < min_spacing:
                adjusted.append(adjusted[-1] + min_spacing)
            else:
                adjusted.append(positions[i])
        positions = adjusted

    # Clamp to valid range
    positions = [max(0, min(height, p)) for p in positions]

    return sorted(positions)


def apply_size_variance(
    base_size: float,
    variance: float,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Apply random variance to a base size value.

    Args:
        base_size: The nominal/base size value
        variance: Maximum fractional variance (0-1). A variance of 0.2 means
                  the result can vary by +/-20% from base_size.
        rng: Random number generator for reproducibility

    Returns:
        Size value with random variance applied

    Example:
        >>> # Radius that can vary +/-15% from 1.0
        >>> radius = apply_size_variance(1.0, 0.15)  # Returns 0.85 to 1.15
    """
    if variance <= 0:
        return base_size

    if rng is None:
        rng = np.random.default_rng()

    multiplier = 1.0 + (rng.random() - 0.5) * 2 * variance
    return base_size * multiplier


def apply_gaussian_noise(
    value: float,
    std_dev: float,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Apply Gaussian (normal distribution) noise to a value.

    Args:
        value: Base value
        std_dev: Standard deviation of the noise
        rng: Random number generator for reproducibility

    Returns:
        Value with Gaussian noise added

    Example:
        >>> # Position with 0.1mm standard deviation noise
        >>> pos = apply_gaussian_noise(5.0, 0.1)
    """
    if std_dev <= 0:
        return value

    if rng is None:
        rng = np.random.default_rng()

    return value + rng.normal(0, std_dev)


def jitter_grid(
    grid_positions: List[Tuple[float, float]],
    jitter_x: float,
    jitter_y: float,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[float, float]]:
    """
    Apply independent jitter to X and Y coordinates of grid positions.

    Unlike perturb_positions which uses uniform noise in all directions,
    this allows different noise amounts for each axis.

    Args:
        grid_positions: List of (x, y) positions
        jitter_x: Maximum X-axis perturbation
        jitter_y: Maximum Y-axis perturbation
        rng: Random number generator for reproducibility

    Returns:
        List of jittered (x, y) positions

    Example:
        >>> # Grid with more horizontal than vertical jitter
        >>> positions = jitter_grid(grid_points, jitter_x=0.5, jitter_y=0.1)
    """
    if rng is None:
        rng = np.random.default_rng()

    result = []
    for x, y in grid_positions:
        new_x = x + (rng.random() - 0.5) * 2 * jitter_x
        new_y = y + (rng.random() - 0.5) * 2 * jitter_y
        result.append((new_x, new_y))

    return result


def create_noise_field(
    width: int,
    height: int,
    scale: float = 1.0,
    octaves: int = 4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a 2D noise field using value noise with multiple octaves.

    This creates a Perlin-like noise field useful for organic variations.

    Args:
        width: Width of noise field in samples
        height: Height of noise field in samples
        scale: Base frequency of the noise
        octaves: Number of noise octaves to combine
        seed: Random seed for reproducibility

    Returns:
        2D numpy array of noise values in range [0, 1]

    Example:
        >>> # Create 100x100 noise field
        >>> noise = create_noise_field(100, 100, scale=0.1, octaves=4)
        >>> # Use noise[y, x] to get value at position
    """
    rng = np.random.default_rng(seed)

    noise = np.zeros((height, width))
    amplitude = 1.0
    max_value = 0.0

    for _ in range(octaves):
        # Generate random gradient points
        random_field = rng.random((height, width))

        # Apply Gaussian smoothing for spatial correlation
        from scipy.ndimage import gaussian_filter
        smooth_field = gaussian_filter(random_field, sigma=1/scale)

        noise += amplitude * smooth_field
        max_value += amplitude
        amplitude *= 0.5
        scale *= 2

    # Normalize to [0, 1]
    noise = noise / max_value

    return noise
