"""
Vascular network geometry generator.

Generates branching tree structures that mimic blood vessel networks.
Uses recursive branching with Murray's law for optimal flow characteristics.

Key features:
- Deterministic (grid) or organic (random) inlet positioning
- Configurable branching depth, spread, and splits
- Murray's law ratio (0.79) for biologically-optimal flow
- Bezier curve interpolation for smooth branches
- Optional tips_down mode for terminal branches pointing downward
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np

from .core import batch_union, get_manifold_module


@dataclass
class VascularParams:
    """
    Parameters for vascular network generation.

    Network Structure:
        inlets: Number of inlet points (1-25)
        levels: Branching depth levels (0-5)
        splits: Number of child branches per split (2-4)
        spread: Horizontal spread factor (0.0-1.0)
        ratio: Child/parent radius ratio (0.5-1.0), Murray's law = 0.79
        curvature: Branch curve intensity (0.0-1.0)

    Behavior:
        seed: Random seed for reproducibility
        tips_down: If True, terminal branches curve downward
        deterministic: If True, use grid pattern instead of random

    Geometry:
        resolution: Cylinder/sphere segments (8-32)
        outer_radius: Outer ring radius
        inner_radius: Inner scaffold radius
        height: Total height
        scaffold_height: Active scaffold height
        inlet_radius: Radius of inlet channels
    """
    # Network structure
    inlets: int = 4
    levels: int = 2
    splits: int = 2
    spread: float = 0.35
    ratio: float = 0.79  # Murray's law optimal
    curvature: float = 0.3

    # Behavior
    seed: int = 42
    tips_down: bool = False
    deterministic: bool = False

    # Geometry
    resolution: int = 12
    outer_radius: float = 4.875
    inner_radius: float = 4.575
    height: float = 2.0
    scaffold_height: float = 1.92
    inlet_radius: float = 0.35

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'VascularParams':
        """Create VascularParams from a dictionary."""
        return cls(
            inlets=params.get('inlets', 4),
            levels=params.get('levels', 2),
            splits=params.get('splits', 2),
            spread=params.get('spread', 0.35),
            ratio=params.get('ratio', 0.79),
            curvature=params.get('curvature', 0.3),
            seed=params.get('seed', 42),
            tips_down=params.get('tips_down', False),
            deterministic=params.get('deterministic', False),
            resolution=params.get('resolution', 12),
            outer_radius=params.get('outer_radius', 4.875),
            inner_radius=params.get('inner_radius', 4.575),
            height=params.get('height', 2.0),
            scaffold_height=params.get('scaffold_height', 1.92),
            inlet_radius=params.get('inlet_radius', 0.35),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'inlets': self.inlets,
            'levels': self.levels,
            'splits': self.splits,
            'spread': self.spread,
            'ratio': self.ratio,
            'curvature': self.curvature,
            'seed': self.seed,
            'tips_down': self.tips_down,
            'deterministic': self.deterministic,
            'resolution': self.resolution,
            'outer_radius': self.outer_radius,
            'inner_radius': self.inner_radius,
            'height': self.height,
            'scaffold_height': self.scaffold_height,
            'inlet_radius': self.inlet_radius,
        }


def make_cyl(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    r1: float, r2: float,
    resolution: int = 12
) -> Optional[Any]:
    """
    Create a cylinder (frustum) manifold between two 3D points.

    The cylinder is created along the z-axis and then rotated/translated
    to connect the two endpoint positions.

    Args:
        x1, y1, z1: Start point coordinates
        x2, y2, z2: End point coordinates
        r1: Radius at start point
        r2: Radius at end point
        resolution: Number of segments around circumference

    Returns:
        Manifold cylinder object, or None if length is too short

    Note:
        The manifold3d cylinder convention has the larger radius at z=0
        and smaller at z=length, so we pass (r2, r1) to cylinder().
    """
    m3d = get_manifold_module()

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 0.01:
        return None

    # Create cylinder (bottom radius, top radius)
    cyl = m3d.Manifold.cylinder(length, r2, r1, resolution)

    # Calculate rotation angles
    h = np.sqrt(dx*dx + dy*dy)  # horizontal distance
    if h > 0.001 or abs(dz) > 0.001:
        # Rotate to align with direction vector
        cyl = cyl.rotate([0, np.arctan2(h, -dz) * 180 / np.pi, 0])
        cyl = cyl.rotate([0, 0, np.arctan2(-dy, -dx) * 180 / np.pi])

    return cyl.translate([x2, y2, z2])


def _generate_inlet_positions(
    n: int,
    net_r: float,
    deterministic: bool,
    rng: np.random.Generator
) -> List[Tuple[float, float]]:
    """
    Generate inlet positions on the scaffold surface.

    For deterministic mode:
    - 1 inlet: center
    - 2-4 inlets: square pattern rotated 45 degrees
    - 9 inlets: 3x3 grid
    - other: Fibonacci spiral (sunflower pattern)

    For organic mode, positions are jittered from the grid.

    Args:
        n: Number of inlets
        net_r: Network radius (boundary for inlets)
        deterministic: If True, use exact grid positions
        rng: Random number generator

    Returns:
        List of (x, y) tuples for inlet positions
    """
    # Generate base positions
    if n == 1:
        inlet_pos = [(0.0, 0.0)]
    elif n <= 4:
        r = net_r * 0.45
        inlet_pos = [
            (r * np.cos(np.pi/4 + i * np.pi/2),
             r * np.sin(np.pi/4 + i * np.pi/2))
            for i in range(n)
        ]
    elif n == 9:
        sp = net_r * 0.5
        inlet_pos = [
            (i * sp, j * sp)
            for i in range(-1, 2) for j in range(-1, 2)
        ]
    else:
        # Fibonacci/sunflower spiral for even distribution
        g = np.pi * (3 - np.sqrt(5))  # golden angle
        inlet_pos = [
            (net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.cos(i * g),
             net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.sin(i * g))
            for i in range(n)
        ]

    if deterministic:
        return inlet_pos

    # Add organic jitter
    randomized = []
    for ix, iy in inlet_pos:
        j = 0.12 * rng.uniform(0.5, 1.5)
        a = rng.uniform(0, 2 * np.pi)
        nx = ix + j * np.cos(a)
        ny = iy + j * np.sin(a)
        # Clamp to boundary
        d = np.sqrt(nx*nx + ny*ny)
        if d > net_r - 0.5:
            nx, ny = nx * (net_r - 0.5) / d, ny * (net_r - 0.5) / d
        randomized.append((nx, ny))

    return randomized


def _create_branch_recursive(
    x: float, y: float, z: float,
    r: float,
    ang: float,
    remaining_levels: int,
    params: VascularParams,
    net_r: float,
    net_bot: float,
    rng: np.random.Generator,
    channels: List,
    parent_dir: Optional[Tuple[float, float, float]] = None
) -> None:
    """
    Recursively generate a branching tree structure.

    At each level:
    1. Calculate target position with spread
    2. Create curved path using Bezier interpolation
    3. Add spheres along path for smooth geometry
    4. Spawn child branches if levels remain

    Args:
        x, y, z: Current branch start position
        r: Current radius
        ang: Current angle direction
        remaining_levels: How many more levels to branch
        params: Generation parameters
        net_r: Network radius boundary
        net_bot: Bottom z boundary
        rng: Random number generator
        channels: List to append manifold segments to
        parent_dir: Direction vector from parent (for smooth curves)
    """
    m3d = get_manifold_module()

    # Termination conditions
    if r < 0.03 or z <= net_bot + 0.02:
        return

    is_terminal = remaining_levels == 0

    # Calculate z step
    if remaining_levels > 0:
        base_z_step = (z - net_bot) / (remaining_levels + 1)
        z_step = base_z_step * (1 if params.deterministic else rng.uniform(0.7, 1.3))
    else:
        z_step = z - net_bot - 0.02

    nz = max(z - z_step, net_bot + 0.02)

    # Calculate spread
    if remaining_levels < params.levels:
        sp = params.spread * (1 if params.deterministic else rng.uniform(0.7, 1.3))
    else:
        sp = 0

    # Calculate angle with optional jitter
    sa = ang if params.deterministic else ang + rng.uniform(-0.4, 0.4)

    # New position
    nx = x + sp * np.cos(sa)
    ny = y + sp * np.sin(sa)

    # Clamp to boundary
    d = np.sqrt(nx*nx + ny*ny)
    if d > net_r - 0.1:
        nx, ny = nx * (net_r - 0.1) / d, ny * (net_r - 0.1) / d

    # Child radius using ratio (with optional jitter)
    cr = r * params.ratio * (1 if params.deterministic else rng.uniform(0.85, 1.15))

    # Starting direction (from parent or straight down)
    sdir = np.array(parent_dir) if parent_dir else np.array([0., 0., -1.])

    # Generate branch geometry
    if params.tips_down and is_terminal:
        # Terminal branches curve downward
        dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
        p0 = np.array([x, y, z])
        p3 = np.array([nx, ny, nz])
        p1 = p0 + sdir * dist * 0.4
        p2 = p3 + np.array([0, 0, dist * 0.35])

        prev_pt = None
        prev_r = r
        for i in range(5):
            t = i / 4
            mt = 1 - t
            # Cubic Bezier interpolation
            pt = (mt**3 * p0 + 3*mt**2*t * p1 +
                  3*mt*t**2 * p2 + t**3 * p3)
            cur_r = r + (cr - r) * t

            if prev_pt is not None:
                seg = make_cyl(
                    prev_pt[0], prev_pt[1], prev_pt[2],
                    pt[0], pt[1], pt[2],
                    prev_r, cur_r,
                    params.resolution
                )
                if seg:
                    channels.append(seg)
                # Add sphere for smooth joint
                channels.append(
                    m3d.Manifold.sphere(cur_r * 1.02, params.resolution)
                    .translate([pt[0], pt[1], pt[2]])
                )

            prev_pt = pt
            prev_r = cur_r
    else:
        # Standard curved branch with Bezier
        dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
        if dist > 0.02:
            cd = dist * (0.4 + params.curvature * 0.5)

            # Control point 1: follow parent direction
            c1 = np.array([
                x + sdir[0]*cd,
                y + sdir[1]*cd,
                z + sdir[2]*cd - params.curvature*dist*0.15
            ])

            # Outward direction
            out = np.array([np.cos(sa), np.sin(sa), 0.])

            # End direction
            edir = np.array([out[0]*0.6, out[1]*0.6, -0.5])
            edir /= np.linalg.norm(edir)

            # Control point 2: smooth arrival
            c2 = np.array([
                nx - edir[0]*cd*0.8,
                ny - edir[1]*cd*0.8,
                nz + cd*0.3*params.curvature
            ])

            # Sample points along curve
            n_samples = max(8, int(dist/0.1)) + 1
            for i in range(n_samples):
                t = i / (n_samples - 1)
                mt = 1 - t
                mt2 = mt * mt
                mt3 = mt2 * mt

                # Cubic Bezier point
                c = (mt3 * np.array([x, y, z]) +
                     3*mt2*t * c1 +
                     3*mt*t*t * c2 +
                     t**3 * np.array([nx, ny, nz]))

                # Interpolate radius
                sphere_r = r + (cr - r) * t
                channels.append(
                    m3d.Manifold.sphere(sphere_r, params.resolution)
                    .translate([c[0], c[1], c[2]])
                )

    # Spawn children if levels remain
    if remaining_levels > 0 and nz > net_bot + 0.05:
        # Add junction sphere
        channels.append(
            m3d.Manifold.sphere(cr * 1.15, params.resolution)
            .translate([nx, ny, nz])
        )

        # Calculate child angles
        if params.deterministic:
            if params.splits == 1:
                child_angles = [sa]
            else:
                child_angles = [i * 2 * np.pi / params.splits
                               for i in range(params.splits)]
        else:
            base_step = 2 * np.pi / params.splits
            start_rotation = rng.uniform(0, base_step)
            child_angles = [
                start_rotation + i * base_step + rng.uniform(-0.3, 0.3)
                for i in range(params.splits)
            ]

        # Recurse for each child
        for child_ang in child_angles:
            out = np.array([np.cos(sa), np.sin(sa), 0.])
            ned = np.array([out[0]*0.6, out[1]*0.6, -0.5])
            ned /= np.linalg.norm(ned)

            _create_branch_recursive(
                nx, ny, nz, cr, child_ang,
                remaining_levels - 1,
                params, net_r, net_bot, rng,
                channels,
                parent_dir=tuple(ned)
            )


def generate_vascular_network(
    params: VascularParams,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[Any, Optional[Any], Any]:
    """
    Generate a complete vascular network scaffold.

    Creates a cylindrical scaffold body with internal branching
    vascular channels. The channels mimic blood vessel networks
    using recursive branching with Murray's law ratios.

    Args:
        params: VascularParams configuration object
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (scaffold_body, channels, result):
        - scaffold_body: The solid scaffold ring/base
        - channels: Combined channel manifold (or None if no channels)
        - result: scaffold_body - channels (the final perforated scaffold)

    Example:
        params = VascularParams(inlets=9, levels=3, spread=0.4)
        body, channels, result = generate_vascular_network(params)
        # Export result as STL
    """
    m3d = get_manifold_module()

    # Derived geometry parameters
    net_r = params.inner_radius - 0.12
    net_top = params.scaffold_height
    net_bot = 0.06

    if progress_callback:
        progress_callback("Building scaffold body...")

    # Create scaffold body
    outer = m3d.Manifold.cylinder(
        params.height, params.outer_radius, params.outer_radius, 48
    )
    inner_cut = m3d.Manifold.cylinder(
        params.height + 0.02, params.inner_radius, params.inner_radius, 48
    ).translate([0, 0, -0.01])
    body = m3d.Manifold.cylinder(
        params.scaffold_height, params.inner_radius, params.inner_radius, 48
    )
    scaffold_body = (outer - inner_cut) + body

    # Initialize random generator
    rng = np.random.default_rng(params.seed)

    # Generate inlet positions
    inlet_positions = _generate_inlet_positions(
        params.inlets, net_r, params.deterministic, rng
    )

    if progress_callback:
        progress_callback(f"Generating {len(inlet_positions)} inlet trees...")

    channels: List = []

    # Create branching trees from each inlet
    for ix, iy in inlet_positions:
        # Inlet cylinder (vertical channel from top)
        channels.append(
            m3d.Manifold.cylinder(
                params.height - net_top + 0.03,
                params.inlet_radius,
                params.inlet_radius,
                params.resolution
            ).translate([ix, iy, net_top - 0.01])
        )

        # Calculate initial outward angle
        if params.deterministic and (abs(ix) > 0.01 or abs(iy) > 0.01):
            out_ang = np.arctan2(iy, ix)
        else:
            out_ang = rng.uniform(0, 2 * np.pi)

        # Generate branching tree
        _create_branch_recursive(
            ix, iy, net_top,
            params.inlet_radius * 1.1,
            out_ang,
            params.levels,
            params, net_r, net_bot, rng,
            channels,
            parent_dir=None
        )

    if not channels:
        return scaffold_body, None, scaffold_body

    if progress_callback:
        progress_callback(f"Combining {len(channels)} segments...")

    # Combine all channel segments
    combined = batch_union(channels, progress_callback=progress_callback)

    if progress_callback:
        progress_callback("Finalizing scaffold...")

    # Boolean subtract channels from scaffold
    result = scaffold_body - combined

    return scaffold_body, combined, result


def generate_vascular_network_from_dict(
    params: Dict[str, Any],
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[Any, Optional[Any], Any]:
    """
    Convenience function to generate vascular network from dictionary params.

    Args:
        params: Dictionary of parameters (see VascularParams for keys)
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (scaffold_body, channels, result)
    """
    return generate_vascular_network(
        VascularParams.from_dict(params),
        progress_callback
    )
