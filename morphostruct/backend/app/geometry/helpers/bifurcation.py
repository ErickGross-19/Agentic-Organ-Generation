"""
Functions for Y-junctions and branching structures.

Provides utilities for creating bifurcations (1->2 splits), trifurcations (1->3),
and recursive branching networks for vascular and airway structures.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    m3d = None
    HAS_MANIFOLD = False


def _get_manifold():
    """Get manifold3d module, raising ImportError if not available."""
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )
    return m3d


def create_bifurcation(
    trunk_start: Tuple[float, float, float],
    split_point: Tuple[float, float, float],
    branch_ends: List[Tuple[float, float, float]],
    trunk_radius: float,
    branch_radius: float,
    n_sides: int = 16
) -> Optional["m3d.Manifold"]:
    """
    Create a Y-junction bifurcation where one tube splits into multiple.

    Creates a smooth transition from a single trunk tube to multiple branch
    tubes. Uses union of cones/cylinders with junction spheres for smooth
    connections.

    Args:
        trunk_start: (x, y, z) starting point of the trunk
        split_point: (x, y, z) point where the split occurs
        branch_ends: List of (x, y, z) end points for each branch
        trunk_radius: Radius of the trunk section
        branch_radius: Radius of each branch (typically smaller than trunk)
        n_sides: Number of sides for cylinders (higher = smoother)

    Returns:
        Manifold representing the bifurcation, or None if invalid

    Example:
        >>> # Create a Y-junction splitting upward
        >>> bif = create_bifurcation(
        ...     trunk_start=(0, 0, 0),
        ...     split_point=(0, 0, 5),
        ...     branch_ends=[(2, 0, 8), (-2, 0, 8)],
        ...     trunk_radius=0.5,
        ...     branch_radius=0.35
        ... )
    """
    m3d = _get_manifold()
    from .mesh_utils import tree_union

    n_branches = len(branch_ends)
    if n_branches < 2:
        return None

    solids = []

    # Create trunk section
    trunk_vec = np.array(split_point) - np.array(trunk_start)
    trunk_len = np.linalg.norm(trunk_vec)
    if trunk_len < 1e-6:
        return None

    trunk_dir = trunk_vec / trunk_len

    # Create trunk cylinder
    trunk_cyl = m3d.Manifold.cylinder(trunk_len, trunk_radius, trunk_radius, n_sides)

    # Rotate to align with direction
    z_axis = np.array([0, 0, 1])
    dot = np.dot(z_axis, trunk_dir)

    if dot < -0.9999:
        trunk_cyl = trunk_cyl.rotate([180, 0, 0])
    elif dot < 0.9999:
        axis = np.cross(z_axis, trunk_dir)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-10:
            axis = axis / axis_len
            angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi

            c = np.cos(np.radians(angle))
            s = np.sin(np.radians(angle))
            ux, uy, uz = axis

            rot_mat = np.array([
                [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s, 0],
                [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s, 0],
                [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c), 0]
            ])
            trunk_cyl = trunk_cyl.transform(rot_mat)

    trunk_cyl = trunk_cyl.translate(list(trunk_start))
    solids.append(trunk_cyl)

    # Junction sphere at split point
    junction_sphere = m3d.Manifold.sphere(trunk_radius * 1.1, n_sides)
    junction_sphere = junction_sphere.translate(list(split_point))
    solids.append(junction_sphere)

    # Create branches
    for branch_end in branch_ends:
        branch_vec = np.array(branch_end) - np.array(split_point)
        branch_len = np.linalg.norm(branch_vec)
        if branch_len < 1e-6:
            continue

        branch_dir = branch_vec / branch_len

        # Tapered cylinder: starts at trunk_radius * 0.8, ends at branch_radius
        branch_cyl = m3d.Manifold.cylinder(
            branch_len,
            trunk_radius * 0.8,  # Larger at junction
            branch_radius,       # Smaller at end
            n_sides
        )

        # Rotate to align with direction
        dot = np.dot(z_axis, branch_dir)

        if dot < -0.9999:
            branch_cyl = branch_cyl.rotate([180, 0, 0])
        elif dot < 0.9999:
            axis = np.cross(z_axis, branch_dir)
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-10:
                axis = axis / axis_len
                angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi

                c = np.cos(np.radians(angle))
                s = np.sin(np.radians(angle))
                ux, uy, uz = axis

                rot_mat = np.array([
                    [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s, 0],
                    [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s, 0],
                    [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c), 0]
                ])
                branch_cyl = branch_cyl.transform(rot_mat)

        branch_cyl = branch_cyl.translate(list(split_point))
        solids.append(branch_cyl)

    if not solids:
        return None

    return tree_union(solids)


def create_trifurcation(
    trunk_start: Tuple[float, float, float],
    split_point: Tuple[float, float, float],
    branch_ends: List[Tuple[float, float, float]],
    trunk_radius: float,
    branch_radius: float,
    n_sides: int = 16
) -> Optional["m3d.Manifold"]:
    """
    Create a trifurcation where one tube splits into three branches.

    This is a convenience wrapper around create_bifurcation that expects
    exactly 3 branch ends. Useful for creating Y-shaped airways or vessels.

    Args:
        trunk_start: (x, y, z) starting point of the trunk
        split_point: (x, y, z) point where the split occurs
        branch_ends: List of exactly 3 (x, y, z) end points
        trunk_radius: Radius of the trunk section
        branch_radius: Radius of each branch
        n_sides: Number of sides for cylinders

    Returns:
        Manifold representing the trifurcation, or None if invalid

    Raises:
        ValueError: If branch_ends does not contain exactly 3 points
    """
    if len(branch_ends) != 3:
        raise ValueError(f"Trifurcation requires exactly 3 branch ends, got {len(branch_ends)}")

    return create_bifurcation(
        trunk_start, split_point, branch_ends,
        trunk_radius, branch_radius, n_sides
    )


def create_branching_network(
    root_pos: Tuple[float, float, float],
    root_radius: float,
    generations: int,
    branch_angle: float,
    direction: Tuple[float, float, float],
    length_ratio: float = 0.7,
    radius_ratio: float = 0.75,
    n_sides: int = 16,
    asymmetry: float = 0.0,
    seed: Optional[int] = None
) -> Optional["m3d.Manifold"]:
    """
    Create a recursive branching network (binary tree structure).

    Generates a fractal-like branching pattern where each generation
    splits into two branches. Follows Murray's law approximation for
    biological realism when radius_ratio ~= 0.79 (cube-root of 0.5).

    Args:
        root_pos: Starting position (x, y, z)
        root_radius: Radius of the root (first generation) trunk
        generations: Number of branching generations (1-6 recommended)
        branch_angle: Angle between branches in degrees
        direction: Initial direction vector (will be normalized)
        length_ratio: Length multiplier per generation (default 0.7)
        radius_ratio: Radius multiplier per generation (default 0.75)
        n_sides: Number of sides for cylinders
        asymmetry: Random variation in angles (0-1, default 0 = symmetric)
        seed: Random seed for reproducible asymmetry

    Returns:
        Manifold representing the entire branching network

    Example:
        >>> # Create a 4-generation binary tree
        >>> tree = create_branching_network(
        ...     root_pos=(0, 0, 0),
        ...     root_radius=1.0,
        ...     generations=4,
        ...     branch_angle=35,
        ...     direction=(0, 0, 1)
        ... )
    """
    m3d = _get_manifold()
    from .mesh_utils import tree_union

    if generations < 1:
        return None

    rng = np.random.default_rng(seed)

    # Normalize direction
    dir_vec = np.array(direction, dtype=float)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-10)

    # Calculate initial segment length based on radius
    base_length = root_radius * 8  # Length proportional to radius

    solids = []

    def add_branch(pos, direction, radius, length, gen):
        """Recursively add branches."""
        if gen > generations:
            return

        # Calculate end point
        end_pos = pos + direction * length

        # Create this segment
        cyl = m3d.Manifold.cylinder(length, radius, radius, n_sides)

        # Rotate to align with direction
        z_axis = np.array([0, 0, 1])
        dot = np.dot(z_axis, direction)

        if dot < -0.9999:
            cyl = cyl.rotate([180, 0, 0])
        elif dot < 0.9999:
            axis = np.cross(z_axis, direction)
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-10:
                axis = axis / axis_len
                angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi

                c = np.cos(np.radians(angle))
                s = np.sin(np.radians(angle))
                ux, uy, uz = axis

                rot_mat = np.array([
                    [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s, 0],
                    [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s, 0],
                    [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c), 0]
                ])
                cyl = cyl.transform(rot_mat)

        cyl = cyl.translate(list(pos))
        solids.append(cyl)

        # Add junction sphere
        sphere = m3d.Manifold.sphere(radius * 0.99, n_sides)
        sphere = sphere.translate(list(end_pos))
        solids.append(sphere)

        if gen < generations:
            # Calculate new parameters for child branches
            new_radius = radius * radius_ratio
            new_length = length * length_ratio

            # Calculate perpendicular vectors for rotation
            if abs(direction[2]) < 0.9:
                perp1 = np.cross(direction, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(direction, np.array([1, 0, 0]))
            perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)
            perp2 = np.cross(direction, perp1)

            # Add asymmetry
            angle1 = branch_angle + asymmetry * (rng.random() - 0.5) * branch_angle
            angle2 = branch_angle + asymmetry * (rng.random() - 0.5) * branch_angle

            # Rotate for two branches (on opposite sides)
            rad1 = np.radians(angle1)
            rad2 = np.radians(-angle2)

            # Branch 1
            dir1 = (direction * np.cos(rad1) + perp1 * np.sin(rad1))
            dir1 = dir1 / (np.linalg.norm(dir1) + 1e-10)

            # Branch 2
            dir2 = (direction * np.cos(rad2) + perp1 * np.sin(rad2))
            dir2 = dir2 / (np.linalg.norm(dir2) + 1e-10)

            # Recursively add child branches
            add_branch(end_pos, dir1, new_radius, new_length, gen + 1)
            add_branch(end_pos, dir2, new_radius, new_length, gen + 1)

    # Start the recursion
    add_branch(np.array(root_pos), dir_vec, root_radius, base_length, 1)

    if not solids:
        return None

    return tree_union(solids)


def create_branching_angles(
    num_branches: int,
    total_angle: float = 360.0,
    asymmetry: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> List[float]:
    """
    Calculate branch angles for radial distribution around a split point.

    Args:
        num_branches: Number of branches to distribute
        total_angle: Total angle to distribute across (degrees)
        asymmetry: Random variation factor (0-1)
        rng: Random number generator for reproducibility

    Returns:
        List of angles in degrees for each branch
    """
    if rng is None:
        rng = np.random.default_rng()

    base_spacing = total_angle / num_branches
    angles = []

    for i in range(num_branches):
        angle = i * base_spacing
        if asymmetry > 0:
            angle += asymmetry * (rng.random() - 0.5) * base_spacing
        angles.append(angle)

    return angles
