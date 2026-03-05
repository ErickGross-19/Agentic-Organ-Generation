"""
Hepatic lobule scaffold generator.

Creates hexagonal prism structures mimicking liver lobule architecture with:
- Hexagonal unit cells in honeycomb arrangement
- Portal triads at corners (portal vein, hepatic artery)
- Central vein through the center
- Optional sinusoidal channels connecting periphery to center
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class HepaticLobuleParams:
    """
    Parameters for hepatic lobule scaffold generation.

    Creates hexagonal prism structures mimicking liver lobule architecture with
    portal triads at corners, central vein, and optional sinusoidal channels.

    Attributes:
        Basic Geometry:
            num_lobules: Number of lobules in honeycomb arrangement (1-19)
            lobule_radius: Outer radius of each hexagonal lobule (mm)
            lobule_height: Height of lobule prism (mm)
            wall_thickness: Thickness of hexagonal walls (mm)
            resolution: Mesh resolution for cylinders (segments)

        Central Vein:
            central_vein_radius: Radius of central vein (mm)
            cv_entrance_length: Length of central vein entrance channels (mm)
            cv_entrance_radius: Radius of entrance channels (mm)
            cv_entrance_count: Number of entrance channels per corner direction
            cv_entrance_z_randomness: Randomness in Z positioning (0-1)
            cv_entrance_min_spacing: Minimum spacing between entrances (mm)

        Portal Triad:
            portal_vein_radius: Radius of portal veins at corners (mm)
            hepatic_artery_radius: Radius of hepatic arteries (mm)
            bile_duct_radius: Radius of bile ducts (mm)
            show_bile_ducts: Whether to include bile ducts
            triad_wall_distance: Distance of triad from corner (mm)
            entrance_length: Length of portal entrance channels (mm)
            entrance_radius: Radius of portal entrance channels (mm)
            entrance_count: Number of portal entrances per triad
            entrance_z_randomness: Randomness in Z positioning (0-1)
            entrance_min_spacing: Minimum spacing between entrances (mm)

        Sinusoids:
            show_sinusoids: Whether to generate sinusoidal channels
            sinusoid_radius: Radius of sinusoid channels (mm)
            sinusoid_count: Number of sinusoids per corner
            sinusoid_levels: Number of Z-levels for sinusoids

        Organic Variation:
            corner_noise: Random perturbation of corner positions (0-1)
            angle_noise: Random angular perturbation (0-1)
            stretch_noise: Random stretch deformation (0-1)
            size_variance: Variation in lobule sizes (0-1)
            edge_curve: Curvature of hexagon edges (0-1)
            seed: Random seed for reproducibility

        Collectors:
            show_hepatic_collector: Show hepatic vein collector network
            show_portal_collector: Show portal vein collector network
            hepatic_collector_height: Height of hepatic collector above lobules (mm)
            portal_collector_height: Height of portal collector below lobules (mm)
    """
    # === Basic Geometry ===
    num_lobules: int = 7
    lobule_radius: float = 1.5  # mm
    lobule_height: float = 3.0  # mm
    wall_thickness: float = 0.1  # mm
    resolution: int = 8

    # === Central Vein ===
    central_vein_radius: float = 0.15  # mm
    cv_entrance_length: float = 0.25  # mm
    cv_entrance_radius: float = 0.04  # mm
    cv_entrance_count: int = 5
    cv_entrance_z_randomness: float = 0.0  # 0 = evenly spaced, 1 = fully random
    cv_entrance_min_spacing: float = 0.1  # mm

    # === Portal Triad ===
    portal_vein_radius: float = 0.12  # mm
    hepatic_artery_radius: float = 0.06  # mm
    bile_duct_radius: float = 0.05  # mm
    show_bile_ducts: bool = True
    triad_wall_distance: float = 0.0  # mm
    entrance_length: float = 0.25  # mm
    entrance_radius: float = 0.04  # mm
    entrance_count: int = 5
    entrance_z_randomness: float = 0.0  # 0 = evenly spaced, 1 = fully random
    entrance_min_spacing: float = 0.1  # mm

    # === Sinusoids ===
    show_sinusoids: bool = False
    sinusoid_radius: float = 0.012  # mm (~23.5µm diameter literature value)
    sinusoid_count: int = 6
    sinusoid_levels: int = 3

    # === Organic Variation ===
    corner_noise: float = 0.0  # 0-1, perturbation of corner positions
    angle_noise: float = 0.0  # 0-1, angular perturbation
    stretch_noise: float = 0.0  # 0-1, stretch deformation
    size_variance: float = 0.0  # 0-1, variation in lobule sizes
    edge_curve: float = 0.0  # 0-1, curvature of hex edges
    seed: int = 42  # random seed

    # === Collectors ===
    show_hepatic_collector: bool = True
    show_portal_collector: bool = True
    hepatic_collector_height: float = 2.0  # mm
    portal_collector_height: float = 2.0  # mm


# =============================================================================
# Helper Functions
# =============================================================================

def tree_union(manifolds: list) -> m3d.Manifold | None:
    """Use manifold3d's native batch_boolean for faster union."""
    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]
    return m3d.Manifold.batch_boolean(manifolds, m3d.OpType.Add)


def generate_honeycomb_centers(num_lobules: int, radius: float) -> list[tuple[float, float]]:
    """
    Generate center positions for hexagons in a honeycomb pattern.
    Adjacent hexagons share edges (and thus 2 corners each).
    Starts with center at origin, then adds neighbors in rings.
    Returns list of (x, y) center positions.
    """
    if num_lobules <= 0:
        return []
    if num_lobules == 1:
        return [(0.0, 0.0)]

    hex_spacing = radius * np.sqrt(3)
    centers = [(0.0, 0.0)]
    added = {(0, 0)}
    axial_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    queue = [(0, 0)]
    while len(centers) < num_lobules and queue:
        q, r = queue.pop(0)
        for dq, dr in axial_dirs:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in added and len(centers) < num_lobules:
                added.add((nq, nr))
                x = hex_spacing * (nq + nr * 0.5)
                y = hex_spacing * nr * (np.sqrt(3) / 2)
                centers.append((x, y))
                queue.append((nq, nr))

    return centers[:num_lobules]


def get_corner_positions(center_x: float, center_y: float, radius: float,
                         rotation_offset: float = None) -> list[tuple[float, float]]:
    """Get the 6 corner positions of a hexagon at given center."""
    if rotation_offset is None:
        rotation_offset = np.radians(30)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + rotation_offset
    return [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]


def find_unique_corners(lobule_centers: list[tuple[float, float]], radius: float,
                        tolerance: float = 0.05) -> tuple[list[tuple[float, float]], dict]:
    """
    Find all unique corner positions across all lobules.
    Returns:
    - unique_corners: list of (x, y) unique corner positions
    - corner_to_lobules: dict mapping corner index to list of lobule indices that share it
    """
    all_corners = []

    for lobule_idx, (cx, cy) in enumerate(lobule_centers):
        corners = get_corner_positions(cx, cy, radius)
        for corner_x, corner_y in corners:
            all_corners.append((corner_x, corner_y, lobule_idx))

    unique_corners = []
    corner_to_lobules = {}

    for cx, cy, lobule_idx in all_corners:
        found = False
        for unique_idx, (ux, uy) in enumerate(unique_corners):
            if abs(cx - ux) < tolerance and abs(cy - uy) < tolerance:
                if lobule_idx not in corner_to_lobules[unique_idx]:
                    corner_to_lobules[unique_idx].append(lobule_idx)
                found = True
                break

        if not found:
            unique_idx = len(unique_corners)
            unique_corners.append((cx, cy))
            corner_to_lobules[unique_idx] = [lobule_idx]

    return unique_corners, corner_to_lobules


def perturb_unique_corners(unique_corners: list[tuple[float, float]],
                           lobule_centers: list[tuple[float, float]],
                           corner_to_lobules: dict, radius: float,
                           corner_noise: float, angle_noise: float,
                           stretch_noise: float, rng) -> dict:
    """
    Apply random perturbations to unique corner positions.
    Returns dict mapping corner index to perturbed (x, y) position.
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


def get_lobule_corners_from_unique(lobule_idx: int, lobule_center: tuple[float, float],
                                   unique_corners: list[tuple[float, float]],
                                   corner_to_lobules: dict,
                                   perturbed_corners: dict, radius: float,
                                   stretch_noise: float, rng) -> list:
    """Get the 6 corner positions for a specific lobule, using perturbed unique corners."""
    lx, ly = lobule_center

    lobule_corners = []
    for corner_idx, lobules in corner_to_lobules.items():
        if lobule_idx in lobules:
            lobule_corners.append((corner_idx, perturbed_corners[corner_idx]))

    def angle_from_center(item):
        _, (cx, cy) = item
        return np.arctan2(cy - ly, cx - lx)

    lobule_corners.sort(key=angle_from_center)
    return lobule_corners


def create_rounded_polygon_points(vertices: np.ndarray, fillet_radius: float,
                                   segments_per_corner: int = 8) -> list[list[float]]:
    """
    Create a rounded polygon by adding arc fillets at each corner.

    Args:
        vertices: Nx2 array of polygon vertices in order
        fillet_radius: Radius of the fillet arc at each corner
        segments_per_corner: Number of arc segments per corner

    Returns:
        List of [x, y] points forming the rounded polygon
    """
    n = len(vertices)
    if n < 3 or fillet_radius <= 0:
        return vertices.tolist()

    rounded_points = []

    for i in range(n):
        # Current corner and adjacent vertices
        prev_v = vertices[(i - 1) % n]
        curr_v = vertices[i]
        next_v = vertices[(i + 1) % n]

        # Edge vectors from current corner
        v1 = prev_v - curr_v
        v2 = next_v - curr_v

        # Normalize
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-10 or len2 < 1e-10:
            rounded_points.append(curr_v.tolist())
            continue

        v1_norm = v1 / len1
        v2_norm = v2 / len2

        # Calculate the angle between edges
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle = np.arccos(cos_angle)

        # Half-angle for tangent distance calculation
        half_angle = angle / 2

        # Avoid division by zero for nearly straight edges
        if abs(np.sin(half_angle)) < 1e-10:
            rounded_points.append(curr_v.tolist())
            continue

        # Distance from corner to tangent points
        tangent_dist = fillet_radius / np.tan(half_angle)

        # Clamp tangent distance to avoid overlapping with adjacent corners
        max_dist = min(len1, len2) * 0.45
        if tangent_dist > max_dist:
            tangent_dist = max_dist
            # Recalculate actual fillet radius for this corner
            actual_radius = tangent_dist * np.tan(half_angle)
        else:
            actual_radius = fillet_radius

        # Tangent points on each edge
        tangent1 = curr_v + v1_norm * tangent_dist
        tangent2 = curr_v + v2_norm * tangent_dist

        # Calculate arc center
        # The center is along the angle bisector at distance = actual_radius / sin(half_angle)
        bisector = v1_norm + v2_norm
        bisector_len = np.linalg.norm(bisector)
        if bisector_len < 1e-10:
            # Edges are nearly opposite (180 degrees), use perpendicular
            bisector = np.array([-v1_norm[1], v1_norm[0]])
        else:
            bisector = bisector / bisector_len

        center_dist = actual_radius / np.sin(half_angle)
        arc_center = curr_v + bisector * center_dist

        # Generate arc points from tangent1 to tangent2
        # Calculate start and end angles for the arc
        start_vec = tangent1 - arc_center
        end_vec = tangent2 - arc_center

        start_angle = np.arctan2(start_vec[1], start_vec[0])
        end_angle = np.arctan2(end_vec[1], end_vec[0])

        # Determine arc direction (should go the "short way" around the corner)
        # The arc should be on the outer side of the corner (convex hull side)
        angle_diff = end_angle - start_angle

        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Generate arc points
        for j in range(segments_per_corner + 1):
            t = j / segments_per_corner
            theta = start_angle + angle_diff * t
            arc_pt = arc_center + actual_radius * np.array([np.cos(theta), np.sin(theta)])
            rounded_points.append(arc_pt.tolist())

    return rounded_points


def create_irregular_hexagon_prism(corners: list, height: float, wall_thickness: float,
                                   resolution: int = 6, edge_curve: float = 0.0) -> m3d.Manifold | None:
    """
    Create an irregular hexagonal prism from 6 corner positions.

    Args:
        corners: List of (corner_idx, (x, y)) tuples for 6 corners
        height: Height of the prism
        wall_thickness: Wall thickness
        resolution: Mesh resolution (also used for arc segments when rounding)
        edge_curve: Corner rounding factor (0.0 = sharp, 1.0 = maximum rounding)

    Returns:
        Hollow hexagonal prism manifold, or None on failure
    """
    if len(corners) != 6:
        return None

    points = [pos for _, pos in corners]
    outer_pts = np.array([[p[0], p[1]] for p in points], dtype=np.float64)

    cx = np.mean(outer_pts[:, 0])
    cy = np.mean(outer_pts[:, 1])

    # Calculate average radius for fillet sizing
    avg_radius = np.mean([np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in outer_pts])

    inner_pts = []
    for px, py in outer_pts:
        dx, dy = px - cx, py - cy
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > wall_thickness:
            scale = (dist - wall_thickness) / dist
            inner_pts.append([cx + dx * scale, cy + dy * scale])
        else:
            inner_pts.append([cx, cy])
    inner_pts = np.array(inner_pts, dtype=np.float64)

    try:
        if edge_curve > 0:
            # Calculate fillet radius based on edge_curve
            # At edge_curve=1.0, fillet radius approaches the inscribed circle radius
            # which would make the hexagon nearly circular
            # For a regular hexagon, inscribed circle radius = sqrt(3)/2 * circumradius
            # We use a fraction of avg_radius to ensure we don't exceed edge lengths
            max_fillet = avg_radius * 0.5  # Maximum sensible fillet
            fillet_radius = edge_curve * max_fillet

            # Number of segments per corner for smooth arcs
            segments = max(4, resolution)

            outer_rounded = create_rounded_polygon_points(outer_pts, fillet_radius, segments)
            inner_rounded = create_rounded_polygon_points(inner_pts, fillet_radius * (1 - wall_thickness / avg_radius), segments)

            outer_cross = m3d.CrossSection([outer_rounded])
            inner_cross = m3d.CrossSection([inner_rounded])
        else:
            outer_cross = m3d.CrossSection([outer_pts.tolist()])
            inner_cross = m3d.CrossSection([inner_pts.tolist()])

        outer_prism = outer_cross.extrude(height)
        inner_prism = inner_cross.extrude(height + 0.02).translate([0, 0, -0.01])

        return outer_prism - inner_prism
    except Exception:
        return None


def create_hexagon_prism(radius: float, height: float, wall_thickness: float,
                         resolution: int = 32) -> m3d.Manifold:
    """Create a hexagonal prism (hollow) using manifold3d."""
    outer = m3d.Manifold.cylinder(height, radius, radius, 6)
    inner_radius = radius - wall_thickness
    inner = m3d.Manifold.cylinder(height + 0.02, inner_radius, inner_radius, 6)
    inner = inner.translate([0, 0, -0.01])
    return outer - inner


def create_vertical_tube(x: float, y: float, height: float, radius: float,
                         resolution: int = 16) -> m3d.Manifold:
    """Create a vertical cylindrical tube at position (x, y)."""
    tube = m3d.Manifold.cylinder(height, radius, radius, resolution)
    return tube.translate([x, y, 0])


def create_hollow_vertical_tube(x: float, y: float, height: float, radius: float,
                                wall_thickness: float, resolution: int = 16,
                                cap_bottom: bool = True, cap_top: bool = True) -> m3d.Manifold:
    """
    Create a hollow vertical tube at position (x, y).

    Args:
        x, y: Position of tube center
        height: Height of tube
        radius: Outer radius
        wall_thickness: Wall thickness (inner radius = radius - wall_thickness)
        resolution: Number of sides for cylinder
        cap_bottom: If True, close the bottom end (Z=0)
        cap_top: If True, close the top end (Z=height)

    Returns:
        Hollow tube manifold with specified end caps
    """
    inner_radius = radius - wall_thickness
    if inner_radius <= 0:
        # If wall is too thick, just return solid cylinder
        return m3d.Manifold.cylinder(height, radius, radius, resolution).translate([x, y, 0])

    # Create outer and inner cylinders
    outer = m3d.Manifold.cylinder(height, radius, radius, resolution)

    # Inner cylinder - extend beyond ends to ensure clean subtraction
    inner_ext = 0.1  # Extension amount
    inner_height = height
    inner_z_offset = 0

    if not cap_bottom:
        inner_height += inner_ext
        inner_z_offset = -inner_ext
    if not cap_top:
        inner_height += inner_ext

    inner = m3d.Manifold.cylinder(inner_height, inner_radius, inner_radius, resolution)
    inner = inner.translate([0, 0, inner_z_offset])

    # Subtract inner from outer to create hollow tube
    hollow = outer - inner

    return hollow.translate([x, y, 0])


def get_hexagon_corners(center_x: float, center_y: float, radius: float) -> list[tuple[float, float]]:
    """Get the 6 corner positions of a hexagon at given center."""
    rotation_offset = np.radians(30)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + rotation_offset
    return [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]


def create_sinusoid(start_pos: tuple[float, float, float], end_pos: tuple[float, float, float],
                    radius: float, resolution: int = 8) -> m3d.Manifold | None:
    """Create a sinusoid channel between two points."""
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 0.01:
        return None

    cyl = m3d.Manifold.cylinder(length, radius, radius, resolution)

    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return cyl.translate([x1, y1, z1])


def create_hollow_sinusoid(start_pos: tuple[float, float, float],
                           end_pos: tuple[float, float, float],
                           radius: float, wall_thickness: float,
                           resolution: int = 8) -> m3d.Manifold | None:
    """
    Create a hollow sinusoid channel between two points with both ends open.

    Args:
        start_pos: (x, y, z) start position
        end_pos: (x, y, z) end position
        radius: Outer radius
        wall_thickness: Wall thickness (inner radius = radius - wall_thickness)
        resolution: Number of sides for cylinder

    Returns:
        Hollow tube manifold with both ends open
    """
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 0.01:
        return None

    inner_radius = radius - wall_thickness
    if inner_radius <= 0:
        # Wall too thick, return solid
        return create_sinusoid(start_pos, end_pos, radius, resolution)

    # Create outer cylinder
    outer = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Create inner cylinder - extend beyond both ends to ensure open ends
    inner_ext = 0.1
    inner = m3d.Manifold.cylinder(length + 2 * inner_ext, inner_radius, inner_radius, resolution)
    inner = inner.translate([0, 0, -inner_ext])

    # Hollow tube
    hollow = outer - inner

    # Rotate to align with direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        hollow = hollow.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return hollow.translate([x1, y1, z1])


def bezier_curve(p0: tuple, p1: tuple, p2: tuple, p3: tuple,
                 num_points: int = 20) -> list[tuple[float, float, float]]:
    """Generate points along a cubic Bezier curve."""
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3*p0[0] + 3*mt2*t*p1[0] + 3*mt*t2*p2[0] + t3*p3[0]
        y = mt3*p0[1] + 3*mt2*t*p1[1] + 3*mt*t2*p2[1] + t3*p3[1]
        z = mt3*p0[2] + 3*mt2*t*p1[2] + 3*mt*t2*p2[2] + t3*p3[2]
        points.append((x, y, z))

    return points


def create_solid_tube_manifold(points: list, radius_start: float, radius_end: float,
                               segments: int = 16) -> m3d.Manifold | None:
    """Create a SOLID tapered tube along a path using manifold3d."""
    if len(points) < 2:
        return None

    pts = np.array(points)
    n = len(pts)
    radii = np.linspace(radius_start, radius_end, n)

    solids = []
    for i in range(n - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        r1 = radii[i]
        r2 = radii[i + 1]

        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        direction = direction / length
        cyl = m3d.Manifold.cylinder(length, r1, r2, segments)

        z_axis = np.array([0, 0, 1])
        dot = np.dot(z_axis, direction)

        if dot < -0.9999:
            cyl = cyl.rotate([180, 0, 0])
        elif dot < 0.9999:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
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

        cyl = cyl.translate(list(p1))
        solids.append(cyl)

        if i < n - 2:
            sphere = m3d.Manifold.sphere(r2 * 0.99, segments)
            sphere = sphere.translate(list(p2))
            solids.append(sphere)

    if not solids:
        return None

    return tree_union(solids)


def create_hollow_vessel_network(tube_specs: list, wall_thickness: float = 0.02,
                                 segments: int = 16, junction_spheres: list = None) -> m3d.Manifold | None:
    """
    Create a unified hollow vessel network from multiple connected tubes.

    Extends tube paths slightly at endpoints to ensure overlap and proper
    connection even when bezier curve endpoints don't perfectly align.

    Args:
        tube_specs: List of (points, radius_start, radius_end) tuples
        wall_thickness: Thickness of tube walls (absolute, not fraction)
        segments: Number of segments for tube cross-section
        junction_spheres: Optional list of (position, radius) for explicit junction smoothing

    Returns:
        Manifold3d object representing the hollow network with proper connections
    """
    if not tube_specs:
        return None

    exterior_solids = []
    interior_solids = []

    for points, r_start, r_end in tube_specs:
        pts = np.array(points)
        if len(pts) < 2:
            continue

        # Extend tube at both ends along the tangent direction
        # This ensures tubes overlap at junctions even if endpoints don't match exactly
        extend_amount = max(r_start, r_end) * 0.5

        # Extend at start
        tangent_start = pts[1] - pts[0]
        tangent_start = tangent_start / (np.linalg.norm(tangent_start) + 1e-10)
        new_start = pts[0] - tangent_start * extend_amount

        # Extend at end
        tangent_end = pts[-1] - pts[-2]
        tangent_end = tangent_end / (np.linalg.norm(tangent_end) + 1e-10)
        new_end = pts[-1] + tangent_end * extend_amount

        # Create extended path
        extended_pts = [new_start] + list(pts) + [new_end]

        # Exterior tube with extended path
        exterior = create_solid_tube_manifold(extended_pts, r_start, r_end, segments)
        if exterior:
            exterior_solids.append(exterior)

        # Interior tube (smaller by wall_thickness)
        inner_r_start = max(r_start - wall_thickness, r_start * 0.5)
        inner_r_end = max(r_end - wall_thickness, r_end * 0.5)
        interior = create_solid_tube_manifold(extended_pts, inner_r_start, inner_r_end, segments)
        if interior:
            interior_solids.append(interior)

    # Add explicit junction spheres if provided
    if junction_spheres:
        for pos, radius in junction_spheres:
            # Exterior sphere
            sphere_out = m3d.Manifold.sphere(radius, segments)
            sphere_out = sphere_out.translate(list(pos))
            exterior_solids.append(sphere_out)

            # Interior sphere (hollow)
            inner_r = max(radius - wall_thickness, radius * 0.5)
            sphere_in = m3d.Manifold.sphere(inner_r, segments)
            sphere_in = sphere_in.translate(list(pos))
            interior_solids.append(sphere_in)

    if not exterior_solids:
        return None

    # Union all exteriors
    exterior_union = tree_union(exterior_solids)

    # Union all interiors and subtract
    if interior_solids:
        interior_union = tree_union(interior_solids)
        return exterior_union - interior_union
    else:
        return exterior_union


def generate_randomized_z_positions(count: int, total_height: float, randomness: float,
                                    min_spacing: float, rng, seed_offset: int = 0) -> list[float]:
    """Generate Z positions with randomization and minimum spacing constraint."""
    if count <= 0:
        return []

    # Start with evenly spaced positions
    base_positions = [total_height * (i + 0.5) / count for i in range(count)]

    if randomness <= 0:
        return base_positions

    # Calculate max random offset (half the spacing between adjacent entrances)
    base_spacing = total_height / count if count > 1 else total_height
    max_offset = base_spacing * 0.4 * randomness  # 40% of spacing, scaled by randomness

    # Apply random offsets
    positions = []
    for i, base_z in enumerate(base_positions):
        offset = rng.uniform(-max_offset, max_offset)
        new_z = base_z + offset
        # Clamp to valid range
        new_z = max(min_spacing, min(total_height - min_spacing, new_z))
        positions.append(new_z)

    # Enforce minimum spacing by adjusting positions
    positions.sort()
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] < min_spacing:
            positions[i] = positions[i-1] + min_spacing

    # Clamp again after adjustment
    positions = [max(min_spacing, min(total_height - min_spacing, z)) for z in positions]

    return positions


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_hepatic_lobule(params: HepaticLobuleParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a hepatic lobule scaffold.

    Args:
        params: HepaticLobuleParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, num_lobules, scaffold_type

    Raises:
        ValueError: If no geometry is generated
    """
    # Initialize random seed for reproducibility
    rng = np.random.default_rng(params.seed)

    num_lobules = params.num_lobules
    radius = params.lobule_radius
    height = params.lobule_height
    wall_thickness = params.wall_thickness
    resolution = params.resolution

    central_r = params.central_vein_radius
    portal_r = params.portal_vein_radius
    artery_r = params.hepatic_artery_radius
    bile_r = params.bile_duct_radius

    # Randomness parameters
    corner_noise = params.corner_noise
    angle_noise = params.angle_noise
    stretch_noise = params.stretch_noise
    size_variance = params.size_variance
    edge_curve = params.edge_curve

    # Generate per-lobule size multipliers (for size variance)
    lobule_size_mult = {}
    for i in range(num_lobules):
        if size_variance > 0:
            lobule_size_mult[i] = 1.0 + (rng.random() - 0.4) * size_variance * 1.4
        else:
            lobule_size_mult[i] = 1.0

    # Generate lobule centers in honeycomb pattern
    lobule_centers = generate_honeycomb_centers(num_lobules, radius)

    # Find unique corner positions (shared corners between adjacent lobules)
    unique_corners, corner_to_lobules = find_unique_corners(lobule_centers, radius)

    # Perturb unique corners (ensures shared edges remain compatible)
    perturbed_corners = perturb_unique_corners(
        unique_corners, lobule_centers, corner_to_lobules, radius,
        corner_noise, angle_noise, stretch_noise, rng
    )

    num_triads = len(unique_corners)

    # Store per-lobule corner data for scaffold creation
    lobule_corner_data = {}
    for lob_idx in range(num_lobules):
        lobule_corner_data[lob_idx] = get_lobule_corners_from_unique(
            lob_idx, lobule_centers[lob_idx], unique_corners, corner_to_lobules,
            perturbed_corners, radius, stretch_noise, rng
        )

    # Track statistics
    cv_entrance_total = 0
    portal_triad_total = 0
    hepatic_artery_total = 0
    bile_duct_total = 0
    portal_entrance_total = 0
    sinusoid_total = 0

    # ==========================================================================
    # Create hexagonal prism scaffolds for each lobule
    # ==========================================================================
    scaffolds = []

    for lob_idx, (lob_x, lob_y) in enumerate(lobule_centers):
        corners = lobule_corner_data[lob_idx]
        has_randomness = corner_noise > 0 or angle_noise > 0 or stretch_noise > 0
        size_mult = lobule_size_mult[lob_idx]

        if len(corners) == 6 and (has_randomness or edge_curve > 0):
            # Use irregular hexagon prism when randomness is applied OR edge_curve is set
            if size_variance > 0:
                scaled_corners = []
                for corner_idx, (cx, cy) in corners:
                    scaled_x = lob_x + (cx - lob_x) * size_mult
                    scaled_y = lob_y + (cy - lob_y) * size_mult
                    scaled_corners.append((corner_idx, (scaled_x, scaled_y)))
                scaffold = create_irregular_hexagon_prism(scaled_corners, height, wall_thickness, resolution, edge_curve)
            else:
                scaffold = create_irregular_hexagon_prism(corners, height, wall_thickness, resolution, edge_curve)

            if scaffold is None:
                # Fallback to regular hexagon (edge_curve not supported in fallback)
                lob_radius = radius * size_mult if size_variance > 0 else radius
                outer = m3d.Manifold.cylinder(height, lob_radius, lob_radius, 6).rotate([0, 0, 30])
                inner_radius = lob_radius - wall_thickness
                inner = m3d.Manifold.cylinder(height + 0.02, inner_radius, inner_radius, 6).rotate([0, 0, 30])
                inner = inner.translate([0, 0, -0.01])
                scaffold = (outer - inner).translate([lob_x, lob_y, 0])
            else:
                # scaffold already positioned correctly from irregular prism
                pass
        else:
            # No randomness and no edge_curve - all lobules same size, perfect tiling
            outer = m3d.Manifold.cylinder(height, radius, radius, 6).rotate([0, 0, 30])
            inner_radius = radius - wall_thickness
            inner = m3d.Manifold.cylinder(height + 0.02, inner_radius, inner_radius, 6).rotate([0, 0, 30])
            inner = inner.translate([0, 0, -0.01])
            scaffold = (outer - inner).translate([lob_x, lob_y, 0])

        scaffolds.append(scaffold)

    # ==========================================================================
    # Create central veins (hollow, open at top for hepatic collector)
    # ==========================================================================
    central_veins = []
    cv_positions = []

    for lob_idx, (lob_x, lob_y) in enumerate(lobule_centers):
        cv_wall_thick = central_r * 0.15
        cv = create_hollow_vertical_tube(lob_x, lob_y, height, central_r, cv_wall_thick, resolution,
                                         cap_bottom=True, cap_top=False)
        central_veins.append(cv)
        cv_positions.append((lob_x, lob_y))

    # ==========================================================================
    # Create central vein entrance channels
    # ==========================================================================
    central_vein_entrances = []
    cv_entrance_len = params.cv_entrance_length
    cv_entrance_r = params.cv_entrance_radius
    cv_entrance_count = params.cv_entrance_count
    cv_entrance_wall = cv_entrance_r * 0.15
    cv_wall_thick = central_r * 0.15
    cv_inner_r = central_r - cv_wall_thick
    cv_z_randomness = params.cv_entrance_z_randomness
    cv_min_spacing = params.cv_entrance_min_spacing

    if not params.show_sinusoids and cv_entrance_len > 0 and cv_entrance_count > 0:
        for lob_idx, (lob_x, lob_y) in enumerate(lobule_centers):
            corners = lobule_corner_data[lob_idx]

            for corner_local_idx, (corner_idx, (cx, cy)) in enumerate(corners):
                # Direction from center to corner
                dx = cx - lob_x
                dy = cy - lob_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 0.01:
                    continue
                outward_x, outward_y = dx / dist, dy / dist

                # Generate randomized Z positions
                seed_offset = lob_idx * 1000 + corner_local_idx * 100 + 1
                z_positions = generate_randomized_z_positions(
                    cv_entrance_count, height, cv_z_randomness, cv_min_spacing, rng, seed_offset
                )

                for z in z_positions:
                    # Start from INNER surface of central vein
                    start_x = lob_x + outward_x * cv_inner_r
                    start_y = lob_y + outward_y * cv_inner_r
                    end_x = lob_x + outward_x * (central_r + cv_entrance_len)
                    end_y = lob_y + outward_y * (central_r + cv_entrance_len)

                    cv_entrance = create_hollow_sinusoid(
                        (start_x, start_y, z),
                        (end_x, end_y, z),
                        cv_entrance_r,
                        cv_entrance_wall,
                        resolution=resolution
                    )
                    if cv_entrance:
                        central_vein_entrances.append(cv_entrance)
                        cv_entrance_total += 1

    # ==========================================================================
    # Create portal triads at each unique corner
    # ==========================================================================
    portal_veins = []
    hepatic_arteries = []
    bile_ducts = []
    portal_entrances = []
    artery_entrances = []

    portal_vein_positions = []
    hepatic_artery_positions = []

    triad_offset = params.triad_wall_distance
    entrance_len = params.entrance_length
    entrance_r = params.entrance_radius
    entrance_count = params.entrance_count
    entrance_wall = entrance_r * 0.15
    entrance_z_randomness = params.entrance_z_randomness
    entrance_min_spacing = params.entrance_min_spacing
    pv_wall_thick = portal_r * 0.15
    pv_inner_r = portal_r - pv_wall_thick
    ha_wall_thick = artery_r * 0.15
    ha_inner_r = artery_r - ha_wall_thick

    for triad_idx in range(len(unique_corners)):
        # Use perturbed corner position
        corner_x, corner_y = perturbed_corners[triad_idx]

        # Get the lobules that share this corner
        adjacent_lobule_indices = corner_to_lobules[triad_idx]
        num_adjacent = len(adjacent_lobule_indices)

        # Calculate average direction from corner to adjacent lobule centers
        avg_dx, avg_dy = 0, 0
        for lob_idx in adjacent_lobule_indices:
            lob_x, lob_y = lobule_centers[lob_idx]
            dx = lob_x - corner_x
            dy = lob_y - corner_y
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 0.01:
                avg_dx += dx / dist
                avg_dy += dy / dist

        # Normalize average direction
        avg_dist = np.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
        if avg_dist > 0.01:
            avg_dx /= avg_dist
            avg_dy /= avg_dist
        else:
            if num_adjacent >= 3:
                centroid_x = np.mean([lobule_centers[i][0] for i in adjacent_lobule_indices])
                centroid_y = np.mean([lobule_centers[i][1] for i in adjacent_lobule_indices])
                dx = centroid_x - corner_x
                dy = centroid_y - corner_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.01:
                    avg_dx, avg_dy = dx/dist, dy/dist
                else:
                    avg_dx, avg_dy = 1, 0
            else:
                avg_dx, avg_dy = 1, 0

        # Portal vein position
        pv_x = corner_x + avg_dx * triad_offset
        pv_y = corner_y + avg_dy * triad_offset
        # Hollow tube: open at bottom (connects to portal collector), closed at top
        pv_mesh = create_hollow_vertical_tube(pv_x, pv_y, height, portal_r, pv_wall_thick, resolution,
                                              cap_bottom=False, cap_top=True)
        portal_veins.append(pv_mesh)
        portal_triad_total += 1

        # Hepatic artery - perpendicular offset from portal vein
        perp_x, perp_y = -avg_dy, avg_dx
        ha_x = pv_x + perp_x * (portal_r + artery_r + 0.02)
        ha_y = pv_y + perp_y * (portal_r + artery_r + 0.02)
        ha_mesh = create_hollow_vertical_tube(ha_x, ha_y, height, artery_r, ha_wall_thick, resolution,
                                              cap_bottom=True, cap_top=True)
        hepatic_arteries.append(ha_mesh)
        hepatic_artery_total += 1

        # Store positions for collector routing
        portal_vein_positions.append((pv_x, pv_y))
        hepatic_artery_positions.append((ha_x, ha_y))

        # Bile duct
        if params.show_bile_ducts:
            bd_x = pv_x - perp_x * (portal_r + bile_r + 0.02)
            bd_y = pv_y - perp_y * (portal_r + bile_r + 0.02)
            bd_mesh = create_vertical_tube(bd_x, bd_y, height, bile_r, resolution)
            bile_ducts.append(bd_mesh)
            bile_duct_total += 1

        # Create entrance stubs pointing toward ALL adjacent lobule centers
        if not params.show_sinusoids and entrance_len > 0 and entrance_count > 0:
            for adj_idx, lob_idx in enumerate(adjacent_lobule_indices):
                lob_x, lob_y = lobule_centers[lob_idx]

                # Direction from triad to this lobule's center
                dx = lob_x - pv_x
                dy = lob_y - pv_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 0.01:
                    continue
                inward_x, inward_y = dx / dist, dy / dist

                # Generate randomized Z positions
                seed_offset = 50000 + triad_idx * 1000 + adj_idx * 100 + 2
                z_positions = generate_randomized_z_positions(
                    entrance_count, height, entrance_z_randomness, entrance_min_spacing, rng, seed_offset
                )

                for z in z_positions:
                    # Portal vein entrance
                    pv_start_x = pv_x + inward_x * pv_inner_r
                    pv_start_y = pv_y + inward_y * pv_inner_r
                    pv_end_x = pv_x + inward_x * (portal_r + entrance_len)
                    pv_end_y = pv_y + inward_y * (portal_r + entrance_len)

                    pv_entrance = create_hollow_sinusoid(
                        (pv_start_x, pv_start_y, z),
                        (pv_end_x, pv_end_y, z),
                        entrance_r,
                        entrance_wall,
                        resolution=resolution
                    )
                    if pv_entrance:
                        portal_entrances.append(pv_entrance)
                        portal_entrance_total += 1

                    # Hepatic artery entrance
                    ha_ent_r = entrance_r * 0.7
                    ha_ent_wall = ha_ent_r * 0.15
                    ha_start_x = ha_x + inward_x * ha_inner_r
                    ha_start_y = ha_y + inward_y * ha_inner_r
                    ha_end_x = ha_x + inward_x * (artery_r + entrance_len)
                    ha_end_y = ha_y + inward_y * (artery_r + entrance_len)

                    ha_entrance = create_hollow_sinusoid(
                        (ha_start_x, ha_start_y, z),
                        (ha_end_x, ha_end_y, z),
                        ha_ent_r,
                        ha_ent_wall,
                        resolution=resolution
                    )
                    if ha_entrance:
                        artery_entrances.append(ha_entrance)

    # ==========================================================================
    # Create sinusoids
    # ==========================================================================
    sinusoids = []
    if params.show_sinusoids:
        sinusoid_r = params.sinusoid_radius
        n_levels = params.sinusoid_levels
        n_per_corner = params.sinusoid_count

        for level in range(n_levels):
            z = height * (level + 0.5) / n_levels

            for triad_idx in range(len(unique_corners)):
                corner_x, corner_y = perturbed_corners[triad_idx]

                for lob_idx in corner_to_lobules[triad_idx]:
                    lob_x, lob_y = lobule_centers[lob_idx]

                    dx = lob_x - corner_x
                    dy = lob_y - corner_y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 0.01:
                        continue

                    start_dist = 0.15
                    sx = corner_x + (dx / dist) * start_dist
                    sy = corner_y + (dy / dist) * start_dist

                    end_dist = central_r + 0.05
                    ex = lob_x - (dx / dist) * end_dist
                    ey = lob_y - (dy / dist) * end_dist

                    base_angle = np.arctan2(dy, dx)
                    spread = np.radians(20)

                    for j in range(n_per_corner):
                        if n_per_corner > 1:
                            angle_offset = spread * (j / (n_per_corner - 1) - 0.5) * 2
                        else:
                            angle_offset = 0

                        offset_dist = 0.1
                        s_off_x = sx + np.cos(base_angle + np.pi/2) * offset_dist * angle_offset
                        s_off_y = sy + np.sin(base_angle + np.pi/2) * offset_dist * angle_offset
                        e_off_x = ex + np.cos(base_angle + np.pi/2) * offset_dist * angle_offset * 0.5
                        e_off_y = ey + np.sin(base_angle + np.pi/2) * offset_dist * angle_offset * 0.5

                        z_var = (rng.random() - 0.5) * 0.1

                        sinusoid = create_sinusoid(
                            (s_off_x, s_off_y, z + z_var),
                            (e_off_x, e_off_y, z - z_var),
                            sinusoid_r,
                            resolution=6
                        )
                        if sinusoid:
                            sinusoids.append(sinusoid)
                            sinusoid_total += 1

    # ==========================================================================
    # Create hepatic collector (above lobules)
    # ==========================================================================
    hepatic_collector = []

    if params.show_hepatic_collector and num_lobules > 0:
        lymph_h = params.hepatic_collector_height
        branch_r = central_r
        tube_sides = 24

        all_x = [c[0] for c in lobule_centers]
        all_y = [c[1] for c in lobule_centers]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        span_x = max_x - min_x
        span_y = max_y - min_y

        if span_x >= span_y:
            collector_axis = 'x'
            sorted_indices = sorted(range(num_lobules), key=lambda i: lobule_centers[i][0])
        else:
            collector_axis = 'y'
            sorted_indices = sorted(range(num_lobules), key=lambda i: lobule_centers[i][1])

        exit_z = height + lymph_h
        dip_z = height + lymph_h * 0.15

        if num_lobules == 1:
            lx, ly = lobule_centers[0]
            if collector_axis == 'x':
                exit_point = (lx + radius * 2, ly, exit_z)
            else:
                exit_point = (lx, ly + radius * 2, exit_z)

            p0 = (lx, ly, dip_z)
            p3 = exit_point
            p1 = (lx, ly, (dip_z + exit_z) / 2)
            p2 = ((p0[0] + p3[0])/2, (p0[1] + p3[1])/2, exit_z)

            curve_pts = bezier_curve(p0, p1, p2, p3, 20)
            tube = create_solid_tube_manifold(curve_pts, branch_r, branch_r, tube_sides)
            if tube:
                hepatic_collector.append(tube)

            cv_pts = [(lx, ly, height - branch_r * 0.5), (lx, ly, dip_z)]
            stub = create_solid_tube_manifold(cv_pts, branch_r, branch_r, tube_sides)
            if stub:
                hepatic_collector.append(stub)
        else:
            # Cluster central veins by proximity
            remaining = list(range(num_lobules))
            clusters = []

            while len(remaining) >= 2:
                best_dist = float('inf')
                best_i, best_j = 0, 1
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        li, lj = remaining[i], remaining[j]
                        dx = lobule_centers[li][0] - lobule_centers[lj][0]
                        dy = lobule_centers[li][1] - lobule_centers[lj][1]
                        d = dx*dx + dy*dy
                        if d < best_dist:
                            best_dist = d
                            best_i, best_j = i, j

                cluster = [remaining[best_i], remaining[best_j]]
                remaining.pop(best_j)
                remaining.pop(best_i)

                if remaining:
                    cx = np.mean([lobule_centers[i][0] for i in cluster])
                    cy = np.mean([lobule_centers[i][1] for i in cluster])
                    nearest_idx = min(range(len(remaining)), key=lambda i:
                        (lobule_centers[remaining[i]][0] - cx)**2 +
                        (lobule_centers[remaining[i]][1] - cy)**2)
                    nearest = remaining[nearest_idx]
                    dist = np.sqrt((lobule_centers[nearest][0] - cx)**2 +
                                   (lobule_centers[nearest][1] - cy)**2)
                    if dist < radius * 2.5:
                        cluster.append(nearest)
                        remaining.pop(nearest_idx)

                        if remaining:
                            cx = np.mean([lobule_centers[i][0] for i in cluster])
                            cy = np.mean([lobule_centers[i][1] for i in cluster])
                            nearest_idx = min(range(len(remaining)), key=lambda i:
                                (lobule_centers[remaining[i]][0] - cx)**2 +
                                (lobule_centers[remaining[i]][1] - cy)**2)
                            nearest = remaining[nearest_idx]
                            dist = np.sqrt((lobule_centers[nearest][0] - cx)**2 +
                                           (lobule_centers[nearest][1] - cy)**2)
                            if dist < radius * 2:
                                cluster.append(nearest)
                                remaining.pop(nearest_idx)

                clusters.append(cluster)

            if remaining:
                lob = remaining[0]
                lx, ly = lobule_centers[lob]
                best_j = 0
                best_dist = float('inf')
                for j, other in enumerate(clusters):
                    ocx = np.mean([lobule_centers[k][0] for k in other])
                    ocy = np.mean([lobule_centers[k][1] for k in other])
                    d = (lx - ocx)**2 + (ly - ocy)**2
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                clusters[best_j].append(lob)

            def cluster_pos(c):
                if collector_axis == 'x':
                    return np.mean([lobule_centers[i][0] for i in c])
                else:
                    return np.mean([lobule_centers[i][1] for i in c])
            clusters.sort(key=cluster_pos)

            cluster_data = []
            cumulative_veins = 0
            for cluster in clusters:
                n_in_cluster = len(cluster)
                cx = np.mean([lobule_centers[i][0] for i in cluster])
                cy = np.mean([lobule_centers[i][1] for i in cluster])

                if collector_axis == 'x':
                    progress = (cx - min_x) / max(span_x, 0.001)
                    attach_x, attach_y = cx, center_y
                else:
                    progress = (cy - min_y) / max(span_y, 0.001)
                    attach_x, attach_y = center_x, cy

                collector_z_here = dip_z + (exit_z - dip_z) * progress
                merge_z = height + (collector_z_here - height) * 0.5
                cumulative_veins += n_in_cluster
                collector_r_here = branch_r * np.sqrt(cumulative_veins)

                cluster_data.append({
                    'merge_point': (cx, cy, merge_z),
                    'attach_point': (attach_x, attach_y, collector_z_here),
                    'collector_radius': collector_r_here,
                    'cumulative': cumulative_veins,
                    'size': n_in_cluster,
                    'indices': cluster,
                    'progress': progress
                })

            first_cluster = cluster_data[0]
            last_cluster = cluster_data[-1]
            first_cx, first_cy, _ = first_cluster['merge_point']
            start_point = (first_cx, first_cy, dip_z)
            start_radius = branch_r * np.sqrt(first_cluster['size'])

            last_ax, last_ay, _ = last_cluster['attach_point']
            if collector_axis == 'x':
                end_point = (last_ax + radius * 2.0, center_y, exit_z)
            else:
                end_point = (center_x, last_ay + radius * 2.0, exit_z)
            end_radius = branch_r * np.sqrt(num_lobules)

            p0 = start_point
            p3 = end_point
            rise_z = dip_z + (exit_z - dip_z) * 0.5
            if collector_axis == 'x':
                p1 = (first_cx, center_y, rise_z)
                p2 = ((first_cx + end_point[0]) / 2, center_y, exit_z * 0.95)
            else:
                p1 = (center_x, first_cy, rise_z)
                p2 = (center_x, (first_cy + end_point[1]) / 2, exit_z * 0.95)

            collector_path = bezier_curve(p0, p1, p2, p3, 20)
            collector_path_arr = np.array(collector_path)

            cv_branch_specs = []
            sub_collector_specs = []
            junction_points = []

            for cluster_idx, cdata in enumerate(cluster_data):
                mx, my, mz = cdata['merge_point']
                cluster_r = branch_r * np.sqrt(cdata['size'])

                if cluster_idx == 0:
                    collector_tip = collector_path[0]
                    junction_points.append((collector_tip, start_radius * 1.1))

                    for lob_idx in cdata['indices']:
                        lx, ly = lobule_centers[lob_idx]
                        cv_top = (lx, ly, height)
                        p0 = cv_top
                        p3 = collector_tip
                        rise_z = height + (p3[2] - height) * 0.5
                        p1 = (lx, ly, rise_z)
                        p2 = ((lx + p3[0])/2, (ly + p3[1])/2, p3[2])
                        curve_pts = bezier_curve(p0, p1, p2, p3, 12)
                        cv_branch_specs.append((curve_pts, branch_r))
                else:
                    if collector_axis == 'x':
                        dists = np.abs(collector_path_arr[:, 0] - mx)
                    else:
                        dists = np.abs(collector_path_arr[:, 1] - my)
                    closest_idx = np.argmin(dists)
                    attach_point = collector_path[closest_idx]

                    n_veins = len(cdata['indices'])
                    sub_collector_r = branch_r * np.sqrt(n_veins)
                    merge_z = height + (attach_point[2] - height) * 0.4
                    merge_point = (mx, my, merge_z)

                    junction_points.append((attach_point, sub_collector_r * 1.1))
                    junction_points.append((merge_point, sub_collector_r * 1.1))

                    for vein_idx, lob_idx in enumerate(cdata['indices']):
                        lx, ly = lobule_centers[lob_idx]
                        cv_top = (lx, ly, height)
                        spread = (vein_idx - (n_veins - 1) / 2) * branch_r * 0.8
                        if collector_axis == 'x':
                            vein_end = (mx + spread * 0.3, my + spread, merge_z)
                        else:
                            vein_end = (mx + spread, my + spread * 0.3, merge_z)

                        p0 = cv_top
                        p3 = vein_end
                        rise_z = height + (merge_z - height) * 0.4
                        p1 = (lx, ly, rise_z)
                        approach_x = vein_end[0] + (lx - mx) * 0.3
                        approach_y = vein_end[1] + (ly - my) * 0.3
                        approach_z = merge_z - branch_r
                        p2 = (approach_x, approach_y, approach_z)
                        curve_pts = bezier_curve(p0, p1, p2, p3, 12)
                        cv_branch_specs.append((curve_pts, branch_r))

                    collector_r_here = cdata['collector_radius']
                    if closest_idx == 0:
                        tangent = np.array(collector_path[1]) - np.array(collector_path[0])
                    elif closest_idx >= len(collector_path) - 1:
                        tangent = np.array(collector_path[-1]) - np.array(collector_path[-2])
                    else:
                        tangent = np.array(collector_path[closest_idx + 1]) - np.array(collector_path[closest_idx - 1])
                    tangent = tangent / np.linalg.norm(tangent)

                    approach_vec = np.array(attach_point) - np.array(merge_point)
                    radial = approach_vec - np.dot(approach_vec, tangent) * tangent
                    radial_len = np.linalg.norm(radial)
                    if radial_len > 0.01:
                        radial = radial / radial_len
                    else:
                        radial = np.array([0, 0, -1])

                    attach_arr = np.array(attach_point)
                    end_pt = tuple(attach_arr)
                    p0 = merge_point
                    p3 = end_pt
                    mid_z = (merge_z + end_pt[2]) / 2
                    p1 = (mx, my, mid_z)
                    approach_offset = sub_collector_r * 2 + collector_r_here
                    p2 = tuple(attach_arr - radial * approach_offset)
                    sub_curve_pts = bezier_curve(p0, p1, p2, p3, 15)
                    sub_collector_specs.append((sub_curve_pts, sub_collector_r))

            all_tube_specs = []
            all_tube_specs.append((collector_path, start_radius, end_radius))
            for sub_curve_pts, sub_r in sub_collector_specs:
                all_tube_specs.append((sub_curve_pts, sub_r, sub_r))
            for curve_pts, r in cv_branch_specs:
                all_tube_specs.append((curve_pts, r, r))

            wall_th = branch_r * 0.15
            hollow_network = create_hollow_vessel_network(all_tube_specs, wall_th, tube_sides, junction_points)
            if hollow_network:
                hepatic_collector.append(hollow_network)

    # ==========================================================================
    # Create portal collector (below lobules)
    # ==========================================================================
    portal_collector = []

    if params.show_portal_collector and len(unique_corners) > 0:
        coll_h = params.portal_collector_height
        branch_r = portal_r
        tube_sides = 24

        portal_positions = []
        for i in range(len(unique_corners)):
            px, py = perturbed_corners[i]
            portal_positions.append((i, px, py))

        if len(portal_positions) > 0:
            num_pvs = len(portal_positions)
            all_x = [p[1] for p in portal_positions]
            all_y = [p[2] for p in portal_positions]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            span_x = max(max_x - min_x, 0.001)
            span_y = max(max_y - min_y, 0.001)

            if span_x >= span_y:
                collector_axis = 'x'
            else:
                collector_axis = 'y'

            exit_z = -coll_h
            dip_z = -coll_h * 0.15

            if num_pvs == 1:
                _, px, py = portal_positions[0]
                portal_bottom = (px, py, 0)
                if collector_axis == 'x':
                    exit_point = (px - radius * 2, py, exit_z)
                else:
                    exit_point = (px, py - radius * 2, exit_z)

                p0 = portal_bottom
                p3 = exit_point
                p1 = (px, py, (0 + exit_z) / 2)
                p2 = ((px + exit_point[0])/2, (py + exit_point[1])/2, exit_z * 0.9)
                curve_pts = bezier_curve(p0, p1, p2, p3, 20)
                tube = create_solid_tube_manifold(curve_pts, branch_r, branch_r, tube_sides)
                if tube:
                    portal_collector.append(tube)
            else:
                remaining = list(range(num_pvs))
                clusters = []

                while len(remaining) >= 2:
                    best_dist = float('inf')
                    best_i, best_j = 0, 1
                    for i in range(len(remaining)):
                        for j in range(i + 1, len(remaining)):
                            pi, pj = remaining[i], remaining[j]
                            dx = portal_positions[pi][1] - portal_positions[pj][1]
                            dy = portal_positions[pi][2] - portal_positions[pj][2]
                            d = dx*dx + dy*dy
                            if d < best_dist:
                                best_dist = d
                                best_i, best_j = i, j

                    cluster = [remaining[best_i], remaining[best_j]]
                    remaining.pop(best_j)
                    remaining.pop(best_i)

                    for _ in range(3):
                        if not remaining:
                            break
                        cx = np.mean([portal_positions[i][1] for i in cluster])
                        cy = np.mean([portal_positions[i][2] for i in cluster])
                        nearest_idx = min(range(len(remaining)), key=lambda i:
                            (portal_positions[remaining[i]][1] - cx)**2 +
                            (portal_positions[remaining[i]][2] - cy)**2)
                        nearest = remaining[nearest_idx]
                        dist = np.sqrt((portal_positions[nearest][1] - cx)**2 +
                                       (portal_positions[nearest][2] - cy)**2)
                        threshold = radius * (3.0 - 0.3 * len(cluster))
                        if dist < threshold:
                            cluster.append(nearest)
                            remaining.pop(nearest_idx)
                        else:
                            break

                    clusters.append(cluster)

                if remaining:
                    singleton = remaining[0]
                    px, py = portal_positions[singleton][1], portal_positions[singleton][2]
                    best_j = 0
                    best_dist = float('inf')
                    for j, other in enumerate(clusters):
                        ocx = np.mean([portal_positions[k][1] for k in other])
                        ocy = np.mean([portal_positions[k][2] for k in other])
                        d = (px - ocx)**2 + (py - ocy)**2
                        if d < best_dist:
                            best_dist = d
                            best_j = j
                    clusters[best_j].append(singleton)

                def cluster_pos(c):
                    if collector_axis == 'x':
                        return -np.mean([portal_positions[i][1] for i in c])
                    else:
                        return -np.mean([portal_positions[i][2] for i in c])
                clusters.sort(key=cluster_pos)

                cluster_data = []
                cumulative = 0
                for cluster in clusters:
                    n = len(cluster)
                    cx = np.mean([portal_positions[i][1] for i in cluster])
                    cy = np.mean([portal_positions[i][2] for i in cluster])

                    if collector_axis == 'x':
                        progress = (max_x - cx) / span_x
                    else:
                        progress = (max_y - cy) / span_y

                    collector_z = dip_z + (exit_z - dip_z) * progress
                    merge_z = collector_z * 0.4
                    cumulative += n

                    cluster_data.append({
                        'indices': cluster,
                        'cx': cx, 'cy': cy,
                        'merge_z': merge_z,
                        'collector_z': collector_z,
                        'size': n,
                        'cumulative': cumulative,
                        'progress': progress
                    })

                first_c = cluster_data[0]
                start_point = (first_c['cx'], first_c['cy'], dip_z)
                start_radius = branch_r * np.sqrt(first_c['size'])

                if collector_axis == 'x':
                    end_point = (min_x - radius * 2.0, center_y, exit_z)
                else:
                    end_point = (center_x, min_y - radius * 2.0, exit_z)
                end_radius = branch_r * np.sqrt(num_pvs)

                p0 = start_point
                p3 = end_point
                mid_z = (dip_z + exit_z) / 2
                if collector_axis == 'x':
                    p1 = (first_c['cx'], center_y, mid_z)
                    p2 = ((first_c['cx'] + end_point[0]) / 2, center_y, exit_z * 0.9)
                else:
                    p1 = (center_x, first_c['cy'], mid_z)
                    p2 = (center_x, (first_c['cy'] + end_point[1]) / 2, exit_z * 0.9)

                collector_path = bezier_curve(p0, p1, p2, p3, 20)
                collector_path_arr = np.array(collector_path)

                portal_branch_specs = []
                sub_collector_specs = []
                junction_points = []

                for cluster_idx, cdata in enumerate(cluster_data):
                    cx, cy = cdata['cx'], cdata['cy']
                    merge_z = cdata['merge_z']
                    n = cdata['size']
                    sub_r = branch_r * np.sqrt(n)

                    if collector_axis == 'x':
                        dists = np.abs(collector_path_arr[:, 0] - cx)
                    else:
                        dists = np.abs(collector_path_arr[:, 1] - cy)
                    closest_idx = np.argmin(dists)
                    attach_point = tuple(collector_path[closest_idx])

                    if cluster_idx == 0 and n <= 2:
                        collector_tip = collector_path[0]
                        junction_points.append((collector_tip, start_radius * 1.1))
                        for portal_idx in cdata['indices']:
                            _, px, py = portal_positions[portal_idx]
                            portal_bottom = (px, py, 0)
                            p0 = portal_bottom
                            p3 = collector_tip
                            drop_z = p3[2] * 0.4
                            p1 = (px, py, drop_z)
                            p2 = ((px + p3[0])/2, (py + p3[1])/2, p3[2] * 0.9)
                            curve_pts = bezier_curve(p0, p1, p2, p3, 12)
                            portal_branch_specs.append((curve_pts, branch_r))
                    else:
                        merge_point = (cx, cy, merge_z)
                        junction_points.append((attach_point, sub_r * 1.1))
                        junction_points.append((merge_point, sub_r * 1.1))

                        for vein_idx, portal_idx in enumerate(cdata['indices']):
                            _, px, py = portal_positions[portal_idx]
                            portal_bottom = (px, py, 0)
                            spread = (vein_idx - (n - 1) / 2) * branch_r * 0.6
                            if collector_axis == 'x':
                                vein_end = (cx + spread * 0.3, cy + spread, merge_z)
                            else:
                                vein_end = (cx + spread, cy + spread * 0.3, merge_z)

                            p0 = portal_bottom
                            p3 = vein_end
                            drop_z = merge_z * 0.4
                            p1 = (px, py, drop_z)
                            p2 = (vein_end[0] + (px - cx) * 0.2,
                                  vein_end[1] + (py - cy) * 0.2,
                                  merge_z * 0.8)
                            curve_pts = bezier_curve(p0, p1, p2, p3, 12)
                            portal_branch_specs.append((curve_pts, branch_r))

                        if closest_idx == 0:
                            tangent = np.array(collector_path[1]) - np.array(collector_path[0])
                        elif closest_idx >= len(collector_path) - 1:
                            tangent = np.array(collector_path[-1]) - np.array(collector_path[-2])
                        else:
                            tangent = np.array(collector_path[closest_idx + 1]) - np.array(collector_path[closest_idx - 1])
                        tangent = tangent / np.linalg.norm(tangent)

                        approach_vec = np.array(attach_point) - np.array(merge_point)
                        radial = approach_vec - np.dot(approach_vec, tangent) * tangent
                        radial_len = np.linalg.norm(radial)
                        if radial_len > 0.01:
                            radial = radial / radial_len
                        else:
                            radial = np.array([0, 0, 1])

                        progress = closest_idx / max(len(collector_path) - 1, 1)
                        collector_r_here = start_radius + (end_radius - start_radius) * progress

                        p0 = merge_point
                        p3 = attach_point
                        mid_z = (merge_z + attach_point[2]) / 2
                        p1 = (cx, cy, mid_z)
                        approach_offset = sub_r * 2 + collector_r_here
                        attach_arr = np.array(attach_point)
                        p2 = tuple(attach_arr - radial * approach_offset)
                        sub_curve_pts = bezier_curve(p0, p1, p2, p3, 15)
                        sub_collector_specs.append((sub_curve_pts, sub_r))

                all_tube_specs = []
                all_tube_specs.append((collector_path, start_radius, end_radius))
                for sub_pts, sub_r in sub_collector_specs:
                    all_tube_specs.append((sub_pts, sub_r, sub_r))
                for curve_pts, r in portal_branch_specs:
                    all_tube_specs.append((curve_pts, r, r))

                wall_th = branch_r * 0.15
                hollow_network = create_hollow_vessel_network(all_tube_specs, wall_th, tube_sides, junction_points)
                if hollow_network:
                    portal_collector.append(hollow_network)

    # ==========================================================================
    # Combine all components
    # ==========================================================================
    all_parts = (
        scaffolds +
        central_veins +
        central_vein_entrances +
        portal_veins +
        hepatic_arteries +
        bile_ducts +
        portal_entrances +
        artery_entrances +
        sinusoids +
        hepatic_collector +
        portal_collector
    )

    if not all_parts:
        raise ValueError("No geometry generated")

    result = batch_union(all_parts)

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'num_lobules': params.num_lobules,
        'scaffold_type': 'hepatic_lobule',
        'cv_entrance_count': cv_entrance_total,
        'portal_triad_count': portal_triad_total,
        'hepatic_artery_count': hepatic_artery_total,
        'bile_duct_count': bile_duct_total,
        'portal_entrance_count': portal_entrance_total,
        'sinusoid_count': sinusoid_total,
        'hepatic_collector_present': params.show_hepatic_collector,
        'portal_collector_present': params.show_portal_collector,
    }

    return result, stats


def generate_hepatic_lobule_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate hepatic lobule from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching HepaticLobuleParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_hepatic_lobule(HepaticLobuleParams(
        # Basic Geometry
        num_lobules=params.get('num_lobules', 7),
        lobule_radius=params.get('lobule_radius', 1.5),
        lobule_height=params.get('lobule_height', 3.0),
        wall_thickness=params.get('wall_thickness', 0.1),
        resolution=params.get('resolution', 8),
        # Central Vein
        central_vein_radius=params.get('central_vein_radius', 0.15),
        cv_entrance_length=params.get('cv_entrance_length', 0.25),
        cv_entrance_radius=params.get('cv_entrance_radius', 0.04),
        cv_entrance_count=params.get('cv_entrance_count', 5),
        cv_entrance_z_randomness=params.get('cv_entrance_z_randomness', 0.0),
        cv_entrance_min_spacing=params.get('cv_entrance_min_spacing', 0.1),
        # Portal Triad
        portal_vein_radius=params.get('portal_vein_radius', 0.12),
        hepatic_artery_radius=params.get('hepatic_artery_radius', 0.06),
        bile_duct_radius=params.get('bile_duct_radius', 0.05),
        show_bile_ducts=params.get('show_bile_ducts', True),
        triad_wall_distance=params.get('triad_wall_distance', 0.0),
        entrance_length=params.get('entrance_length', 0.25),
        entrance_radius=params.get('entrance_radius', 0.04),
        entrance_count=params.get('entrance_count', 5),
        entrance_z_randomness=params.get('entrance_z_randomness', 0.0),
        entrance_min_spacing=params.get('entrance_min_spacing', 0.1),
        # Sinusoids
        show_sinusoids=params.get('show_sinusoids', False),
        sinusoid_radius=params.get('sinusoid_radius', 0.025),
        sinusoid_count=params.get('sinusoid_count', 6),
        sinusoid_levels=params.get('sinusoid_levels', 3),
        # Organic Variation
        corner_noise=params.get('corner_noise', 0.0),
        angle_noise=params.get('angle_noise', 0.0),
        stretch_noise=params.get('stretch_noise', 0.0),
        size_variance=params.get('size_variance', 0.0),
        edge_curve=params.get('edge_curve', 0.0),
        seed=params.get('seed', 42),
        # Collectors
        show_hepatic_collector=params.get('show_hepatic_collector', True),
        show_portal_collector=params.get('show_portal_collector', True),
        hepatic_collector_height=params.get('hepatic_collector_height', 2.0),
        portal_collector_height=params.get('portal_collector_height', 2.0),
    ))
