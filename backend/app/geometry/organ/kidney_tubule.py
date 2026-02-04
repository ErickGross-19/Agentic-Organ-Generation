"""
Kidney tubule scaffold generator.

Creates convoluted proximal tubule structures with perfusable lumen:
- Sinusoidal tube path mimicking nephron convolution (PCT, DCT)
- Hollow interior for fluid flow
- Epithelial cell-scale features (brush border/microvilli zones)
- Controlled tubule diameter and wall thickness
- Basement membrane representation
- Peritubular capillary network integration
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class KidneyTubuleParams:
    """
    Parameters for kidney tubule scaffold generation.

    Based on native proximal convoluted tubule (PCT) anatomy:
    - Tubule outer diameter: 50-60 um
    - Tubule inner (lumen) diameter: 15-25 um
    - Epithelial cell height: 15-25 um
    - Microvilli (brush border): 1-3 um height
    - PCT length: ~14 mm
    """
    # === Tubule Geometry ===
    tubule_outer_diameter: float = 500.0  # um (scaled for scaffold, native ~50-60)
    tubule_inner_diameter: float = 350.0  # um (lumen diameter)
    tubule_length: float = 15.0  # mm (native PCT ~14mm)

    # === Epithelial Features ===
    epithelial_cell_height: float = 20.0  # um (15-25 um native)
    microvilli_height: float = 2.7  # um (brush border, 1-3 um native)
    enable_brush_border_texture: bool = False  # surface texture for microvilli
    brush_border_density: float = 0.8  # 0-1 coverage

    # === Basement Membrane ===
    enable_basement_membrane: bool = True
    basement_membrane_thickness: float = 0.3  # um (0.3-0.5 um native)

    # === Convolution Parameters ===
    convolution_amplitude: float = 200.0  # um (reduced from native for scaffold)
    convolution_frequency: float = 2.0  # oscillations per mm
    convolution_phase_xy: float = 0.0  # phase offset between XY planes
    enable_3d_convolution: bool = True  # helical vs planar convolution

    # === Scaffold Structure ===
    scaffold_porosity: float = 0.7  # target void fraction
    wall_porosity: float = 0.3  # porosity within tubule wall
    pore_size: float = 10.0  # um (for nutrient diffusion)

    # === Peritubular Capillaries ===
    enable_peritubular_capillaries: bool = False
    capillary_diameter: float = 8.0  # um (5-10 um native)
    capillary_spacing: float = 50.0  # um
    capillary_wrap_angle: float = 180.0  # degrees around tubule

    # === Segment Types (proximal, distal, collecting) ===
    tubule_segment_type: str = 'proximal'  # proximal, distal, collecting, loop_of_henle
    enable_segment_transitions: bool = False
    transition_length: float = 0.5  # mm

    # === Tubule Network ===
    enable_branching: bool = False
    branch_count: int = 0
    branch_angle: float = 45.0  # degrees
    branch_diameter_ratio: float = 0.7  # child/parent diameter

    # === Cell Attachment Features ===
    enable_cell_attachment_sites: bool = False
    attachment_site_diameter: float = 5.0  # um
    attachment_site_spacing: float = 20.0  # um
    attachment_site_depth: float = 2.0  # um

    # === Stochastic Variation ===
    diameter_variance: float = 0.0  # 0-1 random diameter variation
    position_noise: float = 0.0  # 0-1 random position jitter
    seed: int = 42

    # === Resolution ===
    resolution: int = 12  # segments around cylinder


def create_convoluted_path(length: float, amplitude: float, frequency: float,
                           phase_xy: float = 0.0, enable_3d: bool = True,
                           num_points: int = 100) -> np.ndarray:
    """
    Generate a sinusoidal convoluted path in 3D.

    Args:
        length: Total path length (mm)
        amplitude: Amplitude of convolution (mm)
        frequency: Oscillation frequency (per mm)
        phase_xy: Phase offset between X and Y oscillations
        enable_3d: True for helical, False for planar
        num_points: Number of points along path

    Returns:
        Array of (x, y, z) positions along path
    """
    t = np.linspace(0, length, num_points)
    omega = 2 * np.pi * frequency

    # Create 3D sinusoidal path (convoluted in XY plane, progressing in Z)
    x = amplitude * np.sin(omega * t / length * length)

    if enable_3d:
        y = amplitude * np.cos(omega * t / length * length + phase_xy) * 0.7  # Elliptical
    else:
        y = np.zeros_like(t)

    z = t

    return np.column_stack([x, y, z])


def create_hollow_tube_segment(p1: np.ndarray, p2: np.ndarray, outer_radius: float,
                               inner_radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a hollow cylindrical segment between two points.

    Args:
        p1: Start point (x, y, z)
        p2: End point (x, y, z)
        outer_radius: Outer radius
        inner_radius: Inner radius (for hollow lumen)
        resolution: Number of segments around cylinder

    Returns:
        Hollow cylinder manifold
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create outer and inner cylinders
    outer = m3d.Manifold.cylinder(length, outer_radius, outer_radius, resolution)
    inner = m3d.Manifold.cylinder(length + 0.02, inner_radius, inner_radius, resolution)
    inner = inner.translate([0, 0, -0.01])

    hollow = outer - inner

    # Rotate to align with direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        hollow = hollow.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return hollow.translate([p1[0], p1[1], p1[2]])


def get_segment_diameters(segment_type: str) -> tuple[float, float]:
    """
    Get outer and inner diameter ranges for different nephron segment types.

    Args:
        segment_type: 'proximal', 'loop_of_henle', 'distal', or 'collecting'

    Returns:
        Tuple of (outer_diameter_um, inner_diameter_um) for the segment type
    """
    segment_specs = {
        'proximal': (55.0, 20.0),       # 50-60um outer, prominent brush border
        'loop_of_henle': (22.0, 12.0),  # 15-30um thin descending limb
        'distal': (40.0, 25.0),         # 30-50um, no brush border
        'collecting': (50.0, 35.0),     # 40-60um, tapered
    }
    return segment_specs.get(segment_type, (55.0, 20.0))


def create_brush_border_texture(path: np.ndarray, inner_radius: float,
                                 microvilli_height: float, density: float,
                                 resolution: int, seed: int) -> list[m3d.Manifold]:
    """
    Create brush border (microvilli) texture on inner surface of tubule.

    Brush border consists of densely packed microvilli on the luminal surface
    of proximal tubule epithelial cells. Native microvilli: 1-3um height,
    0.1um diameter, ~3000 per cell.

    Args:
        path: Array of (x, y, z) positions along tubule path
        inner_radius: Inner radius of tubule (mm)
        microvilli_height: Height of microvilli (mm)
        density: Fraction of surface covered (0-1)
        resolution: Base resolution for geometry
        seed: Random seed

    Returns:
        List of microvilli manifolds (small cylinders/cones)
    """
    np.random.seed(seed)
    microvilli = []

    if density <= 0 or microvilli_height <= 0:
        return microvilli

    # Calculate spacing based on density
    # Use symbolic representation - larger, fewer structures for printability
    microvilli_radius = microvilli_height * 0.15  # Aspect ratio ~6:1
    microvilli_spacing = microvilli_radius * 4 / max(0.1, density)

    # Number of microvilli rings along path
    path_length = 0
    for i in range(len(path) - 1):
        path_length += np.linalg.norm(path[i+1] - path[i])

    num_rings = max(3, int(path_length / microvilli_spacing))
    ring_indices = np.linspace(0, len(path) - 2, num_rings, dtype=int)

    # Number of microvilli around circumference
    circumference = 2 * np.pi * inner_radius
    num_per_ring = max(6, int(circumference / microvilli_spacing))

    for ring_idx, path_idx in enumerate(ring_indices):
        p1 = path[path_idx]
        p2 = path[min(path_idx + 1, len(path) - 1)]

        # Get direction along path
        direction = p2 - p1
        dir_length = np.linalg.norm(direction)
        if dir_length < 1e-6:
            continue
        direction = direction / dir_length

        # Create local coordinate system
        # Find a perpendicular vector
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Stagger alternate rings
        angle_offset = (ring_idx % 2) * np.pi / num_per_ring

        for i in range(num_per_ring):
            angle = 2 * np.pi * i / num_per_ring + angle_offset

            # Position on inner surface
            radial_dir = perp1 * np.cos(angle) + perp2 * np.sin(angle)
            base_pos = p1 + radial_dir * inner_radius

            # Create small cone/cylinder for microvillus (pointing inward toward lumen center)
            mv = m3d.Manifold.cylinder(
                microvilli_height,
                microvilli_radius,
                microvilli_radius * 0.5,  # Slight taper
                max(4, resolution // 3)
            )

            # Rotate to point inward (toward center)
            # Default cylinder points along Z, need to align with -radial_dir
            # Calculate rotation angles
            inward_dir = -radial_dir
            h = np.sqrt(inward_dir[0]**2 + inward_dir[1]**2)
            if h > 0.001 or abs(inward_dir[2]) > 0.001:
                tilt = np.arctan2(h, inward_dir[2]) * 180 / np.pi
                azim = np.arctan2(inward_dir[1], inward_dir[0]) * 180 / np.pi
                mv = mv.rotate([0, tilt, 0]).rotate([0, 0, azim])

            mv = mv.translate([base_pos[0], base_pos[1], base_pos[2]])
            microvilli.append(mv)

    return microvilli


def create_peritubular_capillaries(path: np.ndarray, outer_radius: float,
                                    capillary_radius: float, spacing: float,
                                    wrap_angle_deg: float,
                                    resolution: int) -> list[m3d.Manifold]:
    """
    Create helical peritubular capillary network wrapping around tubule.

    Peritubular capillaries form a dense network surrounding the proximal
    tubule for reabsorption of water and solutes.

    Args:
        path: Array of (x, y, z) positions along tubule path
        outer_radius: Outer radius of tubule (mm)
        capillary_radius: Radius of capillaries (mm)
        spacing: Axial spacing between capillary rings (mm)
        wrap_angle_deg: Angular extent around tubule (degrees, 180 = half wrap)
        resolution: Cylinder resolution

    Returns:
        List of capillary segment manifolds
    """
    capillaries = []

    if capillary_radius <= 0 or spacing <= 0:
        return capillaries

    # Calculate path length
    path_length = 0
    for i in range(len(path) - 1):
        path_length += np.linalg.norm(path[i+1] - path[i])

    # Number of helical wraps
    num_wraps = max(2, int(path_length / spacing))
    wrap_angle_rad = np.radians(wrap_angle_deg)

    # Placement radius (outside tubule wall)
    placement_radius = outer_radius + capillary_radius * 1.5

    # Create helical segments along path
    points_per_wrap = max(8, int(wrap_angle_deg / 30))

    for wrap_idx in range(num_wraps):
        # Progress along path for this wrap
        t_start = wrap_idx / num_wraps
        t_end = (wrap_idx + 1) / num_wraps

        # Alternate wrap direction for anastomosing pattern
        direction = 1 if wrap_idx % 2 == 0 else -1

        prev_point = None
        for j in range(points_per_wrap + 1):
            # Angular position within this wrap
            angle_frac = j / points_per_wrap
            angle = direction * angle_frac * wrap_angle_rad

            # Position along path
            t = t_start + angle_frac * (t_end - t_start)
            path_idx = min(int(t * (len(path) - 1)), len(path) - 2)
            path_frac = t * (len(path) - 1) - path_idx

            p1 = path[path_idx]
            p2 = path[min(path_idx + 1, len(path) - 1)]
            center = p1 + path_frac * (p2 - p1)

            # Get local coordinate system at this point
            direction_vec = p2 - p1
            dir_length = np.linalg.norm(direction_vec)
            if dir_length < 1e-6:
                continue
            direction_vec = direction_vec / dir_length

            if abs(direction_vec[2]) < 0.9:
                perp1 = np.cross(direction_vec, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(direction_vec, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction_vec, perp1)

            # Calculate position on helix
            radial_pos = perp1 * np.cos(angle) + perp2 * np.sin(angle)
            point = center + radial_pos * placement_radius

            # Connect to previous point
            if prev_point is not None:
                dx = point[0] - prev_point[0]
                dy = point[1] - prev_point[1]
                dz = point[2] - prev_point[2]
                seg_length = np.sqrt(dx*dx + dy*dy + dz*dz)

                if seg_length > capillary_radius * 0.5:
                    cap_seg = m3d.Manifold.cylinder(
                        seg_length, capillary_radius, capillary_radius,
                        max(6, resolution // 2)
                    )

                    # Rotate to align with segment direction
                    h = np.sqrt(dx*dx + dy*dy)
                    if h > 0.001 or abs(dz) > 0.001:
                        tilt = np.arctan2(h, dz) * 180 / np.pi
                        azim = np.arctan2(dy, dx) * 180 / np.pi
                        cap_seg = cap_seg.rotate([0, tilt, 0]).rotate([0, 0, azim])

                    cap_seg = cap_seg.translate([prev_point[0], prev_point[1], prev_point[2]])
                    capillaries.append(cap_seg)

            prev_point = point

    return capillaries


def create_tubule_branches(end_point: np.ndarray, end_direction: np.ndarray,
                            parent_outer_radius: float, parent_inner_radius: float,
                            branch_count: int, branch_angle_deg: float,
                            diameter_ratio: float, branch_length: float,
                            resolution: int) -> list[m3d.Manifold]:
    """
    Create Y-junction branches at end of tubule (for collecting ducts).

    Args:
        end_point: Position of branch point
        end_direction: Direction of parent tubule at branch point
        parent_outer_radius: Outer radius of parent tubule (mm)
        parent_inner_radius: Inner radius of parent tubule (mm)
        branch_count: Number of branches (typically 2)
        branch_angle_deg: Angle between branches (degrees)
        diameter_ratio: Child diameter = parent * ratio (Murray's law ~0.79)
        branch_length: Length of each branch (mm)
        resolution: Cylinder resolution

    Returns:
        List of branch manifolds
    """
    branches = []

    if branch_count <= 0 or branch_length <= 0:
        return branches

    # Calculate child dimensions
    child_outer_radius = parent_outer_radius * diameter_ratio
    child_inner_radius = parent_inner_radius * diameter_ratio

    # Normalize direction
    dir_length = np.linalg.norm(end_direction)
    if dir_length < 1e-6:
        end_direction = np.array([0, 0, 1])
    else:
        end_direction = end_direction / dir_length

    # Find perpendicular vectors for branch plane
    if abs(end_direction[2]) < 0.9:
        perp1 = np.cross(end_direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(end_direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)

    branch_angle_rad = np.radians(branch_angle_deg)

    for i in range(branch_count):
        # Angular position of this branch
        if branch_count == 1:
            angle = 0
        else:
            angle = (i / (branch_count - 1) - 0.5) * branch_angle_rad

        # Calculate branch direction
        branch_dir = (end_direction * np.cos(angle) +
                      perp1 * np.sin(angle))
        branch_dir = branch_dir / np.linalg.norm(branch_dir)

        # Create hollow cylinder for branch
        outer = m3d.Manifold.cylinder(
            branch_length, child_outer_radius, child_outer_radius, resolution
        )
        inner = m3d.Manifold.cylinder(
            branch_length + 0.02, child_inner_radius, child_inner_radius, resolution
        )
        inner = inner.translate([0, 0, -0.01])
        branch = outer - inner

        # Rotate to align with branch direction
        h = np.sqrt(branch_dir[0]**2 + branch_dir[1]**2)
        if h > 0.001 or abs(branch_dir[2]) > 0.001:
            tilt = np.arctan2(h, branch_dir[2]) * 180 / np.pi
            azim = np.arctan2(branch_dir[1], branch_dir[0]) * 180 / np.pi
            branch = branch.rotate([0, tilt, 0]).rotate([0, 0, azim])

        branch = branch.translate([end_point[0], end_point[1], end_point[2]])
        branches.append(branch)

    return branches


def create_cell_attachment_sites(path: np.ndarray, outer_radius: float,
                                  site_radius: float, spacing: float,
                                  depth: float, resolution: int,
                                  seed: int) -> list[m3d.Manifold]:
    """
    Create small indentations on outer surface for cell attachment.

    These represent anchor points on the basement membrane where
    epithelial cells attach.

    Args:
        path: Array of (x, y, z) positions along tubule path
        outer_radius: Outer radius of tubule (mm)
        site_radius: Radius of each attachment site (mm)
        spacing: Distance between sites (mm)
        depth: Depth of indentation (mm)
        resolution: Sphere resolution
        seed: Random seed

    Returns:
        List of sphere manifolds to subtract from tubule wall
    """
    np.random.seed(seed)
    sites = []

    if site_radius <= 0 or spacing <= 0:
        return sites

    # Calculate path length
    path_length = 0
    for i in range(len(path) - 1):
        path_length += np.linalg.norm(path[i+1] - path[i])

    # Number of rings of sites
    num_rings = max(2, int(path_length / spacing))

    # Sites per ring
    circumference = 2 * np.pi * outer_radius
    sites_per_ring = max(4, int(circumference / spacing))

    for ring_idx in range(num_rings):
        # Position along path
        t = (ring_idx + 0.5) / num_rings
        path_idx = min(int(t * (len(path) - 1)), len(path) - 2)
        path_frac = t * (len(path) - 1) - path_idx

        p1 = path[path_idx]
        p2 = path[min(path_idx + 1, len(path) - 1)]
        center = p1 + path_frac * (p2 - p1)

        # Get local coordinate system
        direction = p2 - p1
        dir_length = np.linalg.norm(direction)
        if dir_length < 1e-6:
            continue
        direction = direction / dir_length

        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Stagger alternate rings
        angle_offset = (ring_idx % 2) * np.pi / sites_per_ring

        for i in range(sites_per_ring):
            angle = 2 * np.pi * i / sites_per_ring + angle_offset

            # Position on outer surface
            radial_dir = perp1 * np.cos(angle) + perp2 * np.sin(angle)

            # Place sphere center slightly inside wall surface
            site_pos = center + radial_dir * (outer_radius - depth * 0.3)

            # Create sphere for indentation
            site = m3d.Manifold.sphere(site_radius, max(6, resolution // 2))
            site = site.translate([site_pos[0], site_pos[1], site_pos[2]])
            sites.append(site)

    return sites


def create_wall_pores(path: np.ndarray, outer_radius: float, inner_radius: float,
                       porosity: float, pore_radius: float,
                       resolution: int, seed: int) -> list[m3d.Manifold]:
    """
    Create pores in tubule wall for nutrient diffusion.

    Args:
        path: Array of (x, y, z) positions along tubule path
        outer_radius: Outer radius of tubule (mm)
        inner_radius: Inner radius of tubule (mm)
        porosity: Fraction of wall that is porous (0-1)
        pore_radius: Radius of each pore (mm)
        resolution: Sphere resolution
        seed: Random seed

    Returns:
        List of sphere manifolds to subtract from tubule wall
    """
    np.random.seed(seed)
    pores = []

    if porosity <= 0 or pore_radius <= 0:
        return pores

    wall_thickness = outer_radius - inner_radius
    if wall_thickness <= 0:
        return pores

    # Calculate path length
    path_length = 0
    for i in range(len(path) - 1):
        path_length += np.linalg.norm(path[i+1] - path[i])

    # Calculate number of pores based on wall volume and porosity
    wall_mid_radius = (outer_radius + inner_radius) / 2
    wall_surface = 2 * np.pi * wall_mid_radius * path_length
    pore_area = np.pi * pore_radius**2

    # Number of pores to achieve target porosity (approximate)
    num_pores = max(5, int(wall_surface / pore_area * porosity * 0.3))

    # Distribute pores along path
    for _ in range(num_pores):
        # Random position along path
        t = np.random.uniform(0.05, 0.95)
        path_idx = min(int(t * (len(path) - 1)), len(path) - 2)
        path_frac = t * (len(path) - 1) - path_idx

        p1 = path[path_idx]
        p2 = path[min(path_idx + 1, len(path) - 1)]
        center = p1 + path_frac * (p2 - p1)

        # Get local coordinate system
        direction = p2 - p1
        dir_length = np.linalg.norm(direction)
        if dir_length < 1e-6:
            continue
        direction = direction / dir_length

        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Random angle around circumference
        angle = np.random.uniform(0, 2 * np.pi)
        radial_dir = perp1 * np.cos(angle) + perp2 * np.sin(angle)

        # Random radial position within wall thickness
        radial_pos = np.random.uniform(inner_radius + pore_radius,
                                        outer_radius - pore_radius)
        pore_pos = center + radial_dir * radial_pos

        # Create sphere for pore
        pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 3))
        pore = pore.translate([pore_pos[0], pore_pos[1], pore_pos[2]])
        pores.append(pore)

    return pores


def create_basement_membrane_segment(p1: np.ndarray, p2: np.ndarray,
                                     outer_radius: float, thickness: float,
                                     resolution: int) -> m3d.Manifold:
    """
    Create a thin basement membrane layer segment.

    Args:
        p1: Start point
        p2: End point
        outer_radius: Outer radius of tubule
        thickness: Membrane thickness (mm)
        resolution: Cylinder resolution

    Returns:
        Thin shell manifold for basement membrane
    """
    bm_outer = outer_radius + thickness
    return create_hollow_tube_segment(p1, p2, bm_outer, outer_radius, resolution)


def create_epithelial_layer(path: np.ndarray, inner_radius: float,
                            epithelial_height: float, resolution: int) -> list[m3d.Manifold]:
    """
    Create an inner epithelial cell layer lining the tubule lumen.

    Native PCT epithelial cells are 15-25um tall and form a continuous
    layer on the basement membrane. This creates a visible inner layer
    representing the epithelial cell sheet.

    Args:
        path: Array of (x, y, z) positions along tubule path
        inner_radius: Inner radius of tubule wall (mm) - epithelium outer edge
        epithelial_height: Height/thickness of epithelial layer (mm)
        resolution: Cylinder resolution

    Returns:
        List of epithelial layer segment manifolds
    """
    segments = []

    if epithelial_height <= 0:
        return segments

    # Epithelial layer sits inside the tubule wall
    # Inner surface of epithelium = lumen boundary
    epithelial_outer = inner_radius
    epithelial_inner = inner_radius - epithelial_height

    # Ensure we don't create inverted geometry
    if epithelial_inner <= 0:
        epithelial_inner = inner_radius * 0.3  # Minimum lumen

    for i in range(len(path) - 1):
        seg = create_hollow_tube_segment(
            path[i], path[i + 1],
            epithelial_outer, epithelial_inner,
            resolution
        )
        if seg.num_vert() > 0:
            segments.append(seg)

    return segments


def create_scaffold_support_matrix(path: np.ndarray, outer_radius: float,
                                    scaffold_thickness: float, porosity: float,
                                    pore_size: float, resolution: int,
                                    seed: int) -> tuple[m3d.Manifold, list[m3d.Manifold]]:
    """
    Create a porous interstitial scaffold matrix surrounding the tubule.

    This represents the supporting ECM/hydrogel scaffold that provides
    mechanical support and allows nutrient diffusion to the tubule.

    Args:
        path: Array of (x, y, z) positions along tubule path
        outer_radius: Outer radius of tubule (mm)
        scaffold_thickness: Thickness of surrounding scaffold (mm)
        porosity: Target void fraction (0-1), e.g., 0.7 = 70% void
        pore_size: Size of individual pores (mm)
        resolution: Cylinder resolution
        seed: Random seed for pore placement

    Returns:
        Tuple of (scaffold_shell, pores_to_subtract)
        - scaffold_shell: Solid cylindrical shell around tubule
        - pores_to_subtract: List of sphere manifolds for creating porosity
    """
    np.random.seed(seed)

    if scaffold_thickness <= 0 or porosity <= 0:
        return m3d.Manifold(), []

    # Create the scaffold shell
    scaffold_inner = outer_radius
    scaffold_outer = outer_radius + scaffold_thickness

    shell_segments = []
    for i in range(len(path) - 1):
        seg = create_hollow_tube_segment(
            path[i], path[i + 1],
            scaffold_outer, scaffold_inner,
            resolution
        )
        if seg.num_vert() > 0:
            shell_segments.append(seg)

    if not shell_segments:
        return m3d.Manifold(), []

    scaffold_shell = batch_union(shell_segments)

    # Calculate pores to achieve target porosity
    # Porosity = void volume / total volume
    # We create spherical pores distributed throughout the scaffold
    pores = []

    if porosity > 0 and pore_size > 0:
        pore_radius = pore_size / 2

        # Calculate path length
        path_length = 0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(path[i + 1] - path[i])

        # Scaffold volume (approximate as cylinder)
        scaffold_mid_radius = (scaffold_inner + scaffold_outer) / 2
        scaffold_volume = np.pi * (scaffold_outer**2 - scaffold_inner**2) * path_length

        # Single pore volume
        pore_volume = (4/3) * np.pi * pore_radius**3

        # Number of pores needed for target porosity
        # Account for overlap by using a packing efficiency factor
        packing_efficiency = 0.6  # Spheres don't pack perfectly
        num_pores = max(10, int(scaffold_volume * porosity / pore_volume * packing_efficiency))

        # Limit to reasonable number for performance
        num_pores = min(num_pores, 500)

        for _ in range(num_pores):
            # Random position along path
            t = np.random.uniform(0.02, 0.98)
            path_idx = min(int(t * (len(path) - 1)), len(path) - 2)
            path_frac = t * (len(path) - 1) - path_idx

            p1 = path[path_idx]
            p2 = path[min(path_idx + 1, len(path) - 1)]
            center = p1 + path_frac * (p2 - p1)

            # Get local coordinate system
            direction = p2 - p1
            dir_length = np.linalg.norm(direction)
            if dir_length < 1e-6:
                continue
            direction = direction / dir_length

            if abs(direction[2]) < 0.9:
                perp1 = np.cross(direction, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(direction, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)

            # Random angle around circumference
            angle = np.random.uniform(0, 2 * np.pi)
            radial_dir = perp1 * np.cos(angle) + perp2 * np.sin(angle)

            # Random radial position within scaffold thickness
            # Ensure pore is fully within scaffold
            min_r = scaffold_inner + pore_radius
            max_r = scaffold_outer - pore_radius
            if max_r <= min_r:
                radial_pos = (scaffold_inner + scaffold_outer) / 2
            else:
                radial_pos = np.random.uniform(min_r, max_r)

            pore_pos = center + radial_dir * radial_pos

            # Create sphere for pore
            pore = m3d.Manifold.sphere(pore_radius, max(6, resolution // 2))
            pore = pore.translate([pore_pos[0], pore_pos[1], pore_pos[2]])
            pores.append(pore)

    return scaffold_shell, pores


def generate_kidney_tubule(params: KidneyTubuleParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a kidney tubule scaffold with convoluted path.

    Args:
        params: KidneyTubuleParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, path_length, scaffold_type

    Raises:
        ValueError: If no segments are generated
    """
    np.random.seed(params.seed)

    # Convert um to mm
    outer_radius = params.tubule_outer_diameter / 2000.0
    inner_radius = params.tubule_inner_diameter / 2000.0
    amplitude_mm = params.convolution_amplitude / 1000.0
    bm_thickness_mm = params.basement_membrane_thickness / 1000.0
    microvilli_height_mm = params.microvilli_height / 1000.0
    capillary_radius_mm = params.capillary_diameter / 2000.0
    capillary_spacing_mm = params.capillary_spacing / 1000.0
    attachment_site_radius_mm = params.attachment_site_diameter / 2000.0
    attachment_site_spacing_mm = params.attachment_site_spacing / 1000.0
    attachment_site_depth_mm = params.attachment_site_depth / 1000.0
    pore_radius_mm = params.pore_size / 2000.0
    epithelial_height_mm = params.epithelial_cell_height / 1000.0

    # Scaffold support matrix thickness - proportional to outer diameter
    # Creates a support structure around the tubule for mechanical stability
    scaffold_thickness_mm = outer_radius * 0.5  # 50% of tubule radius as support

    # Handle segment transitions - adjust diameters based on segment type
    if params.enable_segment_transitions:
        seg_outer_um, seg_inner_um = get_segment_diameters(params.tubule_segment_type)
        # Scale by the ratio of provided to default proximal diameter
        scale_factor = params.tubule_outer_diameter / 500.0  # 500 um is default
        outer_radius = seg_outer_um * scale_factor / 1000.0
        inner_radius = seg_inner_um * scale_factor / 1000.0

    # Validate inner radius
    if inner_radius <= 0 or inner_radius >= outer_radius:
        inner_radius = outer_radius * 0.7

    # Generate convoluted path (reduced points for faster generation)
    num_points = max(30, int(params.tubule_length * 8))
    path = create_convoluted_path(
        params.tubule_length,
        amplitude_mm,
        params.convolution_frequency,
        params.convolution_phase_xy,
        params.enable_3d_convolution,
        num_points
    )

    # Apply position noise if enabled
    if params.position_noise > 0:
        noise = np.random.uniform(-1, 1, path.shape) * params.position_noise * amplitude_mm * 0.1
        noise[:, 2] = 0  # Don't add noise to Z progression
        path = path + noise

    # Create tubule segments along path
    segments = []
    bm_segments = []

    # Calculate transition parameters if enabled
    if params.enable_segment_transitions and params.transition_length > 0:
        transition_points = int(params.transition_length / params.tubule_length * len(path))
    else:
        transition_points = 0

    for i in range(len(path) - 1):
        # Apply diameter variance
        if params.diameter_variance > 0:
            variance = 1 + np.random.uniform(-1, 1) * params.diameter_variance * 0.1
            seg_outer = outer_radius * variance
            seg_inner = inner_radius * variance
        else:
            seg_outer = outer_radius
            seg_inner = inner_radius

        # Apply segment transition (taper at start)
        if transition_points > 0 and i < transition_points:
            # Gradual transition from smaller to target diameter
            t = i / transition_points
            seg_outer = seg_outer * (0.7 + 0.3 * t)
            seg_inner = seg_inner * (0.7 + 0.3 * t)

        seg = create_hollow_tube_segment(
            path[i], path[i+1],
            seg_outer, seg_inner,
            params.resolution
        )
        if seg.num_vert() > 0:
            segments.append(seg)

        # Create basement membrane if enabled
        if params.enable_basement_membrane:
            bm_seg = create_basement_membrane_segment(
                path[i], path[i+1],
                seg_outer, bm_thickness_mm,
                max(8, params.resolution - 2)
            )
            if bm_seg.num_vert() > 0:
                bm_segments.append(bm_seg)

    if not segments:
        raise ValueError("No tubule segments generated")

    # Union all segments using batch_union for efficiency
    result = batch_union(segments)

    # Add basement membrane
    if bm_segments:
        bm_union = batch_union(bm_segments)
        result = result + bm_union

    # Track statistics for new features
    brush_border_count = 0
    capillary_count = 0
    branch_count = 0
    attachment_site_count = 0
    pore_count = 0
    epithelial_layer_count = 0
    scaffold_pore_count = 0

    # === Add epithelial cell layer (uses epithelial_cell_height) ===
    # Creates an inner layer representing the epithelial cell sheet lining the tubule
    if epithelial_height_mm > 0:
        epithelial_segments = create_epithelial_layer(
            path, inner_radius, epithelial_height_mm, params.resolution
        )
        if epithelial_segments:
            epithelial_layer_count = len(epithelial_segments)
            epithelial_union = batch_union(epithelial_segments)
            result = result + epithelial_union

    # === Add scaffold support matrix (uses scaffold_porosity) ===
    # Creates a porous interstitial scaffold surrounding the tubule
    if params.scaffold_porosity > 0:
        scaffold_shell, scaffold_pores = create_scaffold_support_matrix(
            path, outer_radius + bm_thickness_mm,
            scaffold_thickness_mm, params.scaffold_porosity,
            pore_radius_mm * 2,  # Scaffold pores larger than wall pores
            params.resolution, params.seed + 100
        )
        if scaffold_shell.num_vert() > 0:
            result = result + scaffold_shell
        if scaffold_pores:
            scaffold_pore_count = len(scaffold_pores)
            # These pores will be subtracted later with other features
            features_to_subtract_scaffold = scaffold_pores
        else:
            features_to_subtract_scaffold = []
    else:
        features_to_subtract_scaffold = []

    # === Add brush border texture (microvilli) ===
    if params.enable_brush_border_texture and params.tubule_segment_type == 'proximal':
        microvilli = create_brush_border_texture(
            path, inner_radius, microvilli_height_mm,
            params.brush_border_density, params.resolution, params.seed
        )
        if microvilli:
            brush_border_count = len(microvilli)
            microvilli_union = batch_union(microvilli)
            result = result + microvilli_union

    # === Add peritubular capillaries ===
    if params.enable_peritubular_capillaries:
        capillaries = create_peritubular_capillaries(
            path, outer_radius + bm_thickness_mm,
            capillary_radius_mm, capillary_spacing_mm,
            params.capillary_wrap_angle, params.resolution
        )
        if capillaries:
            capillary_count = len(capillaries)
            capillary_union = batch_union(capillaries)
            result = result + capillary_union

    # === Add branches (for collecting ducts) ===
    if params.enable_branching and params.branch_count > 0:
        # Get end point and direction
        end_point = path[-1]
        if len(path) >= 2:
            end_direction = path[-1] - path[-2]
        else:
            end_direction = np.array([0, 0, 1])

        # Branch length proportional to tubule length
        branch_length_mm = params.tubule_length * 0.2

        branches = create_tubule_branches(
            end_point, end_direction,
            outer_radius, inner_radius,
            params.branch_count, params.branch_angle,
            params.branch_diameter_ratio, branch_length_mm,
            params.resolution
        )
        if branches:
            branch_count = len(branches)
            branch_union = batch_union(branches)
            result = result + branch_union

    # === Collect features to subtract ===
    features_to_subtract = []

    # Add scaffold pores (from scaffold_porosity)
    if features_to_subtract_scaffold:
        features_to_subtract.extend(features_to_subtract_scaffold)

    # === Add cell attachment sites (subtractive) ===
    if params.enable_cell_attachment_sites:
        attachment_sites = create_cell_attachment_sites(
            path, outer_radius + bm_thickness_mm,
            attachment_site_radius_mm, attachment_site_spacing_mm,
            attachment_site_depth_mm, params.resolution, params.seed
        )
        if attachment_sites:
            attachment_site_count = len(attachment_sites)
            features_to_subtract.extend(attachment_sites)

    # === Add wall porosity (subtractive) ===
    if params.wall_porosity > 0:
        wall_pores = create_wall_pores(
            path, outer_radius, inner_radius,
            params.wall_porosity, pore_radius_mm,
            params.resolution, params.seed + 1
        )
        if wall_pores:
            pore_count = len(wall_pores)
            features_to_subtract.extend(wall_pores)

    # Subtract all features at once
    if features_to_subtract:
        subtract_union = batch_union(features_to_subtract)
        result = result - subtract_union

    # Verify we still have geometry
    if result.num_vert() == 0:
        raise ValueError("No geometry remaining after feature subtraction")

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    # Calculate actual path length
    path_length = 0
    for i in range(len(path) - 1):
        dx = path[i+1] - path[i]
        path_length += np.linalg.norm(dx)

    # Calculate wall thickness
    wall_thickness = outer_radius - inner_radius

    stats = {
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'path_length_mm': path_length,
        'wall_thickness_mm': wall_thickness,
        'outer_diameter_mm': outer_radius * 2,
        'inner_diameter_mm': inner_radius * 2,
        'segment_type': params.tubule_segment_type,
        'scaffold_type': 'kidney_tubule',
        # Epithelial layer statistics (uses epithelial_cell_height)
        'epithelial_layer_count': epithelial_layer_count,
        'epithelial_cell_height_um': params.epithelial_cell_height,
        'has_epithelial_layer': epithelial_height_mm > 0 and epithelial_layer_count > 0,
        # Scaffold support matrix statistics (uses scaffold_porosity)
        'scaffold_pore_count': scaffold_pore_count,
        'scaffold_porosity': params.scaffold_porosity,
        'scaffold_thickness_mm': scaffold_thickness_mm if params.scaffold_porosity > 0 else 0,
        'has_scaffold_matrix': params.scaffold_porosity > 0,
        # Other feature statistics
        'brush_border_count': brush_border_count,
        'capillary_count': capillary_count,
        'branch_count': branch_count,
        'attachment_site_count': attachment_site_count,
        'pore_count': pore_count,
        'has_brush_border': params.enable_brush_border_texture and brush_border_count > 0,
        'has_capillaries': params.enable_peritubular_capillaries and capillary_count > 0,
        'has_branches': params.enable_branching and branch_count > 0,
        'has_attachment_sites': params.enable_cell_attachment_sites and attachment_site_count > 0,
        'has_wall_pores': params.wall_porosity > 0 and pore_count > 0,
    }

    return result, stats


def generate_kidney_tubule_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate kidney tubule from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching KidneyTubuleParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle legacy parameters
    tubule_diameter = params.get('tubule_diameter', params.get('tubule_outer_diameter', 500.0))
    wall_thickness = params.get('wall_thickness', 75.0)  # Legacy: derived from outer-inner
    length = params.get('length', params.get('tubule_length', 15.0))

    # Calculate inner diameter from legacy wall_thickness if not explicitly provided
    if 'tubule_inner_diameter' in params:
        inner_diameter = params['tubule_inner_diameter']
    else:
        inner_diameter = tubule_diameter - 2 * wall_thickness

    return generate_kidney_tubule(KidneyTubuleParams(
        # Tubule geometry
        tubule_outer_diameter=tubule_diameter,
        tubule_inner_diameter=inner_diameter,
        tubule_length=length,

        # Epithelial features
        epithelial_cell_height=params.get('epithelial_cell_height', 20.0),
        microvilli_height=params.get('microvilli_height', 2.7),
        enable_brush_border_texture=params.get('enable_brush_border_texture', False),
        brush_border_density=params.get('brush_border_density', 0.8),

        # Basement membrane
        enable_basement_membrane=params.get('enable_basement_membrane', True),
        basement_membrane_thickness=params.get('basement_membrane_thickness', 0.3),

        # Convolution
        convolution_amplitude=params.get('convolution_amplitude', 200.0),
        convolution_frequency=params.get('convolution_frequency', 2.0),
        convolution_phase_xy=params.get('convolution_phase_xy', 0.0),
        enable_3d_convolution=params.get('enable_3d_convolution', True),

        # Scaffold structure
        scaffold_porosity=params.get('scaffold_porosity', 0.7),
        wall_porosity=params.get('wall_porosity', 0.3),
        pore_size=params.get('pore_size', 10.0),

        # Peritubular capillaries
        enable_peritubular_capillaries=params.get('enable_peritubular_capillaries', False),
        capillary_diameter=params.get('capillary_diameter', 8.0),
        capillary_spacing=params.get('capillary_spacing', 50.0),
        capillary_wrap_angle=params.get('capillary_wrap_angle', 180.0),

        # Segment type
        tubule_segment_type=params.get('tubule_segment_type', 'proximal'),
        enable_segment_transitions=params.get('enable_segment_transitions', False),
        transition_length=params.get('transition_length', 0.5),

        # Branching
        enable_branching=params.get('enable_branching', False),
        branch_count=params.get('branch_count', 0),
        branch_angle=params.get('branch_angle', 45.0),
        branch_diameter_ratio=params.get('branch_diameter_ratio', 0.7),

        # Cell attachment
        enable_cell_attachment_sites=params.get('enable_cell_attachment_sites', False),
        attachment_site_diameter=params.get('attachment_site_diameter', 5.0),
        attachment_site_spacing=params.get('attachment_site_spacing', 20.0),
        attachment_site_depth=params.get('attachment_site_depth', 2.0),

        # Stochastic variation
        diameter_variance=params.get('diameter_variance', 0.0),
        position_noise=params.get('position_noise', 0.0),
        seed=params.get('seed', 42),

        # Resolution
        resolution=params.get('resolution', 12)
    ))
