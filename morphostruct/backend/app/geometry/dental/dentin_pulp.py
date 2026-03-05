"""
Dentin-Pulp scaffold generator for tooth-like structures.

Provides parametric generation of tooth scaffolds with:
- Outer mineralized dentin shell (crown + root)
- Inner hollow pulp chamber with anatomical horns
- Crown (dome-shaped) + root (tapered cylinder)
- Multi-root support for molars
- Optional enamel shell on crown
- Optional cementum layer on root
- DEJ scallop texture
- Simplified tubule representation

Biologically realistic parameters based on:
- Tubule diameter: 0.8-3.0 μm gradient (DEJ to pulp) (Pashley 1989)
- Tubule density: 19,000/mm² at DEJ, 55,000/mm² near pulp
- Pulp chamber: varies by tooth type (~2.5-4mm height typical)
- DEJ scallop: 25-100 μm convexities (Marshall 1997)
- Enamel: 0.5mm cervical to 2.5mm occlusal
- Cementum: 50-200 μm on root surface
- Root canal taper: 0.04-0.08 mm/mm typical
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..core import batch_union


@dataclass
class DentinPulpParams:
    """
    Parameters for dentin-pulp scaffold generation.

    Biologically realistic defaults based on human molar literature.
    """
    # === Basic Geometry ===
    tooth_height: float = 17.0  # mm, total height (crown + root) - must be > root_length
    crown_diameter: float = 7.0  # mm, width of crown (mesiodistal)
    crown_height: float = 5.0  # mm, height of anatomical crown
    root_length: float = 12.0  # mm, length of root portion (molar ~12mm)
    root_diameter: float = 3.0  # mm, diameter at root tip

    # === Dentin Tubule Parameters (Microstructure) ===
    tubule_diameter_dej: float = 1.0  # μm (0.9-1.2 um at DEJ)
    tubule_diameter_pulp: float = 3.7  # μm, tubule diameter near pulp
    tubule_density_dej: float = 1.9e6  # per cm², tubule density at DEJ (~19,000/mm²)
    tubule_density_pulp: float = 5.75e6  # per cm² (57,500/mm^2 near pulp)
    enable_tubule_representation: bool = False  # simplified tubule channels (performance intensive)
    tubule_resolution: int = 4  # segments per tubule cylinder

    # === Pulp Chamber ===
    pulp_chamber_height: float = 2.76  # mm, typical pulp chamber height
    pulp_chamber_width: float = 2.5  # mm, pulp chamber buccolingual width
    pulp_chamber_size: float = 0.4  # ratio (0-1), relative size of pulp chamber (legacy)
    pulp_horn_count: int = 2  # number of pulp horns (2-5 for molars)
    pulp_horn_height: float = 0.5  # mm, additional height for pulp horns
    root_canal_taper: float = 0.06  # mm/mm, root canal taper (0.04-0.08 typical)

    # === Dentin-Enamel Junction (DEJ) ===
    dej_scallop_size: float = 75.0  # μm, DEJ scallop convexity size (25-100 μm)
    dej_scallop_count: int = 12  # number of scallops around circumference
    dej_width: float = 9.0  # μm, DEJ interface width (~8-10 μm)
    enable_dej_texture: bool = True  # add DEJ scallop texture

    # === Dentin Layers ===
    dentin_thickness: float = 2.0  # mm (1.5-2.2 mm typical)
    peritubular_dentin_ratio: float = 0.15  # ratio of highly mineralized peritubular dentin
    intertubular_dentin_ratio: float = 0.85  # ratio of less mineralized intertubular dentin
    mantle_dentin_thickness: float = 0.025  # mm (15-30 um native)

    # === Enamel Interface ===
    enamel_interface_roughness: float = 0.5  # 0-1, surface roughness at crown
    enamel_thickness_occlusal: float = 2.5  # mm, enamel thickness at occlusal surface
    enamel_thickness_cervical: float = 0.5  # mm, enamel thickness at cervical margin
    enable_enamel_shell: bool = False  # add enamel outer shell (crown only)

    # === Root Features ===
    root_count: int = 1  # number of roots (1 for anterior, 2-3 for molars)
    root_furcation_height: float = 3.0  # mm, height where roots divide (multi-rooted)
    cementum_thickness: float = 0.1  # mm, cementum layer on root (~50-200 μm)
    enable_cementum: bool = False  # add cementum layer on root

    # === Generation Settings ===
    resolution: int = 16  # circular resolution for mesh generation
    seed: int = 42  # random seed for stochastic features
    randomness: float = 0.1  # 0-1, overall randomness intensity
    detail_level: Literal['low', 'medium', 'high'] = 'medium'  # mesh detail level


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_resolution(base_resolution: int, detail_level: str) -> int:
    """
    Adjust resolution based on detail level.

    Args:
        base_resolution: Base resolution value
        detail_level: 'low', 'medium', or 'high'

    Returns:
        Adjusted resolution value
    """
    if detail_level == 'low':
        return max(8, base_resolution // 2)
    elif detail_level == 'high':
        return base_resolution * 2
    return base_resolution  # 'medium'


def _create_crown(params: DentinPulpParams, res: int) -> m3d.Manifold:
    """
    Create the anatomical crown portion of the tooth.

    The crown is a dome-shaped structure (hemisphere scaled vertically).

    Args:
        params: Tooth parameters
        res: Mesh resolution

    Returns:
        Crown manifold positioned at correct height
    """
    crown_radius = params.crown_diameter / 2
    crown_actual_height = params.crown_height

    # Create sphere and cut to get dome (upper hemisphere)
    crown_sphere = m3d.Manifold.sphere(crown_radius, res)

    # Cut plane to create dome (keep upper portion)
    cut_box = m3d.Manifold.cube(
        [params.crown_diameter * 2, params.crown_diameter * 2, crown_radius],
        center=True
    )
    cut_box = cut_box.translate([0, 0, crown_radius / 2])
    crown = m3d.Manifold.batch_boolean([crown_sphere, cut_box], m3d.OpType.Intersect)

    # Scale crown vertically to desired height
    if crown_radius > 0:
        crown = crown.scale([1, 1, crown_actual_height / crown_radius])

    # Position crown at top of root
    crown = crown.translate([0, 0, params.root_length])

    return crown


def _create_single_root(
    params: DentinPulpParams,
    res: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    angle: float = 0.0
) -> m3d.Manifold:
    """
    Create a single root with optional offset and angle.

    Args:
        params: Tooth parameters
        res: Mesh resolution
        offset_x: X offset from center
        offset_y: Y offset from center
        angle: Outward angle in degrees

    Returns:
        Root manifold positioned appropriately
    """
    crown_radius = params.crown_diameter / 2
    root_top_radius = crown_radius * 0.6 if params.root_count > 1 else crown_radius
    root_bottom_radius = params.root_diameter / 2

    root = m3d.Manifold.cylinder(
        params.root_length,
        root_bottom_radius,
        root_top_radius,
        res
    )

    # Apply angle if specified (tilt outward)
    if angle != 0:
        # Rotate around Y axis for outward tilt
        root = root.rotate([0, angle, 0])

    # Apply offset
    if offset_x != 0 or offset_y != 0:
        root = root.translate([offset_x, offset_y, 0])

    return root


def _create_roots(
    params: DentinPulpParams,
    res: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create all roots for the tooth (1-3 roots supported).

    Multi-rooted teeth have roots positioned symmetrically and angled outward.
    Roots join at the furcation height.

    Args:
        params: Tooth parameters
        res: Mesh resolution
        rng: Random number generator

    Returns:
        List of root manifolds
    """
    roots = []
    crown_radius = params.crown_diameter / 2

    if params.root_count == 1:
        # Single root - simple tapered cylinder
        roots.append(_create_single_root(params, res))

    elif params.root_count == 2:
        # Two roots - mesial and distal (buccal/lingual for molars)
        spread = crown_radius * 0.3
        angle = 5 + rng.random() * params.randomness * 3  # 5-8 degrees outward

        # Create two roots offset in X direction
        root1 = _create_single_root(params, res, offset_x=-spread, angle=-angle)
        root2 = _create_single_root(params, res, offset_x=spread, angle=angle)

        # Add furcation connecting piece at top
        furcation_height = min(params.root_furcation_height, params.root_length * 0.8)
        furcation_box = m3d.Manifold.cube(
            [spread * 2.5, crown_radius * 1.2, params.root_length - furcation_height],
            center=True
        )
        furcation_box = furcation_box.translate([0, 0, furcation_height + (params.root_length - furcation_height) / 2])

        roots.extend([root1, root2, furcation_box])

    elif params.root_count >= 3:
        # Three roots - typical molar configuration (2 buccal, 1 palatal)
        spread = crown_radius * 0.35
        angle = 6 + rng.random() * params.randomness * 4  # 6-10 degrees

        # Mesiobuccal root
        root1 = _create_single_root(params, res, offset_x=-spread * 0.7, offset_y=spread * 0.5, angle=-angle)
        # Distobuccal root
        root2 = _create_single_root(params, res, offset_x=spread * 0.7, offset_y=spread * 0.5, angle=angle)
        # Palatal root (larger, more central)
        root3 = _create_single_root(params, res, offset_x=0, offset_y=-spread * 0.8, angle=0)

        # Furcation connecting piece
        furcation_height = min(params.root_furcation_height, params.root_length * 0.75)
        furcation_cyl = m3d.Manifold.cylinder(
            params.root_length - furcation_height,
            crown_radius * 0.6,
            crown_radius * 0.8,
            res
        )
        furcation_cyl = furcation_cyl.translate([0, 0, furcation_height])

        roots.extend([root1, root2, root3, furcation_cyl])

    return roots


def _create_pulp_chamber(
    params: DentinPulpParams,
    res: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the anatomically accurate pulp chamber with horns and tapered canal.

    The pulp chamber includes:
    - Main chamber sized by pulp_chamber_height and pulp_chamber_width
    - Pulp horns as spherical protrusions under cusps
    - Tapered root canal(s) with anatomical taper rate

    Args:
        params: Tooth parameters
        res: Mesh resolution
        rng: Random number generator

    Returns:
        Pulp chamber manifold
    """
    # Main pulp chamber dimensions
    chamber_width = params.pulp_chamber_width
    chamber_height = params.pulp_chamber_height

    # Create ellipsoidal main chamber
    chamber_radius_xy = chamber_width / 2
    chamber_radius_z = chamber_height / 2

    # Use sphere scaled to ellipsoid for smooth chamber
    main_chamber = m3d.Manifold.sphere(chamber_radius_xy, res)
    if chamber_radius_z != chamber_radius_xy:
        main_chamber = main_chamber.scale([1, 1, chamber_radius_z / chamber_radius_xy])

    # Position chamber at correct height (just below crown top)
    chamber_z = params.root_length + params.crown_height * 0.3
    main_chamber = main_chamber.translate([0, 0, chamber_z])

    # Add pulp horns
    horn_parts = []
    if params.pulp_horn_count > 0:
        horn_radius = chamber_width * 0.15  # Horn radius proportional to chamber
        horn_height = params.pulp_horn_height

        for i in range(params.pulp_horn_count):
            # Distribute horns evenly around the chamber top
            angle = (2 * np.pi * i) / params.pulp_horn_count
            # Add slight randomness to position
            angle += rng.random() * params.randomness * 0.3

            # Horn positioned at edge of chamber, pointing up
            horn_x = chamber_radius_xy * 0.5 * np.cos(angle)
            horn_y = chamber_radius_xy * 0.5 * np.sin(angle)
            horn_z = chamber_z + chamber_radius_z

            # Create horn as elongated sphere
            horn = m3d.Manifold.sphere(horn_radius, max(8, res // 2))
            horn = horn.scale([1, 1, (horn_height + horn_radius) / horn_radius])
            horn = horn.translate([horn_x, horn_y, horn_z])

            horn_parts.append(horn)

    # Create root canal(s)
    canal_parts = []
    crown_radius = params.crown_diameter / 2

    # Canal dimensions with taper
    canal_top_radius = chamber_width * 0.25  # Opening at chamber
    canal_bottom_radius = canal_top_radius * (1 - params.root_canal_taper * params.root_length)
    canal_bottom_radius = max(0.1, canal_bottom_radius)  # Ensure positive radius

    if params.root_count == 1:
        # Single canal
        canal = m3d.Manifold.cylinder(
            params.root_length * 0.95,
            canal_bottom_radius,
            canal_top_radius,
            max(6, res // 2)
        )
        canal = canal.translate([0, 0, params.root_length * 0.025])
        canal_parts.append(canal)

    elif params.root_count == 2:
        # Two canals
        spread = crown_radius * 0.25
        for offset_x in [-spread, spread]:
            canal = m3d.Manifold.cylinder(
                params.root_length * 0.9,
                canal_bottom_radius * 0.8,
                canal_top_radius * 0.7,
                max(6, res // 2)
            )
            canal = canal.translate([offset_x, 0, params.root_length * 0.05])
            canal_parts.append(canal)

    else:  # 3+ roots
        # Three canals
        positions = [
            (-crown_radius * 0.2, crown_radius * 0.15),   # Mesiobuccal
            (crown_radius * 0.2, crown_radius * 0.15),    # Distobuccal
            (0, -crown_radius * 0.25)                     # Palatal
        ]
        for px, py in positions:
            canal = m3d.Manifold.cylinder(
                params.root_length * 0.85,
                canal_bottom_radius * 0.7,
                canal_top_radius * 0.6,
                max(6, res // 2)
            )
            canal = canal.translate([px, py, params.root_length * 0.075])
            canal_parts.append(canal)

    # Combine all pulp components
    pulp = main_chamber
    for horn in horn_parts:
        pulp = pulp + horn
    for canal in canal_parts:
        pulp = pulp + canal

    return pulp


def _create_enamel_shell(
    params: DentinPulpParams,
    res: int
) -> m3d.Manifold:
    """
    Create the enamel outer shell on the crown.

    Enamel thickness varies from occlusal (thicker) to cervical (thinner).
    Only covers the crown portion, not the root.

    Args:
        params: Tooth parameters
        res: Mesh resolution

    Returns:
        Enamel shell manifold
    """
    crown_radius = params.crown_diameter / 2
    crown_actual_height = params.crown_height

    # Outer enamel dimensions (add thickness to crown)
    outer_radius_top = crown_radius + params.enamel_thickness_occlusal
    outer_radius_bottom = crown_radius + params.enamel_thickness_cervical

    # Create outer enamel surface as scaled sphere dome
    avg_enamel = (params.enamel_thickness_occlusal + params.enamel_thickness_cervical) / 2
    outer_sphere = m3d.Manifold.sphere(crown_radius + avg_enamel, res)

    # Cut to hemisphere
    cut_box = m3d.Manifold.cube(
        [params.crown_diameter * 3, params.crown_diameter * 3, crown_radius + avg_enamel],
        center=True
    )
    cut_box = cut_box.translate([0, 0, (crown_radius + avg_enamel) / 2])
    outer_crown = m3d.Manifold.batch_boolean([outer_sphere, cut_box], m3d.OpType.Intersect)

    # Scale to desired height plus enamel
    scale_z = (crown_actual_height + params.enamel_thickness_occlusal) / (crown_radius + avg_enamel)
    outer_crown = outer_crown.scale([1, 1, scale_z])

    # Create inner surface (original crown shape) to subtract
    inner_sphere = m3d.Manifold.sphere(crown_radius, res)
    inner_cut_box = m3d.Manifold.cube(
        [params.crown_diameter * 2, params.crown_diameter * 2, crown_radius],
        center=True
    )
    inner_cut_box = inner_cut_box.translate([0, 0, crown_radius / 2])
    inner_crown = m3d.Manifold.batch_boolean([inner_sphere, inner_cut_box], m3d.OpType.Intersect)
    inner_crown = inner_crown.scale([1, 1, crown_actual_height / crown_radius])

    # Subtract inner from outer to get shell
    enamel = outer_crown - inner_crown

    # Position at root top
    enamel = enamel.translate([0, 0, params.root_length])

    return enamel


def _create_cementum_layer(
    params: DentinPulpParams,
    res: int
) -> m3d.Manifold:
    """
    Create the cementum layer on the root surface.

    Cementum is a thin mineralized layer (50-200 μm) covering the root.

    Args:
        params: Tooth parameters
        res: Mesh resolution

    Returns:
        Cementum layer manifold
    """
    crown_radius = params.crown_diameter / 2
    root_top_radius = crown_radius
    root_bottom_radius = params.root_diameter / 2
    thickness = params.cementum_thickness

    # Outer cementum surface
    outer = m3d.Manifold.cylinder(
        params.root_length,
        root_bottom_radius + thickness,
        root_top_radius + thickness,
        res
    )

    # Inner surface (original root)
    inner = m3d.Manifold.cylinder(
        params.root_length + 0.01,  # Slightly longer to ensure clean subtraction
        root_bottom_radius,
        root_top_radius,
        res
    )
    inner = inner.translate([0, 0, -0.005])

    # Cementum is the shell
    cementum = outer - inner

    return cementum


def _add_dej_texture(
    dentin: m3d.Manifold,
    params: DentinPulpParams,
    res: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Add DEJ (dentin-enamel junction) scallop texture to crown surface.

    DEJ has characteristic scalloped interface (25-100 μm convexities).
    This creates small bumps on the crown surface to represent this texture.

    Args:
        dentin: Base dentin manifold
        params: Tooth parameters
        res: Mesh resolution
        rng: Random number generator

    Returns:
        Dentin with DEJ texture added
    """
    crown_radius = params.crown_diameter / 2
    scallop_size = params.dej_scallop_size / 1000.0  # Convert μm to mm
    scallop_count = params.dej_scallop_count

    if scallop_size <= 0 or scallop_count <= 0:
        return dentin

    scallops = []

    # Create scallop bumps around the crown circumference at multiple heights
    height_levels = 3  # Number of vertical levels of scallops
    for level in range(height_levels):
        z_offset = params.root_length + params.crown_height * (0.3 + 0.5 * level / height_levels)

        for i in range(scallop_count):
            angle = (2 * np.pi * i) / scallop_count
            # Add randomness
            angle += rng.random() * params.randomness * 0.2
            size_var = 1.0 + (rng.random() - 0.5) * params.randomness

            # Position on crown surface
            r = crown_radius * (0.85 + 0.1 * level / height_levels)
            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Create small spherical bump
            bump = m3d.Manifold.sphere(scallop_size * size_var, max(4, res // 4))
            bump = bump.translate([x, y, z_offset])
            scallops.append(bump)

    if scallops:
        scallop_union = batch_union(scallops)
        if scallop_union is not None:
            dentin = dentin + scallop_union

    return dentin


def _create_tubule_channels(
    params: DentinPulpParams,
    res: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create simplified radial tubule channels for visualization.

    Creates sparse symbolic channels (8-16) to represent the tubule pattern
    without generating the actual millions of tubules.

    Tubule diameter and density gradient from DEJ to pulp is visualized
    by varying channel size.

    Args:
        params: Tooth parameters
        res: Mesh resolution
        rng: Random number generator

    Returns:
        List of tubule channel manifolds
    """
    channels = []
    crown_radius = params.crown_diameter / 2
    pulp_radius = params.pulp_chamber_width / 2

    # Convert μm to mm
    diam_dej = params.tubule_diameter_dej / 1000.0
    diam_pulp = params.tubule_diameter_pulp / 1000.0

    # Scale up for visibility (actual tubules are too small to render)
    scale_factor = 50  # Make tubules visible in model
    diam_dej *= scale_factor
    diam_pulp *= scale_factor

    # Number of symbolic tubule channels (sparse representation)
    n_channels = 8 + int(8 * (params.tubule_resolution / 8))

    # Create radial channels in crown region
    chamber_z = params.root_length + params.crown_height * 0.3

    for i in range(n_channels):
        angle = (2 * np.pi * i) / n_channels
        angle += rng.random() * params.randomness * 0.1

        # Channel goes from pulp surface to near DEJ
        inner_r = pulp_radius * 0.8
        outer_r = crown_radius * 0.9

        # Create tapered cylinder (wider near pulp)
        length = outer_r - inner_r
        channel = m3d.Manifold.cylinder(
            length,
            diam_dej / 2,  # Smaller at DEJ end
            diam_pulp / 2,  # Larger at pulp end
            params.tubule_resolution
        )

        # Rotate to point outward radially
        channel = channel.rotate([0, 90, 0])

        # Position and rotate around center
        channel = channel.translate([inner_r + length / 2, 0, chamber_z])
        channel = channel.rotate([0, 0, np.degrees(angle)])

        channels.append(channel)

    # Add channels in root region (fewer, more sparse)
    n_root_channels = max(4, n_channels // 2)
    for i in range(n_root_channels):
        angle = (2 * np.pi * i) / n_root_channels
        angle += rng.random() * params.randomness * 0.15

        # Height along root
        z_pos = params.root_length * (0.3 + 0.5 * i / n_root_channels)

        # Radii at this height (interpolate for tapered root)
        t = z_pos / params.root_length
        root_r = params.root_diameter / 2 + t * (crown_radius - params.root_diameter / 2)
        canal_r = params.pulp_chamber_width * 0.2 * (0.5 + 0.5 * t)

        inner_r = canal_r * 1.2
        outer_r = root_r * 0.85
        length = max(0.1, outer_r - inner_r)

        channel = m3d.Manifold.cylinder(
            length,
            diam_dej * 0.7 / 2,
            diam_pulp * 0.7 / 2,
            params.tubule_resolution
        )

        channel = channel.rotate([0, 90, 0])
        channel = channel.translate([inner_r + length / 2, 0, z_pos])
        channel = channel.rotate([0, 0, np.degrees(angle)])

        channels.append(channel)

    return channels


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_dentin_pulp(params: DentinPulpParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a dentin-pulp tooth scaffold.

    Creates anatomically accurate tooth geometry with:
    - Crown and root structure (multi-root support)
    - Pulp chamber with horns and tapered canals
    - Optional enamel shell on crown
    - Optional cementum layer on root
    - Optional DEJ scallop texture
    - Optional simplified tubule representation

    Args:
        params: DentinPulpParams specifying tooth geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, scaffold_type,
                      and all parameter values

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate parameters
    if params.pulp_chamber_size <= 0 or params.pulp_chamber_size >= 1:
        raise ValueError("pulp_chamber_size must be between 0 and 1")
    if params.dentin_thickness <= 0:
        raise ValueError("dentin_thickness must be positive")
    if params.crown_diameter <= params.root_diameter:
        raise ValueError("crown_diameter must be larger than root_diameter")
    if params.root_count < 1 or params.root_count > 3:
        raise ValueError("root_count must be 1, 2, or 3")
    if params.pulp_horn_count < 0 or params.pulp_horn_count > 5:
        raise ValueError("pulp_horn_count must be 0-5")

    crown_height_calc = params.tooth_height - params.root_length
    if crown_height_calc <= 0:
        raise ValueError("tooth_height must be greater than root_length")

    # Set up RNG for reproducibility
    rng = np.random.default_rng(params.seed)

    # Adjust resolution based on detail_level
    res = _get_resolution(params.resolution, params.detail_level)

    # Build outer tooth shape (crown + roots)
    outer_crown = _create_crown(params, res)
    roots = _create_roots(params, res, rng)

    if roots:
        root_union = batch_union(roots)
        if root_union is not None:
            outer_tooth = outer_crown + root_union
        else:
            outer_tooth = outer_crown
    else:
        outer_tooth = outer_crown

    # Build pulp chamber with horns and tapered canal
    pulp = _create_pulp_chamber(params, res, rng)

    # Subtract pulp from outer to create dentin shell
    result = outer_tooth - pulp

    # Add enamel shell if enabled
    if params.enable_enamel_shell:
        enamel = _create_enamel_shell(params, res)
        result = result + enamel

    # Add cementum if enabled
    if params.enable_cementum:
        cementum = _create_cementum_layer(params, res)
        result = result + cementum

    # Add DEJ texture if enabled
    if params.enable_dej_texture:
        result = _add_dej_texture(result, params, res, rng)

    # Add tubule representation if enabled
    if params.enable_tubule_representation:
        tubules = _create_tubule_channels(params, res, rng)
        if tubules:
            tubule_union = batch_union(tubules)
            if tubule_union is not None:
                result = result - tubule_union

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0.0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'scaffold_type': 'dentin_pulp',
        # Basic Geometry
        'tooth_height': params.tooth_height,
        'crown_diameter': params.crown_diameter,
        'crown_height': params.crown_height,
        'root_length': params.root_length,
        'root_diameter': params.root_diameter,
        # Tubule Parameters
        'tubule_diameter_dej': params.tubule_diameter_dej,
        'tubule_diameter_pulp': params.tubule_diameter_pulp,
        'tubule_density_dej': params.tubule_density_dej,
        'tubule_density_pulp': params.tubule_density_pulp,
        'enable_tubule_representation': params.enable_tubule_representation,
        'tubule_resolution': params.tubule_resolution,
        # Pulp Chamber
        'pulp_chamber_height': params.pulp_chamber_height,
        'pulp_chamber_width': params.pulp_chamber_width,
        'pulp_chamber_size': params.pulp_chamber_size,
        'pulp_horn_count': params.pulp_horn_count,
        'pulp_horn_height': params.pulp_horn_height,
        'root_canal_taper': params.root_canal_taper,
        # DEJ
        'dej_scallop_size': params.dej_scallop_size,
        'dej_scallop_count': params.dej_scallop_count,
        'dej_width': params.dej_width,
        'enable_dej_texture': params.enable_dej_texture,
        # Dentin
        'dentin_thickness': params.dentin_thickness,
        'peritubular_dentin_ratio': params.peritubular_dentin_ratio,
        'intertubular_dentin_ratio': params.intertubular_dentin_ratio,
        'mantle_dentin_thickness': params.mantle_dentin_thickness,
        # Enamel
        'enamel_interface_roughness': params.enamel_interface_roughness,
        'enamel_thickness_occlusal': params.enamel_thickness_occlusal,
        'enamel_thickness_cervical': params.enamel_thickness_cervical,
        'enable_enamel_shell': params.enable_enamel_shell,
        # Root
        'root_count': params.root_count,
        'root_furcation_height': params.root_furcation_height,
        'cementum_thickness': params.cementum_thickness,
        'enable_cementum': params.enable_cementum,
        # Generation
        'resolution': params.resolution,
        'effective_resolution': res,
        'seed': params.seed,
        'randomness': params.randomness,
        'detail_level': params.detail_level,
    }

    return result, stats


def generate_dentin_pulp_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate dentin-pulp scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching DentinPulpParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_dentin_pulp(DentinPulpParams(
        # Basic Geometry
        tooth_height=params.get('tooth_height', 17.0),
        crown_diameter=params.get('crown_diameter', 7.0),
        crown_height=params.get('crown_height', 5.0),
        root_length=params.get('root_length', 12.0),
        root_diameter=params.get('root_diameter', 3.0),

        # Dentin Tubule Parameters
        tubule_diameter_dej=params.get('tubule_diameter_dej', 1.0),
        tubule_diameter_pulp=params.get('tubule_diameter_pulp', 3.7),
        tubule_density_dej=params.get('tubule_density_dej', 1.9e6),
        tubule_density_pulp=params.get('tubule_density_pulp', 6.5e6),
        enable_tubule_representation=params.get('enable_tubule_representation', False),
        tubule_resolution=params.get('tubule_resolution', 4),

        # Pulp Chamber
        pulp_chamber_height=params.get('pulp_chamber_height', 2.76),
        pulp_chamber_width=params.get('pulp_chamber_width', 2.5),
        pulp_chamber_size=params.get('pulp_chamber_size', 0.4),
        pulp_horn_count=params.get('pulp_horn_count', 2),
        pulp_horn_height=params.get('pulp_horn_height', 0.5),
        root_canal_taper=params.get('root_canal_taper', 0.06),

        # DEJ Parameters
        dej_scallop_size=params.get('dej_scallop_size', 75.0),
        dej_scallop_count=params.get('dej_scallop_count', 12),
        dej_width=params.get('dej_width', 9.0),
        enable_dej_texture=params.get('enable_dej_texture', True),

        # Dentin Layers
        dentin_thickness=params.get('dentin_thickness', 2.0),
        peritubular_dentin_ratio=params.get('peritubular_dentin_ratio', 0.15),
        intertubular_dentin_ratio=params.get('intertubular_dentin_ratio', 0.85),
        mantle_dentin_thickness=params.get('mantle_dentin_thickness', 0.025),

        # Enamel Interface
        enamel_interface_roughness=params.get('enamel_interface_roughness', 0.5),
        enamel_thickness_occlusal=params.get('enamel_thickness_occlusal', 2.5),
        enamel_thickness_cervical=params.get('enamel_thickness_cervical', 0.5),
        enable_enamel_shell=params.get('enable_enamel_shell', False),

        # Root Features
        root_count=params.get('root_count', 1),
        root_furcation_height=params.get('root_furcation_height', 3.0),
        cementum_thickness=params.get('cementum_thickness', 0.1),
        enable_cementum=params.get('enable_cementum', False),

        # Generation Settings
        resolution=params.get('resolution', 16),
        seed=params.get('seed', 42),
        randomness=params.get('randomness', 0.1),
        detail_level=params.get('detail_level', 'medium'),
    ))
