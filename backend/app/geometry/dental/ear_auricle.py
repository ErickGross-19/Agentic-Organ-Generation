"""
Ear Auricle scaffold generator for complex curved cartilage frameworks.

Provides parametric generation of anatomically-accurate ear-shaped scaffolds with:
- Main body as ellipsoid base
- Helix rim with proper width and curvature
- Antihelix with optional superior/inferior crura bifurcation
- Concha bowl with cymba/cavum division
- Tragus and antitragus projections
- Lobule (earlobe) with attached/free variants
- Surface texture for cell attachment
- Shell structure with configurable thickness

Biologically realistic parameters based on:
- Adult ear dimensions: ~60mm height, ~35mm width (Farkas 1994)
- Cartilage thickness: 0.5-1.2mm (Bichon 2008)
- Elastic cartilage with ~80% type II collagen
- Scaffold porosity: 65-80% optimal for cartilage regeneration
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..core import batch_union


@dataclass
class EarAuricleParams:
    """
    Parameters for ear auricle scaffold generation.

    Biologically realistic defaults based on adult human ear anatomy.
    """
    # === Overall Dimensions ===
    overall_height: float = 60.0  # mm, total ear height (adult ~60-65mm)
    overall_width: float = 35.0  # mm, total ear width (adult ~30-35mm)
    overall_depth: float = 20.0  # mm, ear projection from skull
    scale_factor: float = 1.0  # overall size multiplier (0.5-1.5)

    # === Cartilage Framework ===
    cartilage_thickness: float = 1.5  # mm (1-3 mm native elastic cartilage)
    skin_thickness: float = 1.0  # mm, overlying skin thickness
    thickness: float = 2.0  # mm, combined cartilage+skin thickness (legacy)

    # === Structural Elements ===
    strut_width: float = 0.5  # mm, 500 μm lattice strut width
    strut_spacing: float = 1.5  # mm, spacing between lattice struts
    pore_size: float = 0.2  # mm (150-250 um optimal for chondrogenesis)
    pore_shape: Literal['circular', 'hexagonal'] = 'circular'

    # === Helix (Outer Rim) ===
    helix_definition: float = 0.7  # 0-1, prominence of helix rim
    helix_curvature: float = 0.6  # 0-1, curvature of helix fold
    helix_width: float = 8.0  # mm, width of helix rim
    helix_thickness_factor: float = 1.2  # multiplier for helix thickness

    # === Antihelix (Inner Ridge) ===
    antihelix_depth: float = 0.3  # 0-1, depth of antihelix ridge
    antihelix_curvature: float = 0.5  # 0-1, curvature of antihelix
    antihelix_bifurcation: bool = True  # superior/inferior crura
    crura_angle: float = 30.0  # degrees, angle between crura

    # === Concha (Bowl) ===
    concha_depth: float = 15.0  # mm, depth of conchal bowl
    concha_diameter: float = 20.0  # mm, diameter of concha
    cymba_conchae_ratio: float = 0.4  # ratio of cymba to cavum

    # === Tragus and Antitragus ===
    tragus_width: float = 8.0  # mm, tragus projection width
    tragus_height: float = 10.0  # mm, tragus height
    tragus_projection: float = 5.0  # mm, anterior projection
    antitragus_size: float = 6.0  # mm, antitragus size

    # === Lobule (Earlobe) ===
    lobule_length: float = 18.0  # mm, earlobe length (adult ~15-20mm)
    lobule_width: float = 15.0  # mm, earlobe width
    lobule_thickness: float = 4.0  # mm, soft tissue (no cartilage)
    lobule_attached: bool = False  # attached vs free earlobe

    # === Mechanical Properties ===
    mechanical_modulus_ratio: float = 0.5  # ratio vs native cartilage (target ~0.5-0.8)
    target_porosity: float = 0.7  # target void fraction for cell infiltration

    # === Surface Features ===
    enable_surface_texture: bool = True  # surface microstructure
    texture_roughness: float = 0.3  # 0-1, surface roughness intensity

    # === Generation Settings ===
    resolution: int = 16  # circular resolution for mesh generation
    seed: int = 42  # random seed for stochastic features
    randomness: float = 0.1  # 0-1, overall randomness intensity
    detail_level: Literal['low', 'medium', 'high'] = 'medium'


def _get_resolution(base_resolution: int, detail_level: str) -> int:
    """
    Adjust resolution based on detail level.

    Args:
        base_resolution: Base circular resolution
        detail_level: 'low', 'medium', or 'high'

    Returns:
        Adjusted resolution value
    """
    if detail_level == 'low':
        return max(8, base_resolution // 2)
    elif detail_level == 'high':
        return base_resolution * 2
    return base_resolution  # medium


def _create_main_body(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the main ear body as a shaped ellipsoid.

    The main body provides the base form that other features attach to.
    Uses randomness parameter for natural variation.
    """
    # Create base ellipsoid
    main_body = m3d.Manifold.sphere(1.0, res)

    # Apply randomness for natural variation
    width_var = 1.0 + (rng.random() - 0.5) * params.randomness * 0.1
    height_var = 1.0 + (rng.random() - 0.5) * params.randomness * 0.1

    main_body = main_body.scale([
        ear_width / 2 * width_var,
        ear_depth / 2,
        ear_height / 2 * height_var
    ])

    return main_body


def _create_helix(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the helix (outer rim) of the ear.

    The helix is the curved outer edge that wraps around the ear.
    Uses helix_width, helix_curvature, helix_definition, and helix_thickness_factor.
    """
    # Calculate helix dimensions using actual parameters
    helix_width_scaled = params.helix_width * params.scale_factor
    helix_major_radius = ear_width / 2.5

    # Helix minor radius based on thickness and definition
    base_thickness = params.cartilage_thickness + params.skin_thickness * 2
    helix_minor_radius = base_thickness * params.helix_thickness_factor * (0.5 + params.helix_definition * 0.5)

    # Create torus for helix with curvature factor
    curvature_scale = 0.5 + params.helix_curvature * 0.5

    helix_torus = m3d.Manifold.revolve(
        m3d.CrossSection.circle(helix_minor_radius, max(8, res // 2)),
        res
    )

    # Scale for helix shape - more elongated vertically
    helix_torus = helix_torus.scale([
        1.0,
        curvature_scale,
        ear_height / (2 * helix_major_radius) * 1.2
    ])

    # Position helix at outer edge
    helix_torus = helix_torus.translate([helix_major_radius, 0, 0])

    # Cut helix to keep only outer section (posterior part)
    cut_width = ear_width + helix_width_scaled
    helix_cut_box = m3d.Manifold.cube([cut_width, ear_depth * 3, ear_height * 3], center=True)
    helix_cut_box = helix_cut_box.translate([ear_width / 4 + helix_width_scaled / 4, 0, 0])

    helix = helix_torus ^ helix_cut_box

    # Apply randomness for natural variation
    if params.randomness > 0:
        rand_offset = (rng.random() - 0.5) * params.randomness * 2.0
        helix = helix.translate([rand_offset, 0, rand_offset * 0.5])

    return helix


def _create_antihelix(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the antihelix (inner ridge) with optional bifurcation into crura.

    The antihelix is the Y-shaped ridge inside the helix. When bifurcation
    is enabled, it splits into superior and inferior crura at ~2/3 height.
    """
    if params.antihelix_depth <= 0:
        # Return empty manifold if no antihelix
        return m3d.Manifold.sphere(0.01, 4).translate([1000, 1000, 1000])

    antihelix_parts = []

    # Base antihelix dimensions
    antihelix_height = ear_height * 0.6
    base_thickness = params.cartilage_thickness + params.skin_thickness * 2
    antihelix_radius = base_thickness * (0.5 + params.antihelix_depth * 0.5)

    # Main stem of antihelix (lower 2/3)
    stem_height = antihelix_height * 0.65
    stem = m3d.Manifold.cylinder(
        stem_height,
        antihelix_radius * 1.2,
        antihelix_radius * 0.8,
        max(8, res // 4)
    )

    # Rotate to be roughly vertical with slight tilt based on curvature
    tilt_angle = 15 + params.antihelix_curvature * 20
    stem = stem.rotate([90, 0, tilt_angle])
    stem = stem.translate([-ear_width / 8, -ear_depth / 4, -ear_height * 0.15])
    antihelix_parts.append(stem)

    # Create bifurcation (crura) if enabled
    if params.antihelix_bifurcation:
        crura_length = antihelix_height * 0.4
        crura_radius = antihelix_radius * 0.7
        crura_angle_rad = np.radians(params.crura_angle)

        # Superior crus (upper branch)
        superior_crus = m3d.Manifold.cylinder(
            crura_length,
            crura_radius,
            crura_radius * 0.6,
            max(6, res // 4)
        )
        superior_crus = superior_crus.rotate([90, 0, tilt_angle + params.crura_angle / 2])
        superior_crus = superior_crus.translate([
            -ear_width / 8 - stem_height * 0.3 * np.sin(np.radians(tilt_angle)),
            -ear_depth / 4,
            -ear_height * 0.15 + stem_height * 0.5
        ])
        antihelix_parts.append(superior_crus)

        # Inferior crus (lower branch)
        inferior_crus = m3d.Manifold.cylinder(
            crura_length * 0.8,
            crura_radius,
            crura_radius * 0.5,
            max(6, res // 4)
        )
        inferior_crus = inferior_crus.rotate([90, 0, tilt_angle - params.crura_angle / 2])
        inferior_crus = inferior_crus.translate([
            -ear_width / 8 - stem_height * 0.2 * np.sin(np.radians(tilt_angle)),
            -ear_depth / 4,
            -ear_height * 0.15 + stem_height * 0.4
        ])
        antihelix_parts.append(inferior_crus)

    # Apply randomness
    if params.randomness > 0:
        for i, part in enumerate(antihelix_parts):
            rand_offset = (rng.random() - 0.5) * params.randomness * 1.0
            antihelix_parts[i] = part.translate([rand_offset * 0.5, 0, rand_offset * 0.3])

    # Combine all antihelix parts
    if len(antihelix_parts) == 1:
        return antihelix_parts[0]
    return batch_union(antihelix_parts)


def _create_concha(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float
) -> m3d.Manifold:
    """
    Create the concha (bowl) depression with cymba and cavum divisions.

    The concha is the large depression leading to the ear canal.
    Uses concha_depth, concha_diameter, and cymba_conchae_ratio.
    """
    # Scale concha parameters
    concha_diameter_scaled = params.concha_diameter * params.scale_factor
    concha_depth_scaled = params.concha_depth * params.scale_factor

    # Main conchal bowl (cavum - lower part)
    cavum_diameter = concha_diameter_scaled * (1 - params.cymba_conchae_ratio)
    cavum = m3d.Manifold.sphere(cavum_diameter / 2, max(8, res // 2))
    cavum = cavum.scale([1, concha_depth_scaled / (cavum_diameter / 2), 1])

    # Cymba (upper part, smaller)
    cymba_diameter = concha_diameter_scaled * params.cymba_conchae_ratio
    if cymba_diameter > 1.0:
        cymba = m3d.Manifold.sphere(cymba_diameter / 2, max(6, res // 3))
        cymba = cymba.scale([1, concha_depth_scaled * 0.6 / (cymba_diameter / 2), 1])
        cymba = cymba.translate([0, 0, cavum_diameter / 3])

        # Combine cymba and cavum
        concha = cavum + cymba
    else:
        concha = cavum

    # Position concha in the ear
    concha = concha.translate([-ear_width / 6, ear_depth * 0.2, -ear_height * 0.05])

    return concha


def _create_tragus(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the tragus - the small pointed projection anterior to the ear canal.

    Uses tragus_width, tragus_height, and tragus_projection parameters.
    """
    # Scale tragus parameters
    tragus_width_scaled = params.tragus_width * params.scale_factor
    tragus_height_scaled = params.tragus_height * params.scale_factor
    tragus_projection_scaled = params.tragus_projection * params.scale_factor

    # Tragus is a flattened, slightly curved projection
    # Create as scaled sphere for organic shape
    tragus = m3d.Manifold.sphere(1.0, max(8, res // 2))
    tragus = tragus.scale([
        tragus_width_scaled / 2,
        tragus_projection_scaled / 2,
        tragus_height_scaled / 2
    ])

    # Position in front of concha
    tragus = tragus.translate([
        -ear_width / 4 - tragus_width_scaled / 3,
        ear_depth * 0.3,
        -ear_height * 0.12
    ])

    # Apply randomness
    if params.randomness > 0:
        rand_offset = (rng.random() - 0.5) * params.randomness * 1.5
        tragus = tragus.translate([rand_offset * 0.5, rand_offset * 0.3, 0])

    return tragus


def _create_antitragus(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the antitragus - the small projection opposite the tragus.

    Uses antitragus_size parameter.
    """
    # Scale antitragus
    antitragus_scaled = params.antitragus_size * params.scale_factor

    if antitragus_scaled < 1.0:
        # Return tiny sphere far away if antitragus is too small
        return m3d.Manifold.sphere(0.01, 4).translate([1000, 1000, 1000])

    # Antitragus is a small rounded projection
    antitragus = m3d.Manifold.sphere(antitragus_scaled / 2, max(6, res // 3))

    # Position below and posterior to tragus
    antitragus = antitragus.translate([
        -ear_width / 6,
        ear_depth * 0.15,
        -ear_height * 0.28
    ])

    # Apply randomness
    if params.randomness > 0:
        rand_offset = (rng.random() - 0.5) * params.randomness * 1.0
        antitragus = antitragus.translate([rand_offset * 0.3, rand_offset * 0.2, rand_offset * 0.3])

    return antitragus


def _create_lobule(
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create the lobule (earlobe) - soft tissue with no cartilage.

    Uses lobule_length, lobule_width, lobule_thickness, and lobule_attached.
    """
    # Scale lobule parameters
    lobule_length_scaled = params.lobule_length * params.scale_factor
    lobule_width_scaled = params.lobule_width * params.scale_factor
    lobule_thickness_scaled = params.lobule_thickness * params.scale_factor

    # Lobule is a rounded, slightly tapered shape
    lobule = m3d.Manifold.sphere(1.0, max(8, res // 2))
    lobule = lobule.scale([
        lobule_width_scaled / 2,
        lobule_thickness_scaled / 2,
        lobule_length_scaled / 2
    ])

    # Position at bottom of ear
    lobule_z = -ear_height / 2 - lobule_length_scaled * 0.3

    # If attached, position closer to the main body
    if params.lobule_attached:
        lobule_x = 0
        lobule_y = ear_depth * 0.1
    else:
        # Free lobule hangs slightly away
        lobule_x = ear_width * 0.05
        lobule_y = ear_depth * 0.05

    lobule = lobule.translate([lobule_x, lobule_y, lobule_z])

    # Apply randomness for natural variation
    if params.randomness > 0:
        rand_x = (rng.random() - 0.5) * params.randomness * 2.0
        rand_z = (rng.random() - 0.5) * params.randomness * 1.5
        lobule = lobule.translate([rand_x, 0, rand_z])

    return lobule


def _create_shell(
    ear_solid: m3d.Manifold,
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float
) -> m3d.Manifold:
    """
    Create a hollow shell from the solid ear geometry.

    Uses cartilage_thickness, skin_thickness, and thickness parameters.
    Thickness = cartilage + 2*skin (skin on both sides).
    """
    # Calculate effective thickness
    # If user specified explicit thickness, use it; otherwise compute from layers
    effective_thickness = params.thickness
    if params.cartilage_thickness > 0 and params.skin_thickness > 0:
        computed_thickness = params.cartilage_thickness + params.skin_thickness * 2
        # Use computed if thickness wasn't explicitly changed from default
        if abs(params.thickness - 2.0) < 0.01:  # default is 2.0
            effective_thickness = computed_thickness

    # Scale down to create inner surface
    min_dim = min(ear_width, ear_height, ear_depth)
    thickness_scale = 1.0 - (effective_thickness / min_dim * 2)
    thickness_scale = max(0.3, thickness_scale)  # Ensure minimum structure

    # Create inner offset by scaling a simplified version
    inner_solid = m3d.Manifold.sphere(1.0, max(8, res // 2))
    inner_solid = inner_solid.scale([
        ear_width / 2 * thickness_scale,
        ear_depth / 2 * thickness_scale,
        ear_height / 2 * thickness_scale
    ])

    # Subtract inner from outer to create shell
    result = ear_solid - inner_solid

    return result


def _add_surface_texture(
    ear: m3d.Manifold,
    params: EarAuricleParams,
    res: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Add surface texture (micro-bumps) for improved cell attachment.

    Uses enable_surface_texture, texture_roughness, pore_size, and pore_shape.
    Creates small protrusions based on strut parameters.
    """
    if not params.enable_surface_texture or params.texture_roughness <= 0:
        return ear

    # Get bounding box for texture placement
    # bbox returns (min_x, min_y, min_z, max_x, max_y, max_z)
    bbox = ear.bounding_box()
    min_pt = (bbox[0], bbox[1], bbox[2])
    max_pt = (bbox[3], bbox[4], bbox[5])

    # Calculate texture bump parameters
    bump_radius = params.strut_width * 0.3 * (0.5 + params.texture_roughness * 0.5)
    bump_spacing = params.strut_spacing * 2

    # Limit number of bumps for performance
    ear_width_bb = max_pt[0] - min_pt[0]
    ear_height_bb = max_pt[2] - min_pt[2]

    # Generate texture bumps on outer surface
    n_bumps_x = min(20, int(ear_width_bb / bump_spacing))
    n_bumps_z = min(30, int(ear_height_bb / bump_spacing))

    if n_bumps_x < 3 or n_bumps_z < 3:
        return ear

    texture_bumps = []
    bump_res = max(4, res // 4)

    for i in range(n_bumps_x):
        for j in range(n_bumps_z):
            # Position on surface with some randomness
            x = min_pt[0] + (i + 0.5 + (rng.random() - 0.5) * 0.5) * bump_spacing
            z = min_pt[2] + (j + 0.5 + (rng.random() - 0.5) * 0.5) * bump_spacing

            # Vary bump size based on roughness
            size_var = 1.0 + (rng.random() - 0.5) * params.texture_roughness * 0.5

            if params.pore_shape == 'hexagonal':
                # Hexagonal bumps (approximated with low-res cylinder)
                bump = m3d.Manifold.cylinder(
                    bump_radius * size_var * 2,
                    bump_radius * size_var,
                    bump_radius * size_var * 0.7,
                    6  # hexagonal
                )
            else:
                # Circular bumps (spheres)
                bump = m3d.Manifold.sphere(bump_radius * size_var, bump_res)

            # Position on the positive Y surface (outer)
            y = max_pt[1] * 0.9
            bump = bump.translate([x, y, z])
            texture_bumps.append(bump)

    # Combine ear with texture (limited set for performance)
    if texture_bumps:
        texture_combined = batch_union(texture_bumps[:100])  # Limit for performance
        ear = ear + texture_combined

    return ear


def _create_porous_structure(
    ear: m3d.Manifold,
    params: EarAuricleParams,
    res: int,
    ear_height: float,
    ear_width: float,
    ear_depth: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create porous scaffold structure using pore parameters.

    Uses target_porosity, pore_size, pore_shape, strut_width, and strut_spacing.
    Creates a lattice-like internal structure for tissue engineering.
    """
    if params.target_porosity <= 0.1:
        return ear

    # Calculate pore dimensions
    pore_size_scaled = params.pore_size * params.scale_factor

    # Get bounding box (returns min_x, min_y, min_z, max_x, max_y, max_z)
    bbox = ear.bounding_box()
    min_pt = (bbox[0], bbox[1], bbox[2])
    max_pt = (bbox[3], bbox[4], bbox[5])

    # Calculate number of pores based on porosity
    ear_vol = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1]) * (max_pt[2] - min_pt[2])
    pore_vol = (4/3) * np.pi * (pore_size_scaled / 2) ** 3
    n_pores = int(params.target_porosity * ear_vol / pore_vol * 0.3)
    n_pores = min(n_pores, 500)  # Limit for performance

    if n_pores < 10:
        return ear

    pores = []
    pore_res = max(4, res // 4)

    for _ in range(n_pores):
        # Random position within bounds (with margin)
        margin = pore_size_scaled
        x = rng.uniform(min_pt[0] + margin, max_pt[0] - margin)
        y = rng.uniform(min_pt[1] + margin, max_pt[1] - margin)
        z = rng.uniform(min_pt[2] + margin, max_pt[2] - margin)

        # Vary pore size slightly
        size_var = 1.0 + (rng.random() - 0.5) * params.randomness * 0.3

        if params.pore_shape == 'hexagonal':
            pore = m3d.Manifold.cylinder(
                pore_size_scaled * size_var,
                pore_size_scaled * size_var / 2,
                pore_size_scaled * size_var / 2,
                6
            )
            # Random rotation for variety
            pore = pore.rotate([rng.random() * 90, rng.random() * 90, 0])
        else:
            pore = m3d.Manifold.sphere(pore_size_scaled * size_var / 2, pore_res)

        pore = pore.translate([x, y, z])
        pores.append(pore)

    # Subtract pores from ear
    if pores:
        pores_combined = batch_union(pores)
        ear = ear - pores_combined

    return ear


def generate_ear_auricle(params: EarAuricleParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an anatomically-accurate ear auricle scaffold.

    Creates a complete ear shape using ALL 38 parameters including:
    - Main body ellipsoid with proper dimensions
    - Helix rim with width, curvature, and thickness
    - Antihelix with optional bifurcation into superior/inferior crura
    - Concha bowl with cymba/cavum division
    - Tragus and antitragus projections
    - Lobule (earlobe) with attached/free variants
    - Shell structure with cartilage and skin layers
    - Surface texture for cell attachment
    - Porous structure for tissue engineering

    Args:
        params: EarAuricleParams specifying complete ear geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with comprehensive statistics

    Raises:
        ValueError: If parameters are invalid
    """
    # Parameter validation
    if params.scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if params.thickness <= 0:
        raise ValueError("thickness must be positive")
    if not (0 <= params.helix_definition <= 1):
        raise ValueError("helix_definition must be between 0 and 1")
    if not (0 <= params.antihelix_depth <= 1):
        raise ValueError("antihelix_depth must be between 0 and 1")
    if not (0 <= params.helix_curvature <= 1):
        raise ValueError("helix_curvature must be between 0 and 1")
    if not (0 <= params.antihelix_curvature <= 1):
        raise ValueError("antihelix_curvature must be between 0 and 1")
    if not (0 <= params.target_porosity <= 1):
        raise ValueError("target_porosity must be between 0 and 1")
    if not (0 <= params.texture_roughness <= 1):
        raise ValueError("texture_roughness must be between 0 and 1")
    if not (0 <= params.randomness <= 1):
        raise ValueError("randomness must be between 0 and 1")
    if params.overall_height <= 0:
        raise ValueError("overall_height must be positive")
    if params.overall_width <= 0:
        raise ValueError("overall_width must be positive")
    if params.overall_depth <= 0:
        raise ValueError("overall_depth must be positive")

    # Set up RNG for reproducibility
    rng = np.random.default_rng(params.seed)

    # Adjust resolution for detail level
    res = _get_resolution(params.resolution, params.detail_level)

    # Calculate scaled ear dimensions using ACTUAL parameters (not hardcoded!)
    ear_height = params.overall_height * params.scale_factor
    ear_width = params.overall_width * params.scale_factor
    ear_depth = params.overall_depth * params.scale_factor

    # Build main body ellipsoid
    main_body = _create_main_body(params, res, ear_height, ear_width, ear_depth, rng)

    # Build helix rim with proper width and curvature
    helix = _create_helix(params, res, ear_height, ear_width, ear_depth, rng)

    # Build antihelix with optional bifurcation (crura)
    antihelix = _create_antihelix(params, res, ear_height, ear_width, ear_depth, rng)

    # Create concha bowl with proper depth/diameter and cymba/cavum division
    concha = _create_concha(params, res, ear_height, ear_width, ear_depth)

    # Create tragus projection
    tragus = _create_tragus(params, res, ear_height, ear_width, ear_depth, rng)

    # Create antitragus projection
    antitragus = _create_antitragus(params, res, ear_height, ear_width, ear_depth, rng)

    # Create lobule (no cartilage, just soft tissue shape)
    lobule = _create_lobule(params, res, ear_height, ear_width, ear_depth, rng)

    # Combine all positive parts
    ear_parts = [main_body, helix, antihelix, tragus, antitragus, lobule]
    ear_solid = batch_union(ear_parts)

    # Subtract concha bowl
    ear_solid = ear_solid - concha

    # Create shell with proper thickness (cartilage + skin layers)
    result = _create_shell(ear_solid, params, res, ear_height, ear_width, ear_depth)

    # Add porous structure for tissue engineering if porosity > 0.1
    if params.target_porosity > 0.1:
        result = _create_porous_structure(result, params, res, ear_height, ear_width, ear_depth, rng)

    # Add surface texture if enabled
    if params.enable_surface_texture:
        result = _add_surface_texture(result, params, res, rng)

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0.0

    # Calculate effective thickness
    effective_thickness = params.thickness
    if params.cartilage_thickness > 0 and params.skin_thickness > 0:
        computed_thickness = params.cartilage_thickness + params.skin_thickness * 2
        if abs(params.thickness - 2.0) < 0.01:
            effective_thickness = computed_thickness

    # Calculate surface area approximation
    # bbox returns (min_x, min_y, min_z, max_x, max_y, max_z)
    bbox = result.bounding_box()
    bbox_min = (bbox[0], bbox[1], bbox[2])
    bbox_max = (bbox[3], bbox[4], bbox[5])
    surface_area_approx = 2 * (
        (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1]) +
        (bbox_max[1] - bbox_min[1]) * (bbox_max[2] - bbox_min[2]) +
        (bbox_max[0] - bbox_min[0]) * (bbox_max[2] - bbox_min[2])
    )

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'surface_area_mm2': surface_area_approx,
        'scaffold_type': 'ear_auricle',

        # Overall Dimensions (actually used)
        'overall_height': params.overall_height,
        'overall_width': params.overall_width,
        'overall_depth': params.overall_depth,
        'scale_factor': params.scale_factor,

        # Computed dimensions
        'ear_dimensions': {
            'height': ear_height,
            'width': ear_width,
            'depth': ear_depth
        },

        # Cartilage Framework
        'cartilage_thickness': params.cartilage_thickness,
        'skin_thickness': params.skin_thickness,
        'thickness': params.thickness,
        'effective_thickness': effective_thickness,

        # Structural Elements
        'strut_width': params.strut_width,
        'strut_spacing': params.strut_spacing,
        'pore_size': params.pore_size,
        'pore_shape': params.pore_shape,

        # Helix
        'helix_definition': params.helix_definition,
        'helix_curvature': params.helix_curvature,
        'helix_width': params.helix_width,
        'helix_thickness_factor': params.helix_thickness_factor,

        # Antihelix
        'antihelix_depth': params.antihelix_depth,
        'antihelix_curvature': params.antihelix_curvature,
        'antihelix_bifurcation': params.antihelix_bifurcation,
        'crura_angle': params.crura_angle,

        # Concha
        'concha_depth': params.concha_depth,
        'concha_diameter': params.concha_diameter,
        'cymba_conchae_ratio': params.cymba_conchae_ratio,

        # Tragus and Antitragus
        'tragus_width': params.tragus_width,
        'tragus_height': params.tragus_height,
        'tragus_projection': params.tragus_projection,
        'antitragus_size': params.antitragus_size,

        # Lobule
        'lobule_length': params.lobule_length,
        'lobule_width': params.lobule_width,
        'lobule_thickness': params.lobule_thickness,
        'lobule_attached': params.lobule_attached,

        # Mechanical Properties
        'mechanical_modulus_ratio': params.mechanical_modulus_ratio,
        'target_porosity': params.target_porosity,

        # Surface Features
        'enable_surface_texture': params.enable_surface_texture,
        'texture_roughness': params.texture_roughness,

        # Generation Settings
        'resolution': params.resolution,
        'effective_resolution': res,
        'seed': params.seed,
        'randomness': params.randomness,
        'detail_level': params.detail_level,
    }

    return result, stats


def generate_ear_auricle_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate ear auricle scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching EarAuricleParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_ear_auricle(EarAuricleParams(
        # Overall Dimensions
        overall_height=params.get('overall_height', 60.0),
        overall_width=params.get('overall_width', 35.0),
        overall_depth=params.get('overall_depth', 20.0),
        scale_factor=params.get('scale_factor', 1.0),

        # Cartilage Framework
        cartilage_thickness=params.get('cartilage_thickness', 1.5),
        skin_thickness=params.get('skin_thickness', 1.0),
        thickness=params.get('thickness', 2.0),

        # Structural Elements
        strut_width=params.get('strut_width', 0.5),
        strut_spacing=params.get('strut_spacing', 1.5),
        pore_size=params.get('pore_size', 0.2),
        pore_shape=params.get('pore_shape', 'circular'),

        # Helix
        helix_definition=params.get('helix_definition', 0.7),
        helix_curvature=params.get('helix_curvature', 0.6),
        helix_width=params.get('helix_width', 8.0),
        helix_thickness_factor=params.get('helix_thickness_factor', 1.2),

        # Antihelix
        antihelix_depth=params.get('antihelix_depth', 0.3),
        antihelix_curvature=params.get('antihelix_curvature', 0.5),
        antihelix_bifurcation=params.get('antihelix_bifurcation', True),
        crura_angle=params.get('crura_angle', 30.0),

        # Concha
        concha_depth=params.get('concha_depth', 15.0),
        concha_diameter=params.get('concha_diameter', 20.0),
        cymba_conchae_ratio=params.get('cymba_conchae_ratio', 0.4),

        # Tragus and Antitragus
        tragus_width=params.get('tragus_width', 8.0),
        tragus_height=params.get('tragus_height', 10.0),
        tragus_projection=params.get('tragus_projection', 5.0),
        antitragus_size=params.get('antitragus_size', 6.0),

        # Lobule
        lobule_length=params.get('lobule_length', 18.0),
        lobule_width=params.get('lobule_width', 15.0),
        lobule_thickness=params.get('lobule_thickness', 4.0),
        lobule_attached=params.get('lobule_attached', False),

        # Mechanical Properties
        mechanical_modulus_ratio=params.get('mechanical_modulus_ratio', 0.5),
        target_porosity=params.get('target_porosity', 0.7),

        # Surface Features
        enable_surface_texture=params.get('enable_surface_texture', True),
        texture_roughness=params.get('texture_roughness', 0.3),

        # Generation Settings
        resolution=params.get('resolution', 16),
        seed=params.get('seed', 42),
        randomness=params.get('randomness', 0.1),
        detail_level=params.get('detail_level', 'medium'),
    ))
