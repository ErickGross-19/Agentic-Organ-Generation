"""
Spinal cord scaffold generator.

Generates spinal cord structures with butterfly-shaped gray matter core
and surrounding white matter channels. Includes meningeal layers and
anatomically accurate regional features.
"""

import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpinalCordParams:
    """
    Parameters for spinal cord scaffold generation.

    Based on human spinal cord anatomy:
    - Gray matter: butterfly/H-shaped core with dorsal/ventral horns
    - White matter: surrounding columns with ascending/descending tracts
    - Central canal: CSF-filled channel
    - Meninges: pia, arachnoid, dura mater
    """

    # === Basic Geometry ===
    total_diameter_mm: float = 10.0  # Overall cord diameter (cervical ~13mm, thoracic ~10mm)
    length_mm: float = 100.0  # Segment length

    # === Gray Matter Configuration ===
    gray_matter_volume_ratio: float = 0.22  # Gray matter as fraction of total (literature: 16-25%, using mid-range)
    gray_matter_pattern: str = "butterfly"  # "butterfly", "h_shape", or "simplified"

    # === Gray Matter Horns ===
    dorsal_horn_width_mm: float = 2.0  # Sensory horn width
    dorsal_horn_height_mm: float = 2.5  # Dorsal horn vertical extent
    ventral_horn_width_mm: float = 2.5  # Motor horn width (larger)
    ventral_horn_height_mm: float = 2.0  # Ventral horn vertical extent
    lateral_horn_present: bool = False  # T1-L2 sympathetic (autonomic)
    lateral_horn_size_mm: float = 0.8  # Lateral horn dimension

    # === Central Canal ===
    central_canal_diameter_mm: float = 0.3  # Ependymal-lined canal (~0.1-0.5mm) - age-dependent, often obliterated in adults >40
    enable_central_canal: bool = True

    # === White Matter ===
    white_matter_thickness_mm: float = 2.5  # Surrounding white matter thickness
    enable_tract_columns: bool = False  # Distinct dorsal/lateral/ventral columns
    dorsal_column_width_mm: float = 3.0  # Posterior funiculus
    lateral_column_width_mm: float = 4.0  # Lateral funiculus
    ventral_column_width_mm: float = 3.5  # Anterior funiculus

    # === Anterior Median Fissure ===
    enable_anterior_fissure: bool = True  # V-shaped groove on ventral surface
    anterior_fissure_depth_mm: float = 3.0  # Depth into ventral white matter
    anterior_fissure_width_mm: float = 1.0  # Width at surface

    # === Guidance Channels (for regeneration) ===
    num_guidance_channels: int = 50  # Channels for axon guidance
    channel_diameter_um: float = 150.0  # Individual channel diameter
    channel_pattern: str = "radial"  # "radial", "grid", or "random"
    channel_region: str = "white_matter"  # "white_matter", "gray_matter", or "both"

    # === Meningeal Layers ===
    enable_pia_mater: bool = False  # Innermost, adherent to cord
    pia_mater_thickness_um: float = 50.0  # ~20-100um (12um typical)
    enable_arachnoid_mater: bool = False  # Middle, delicate layer
    arachnoid_thickness_um: float = 100.0  # ~38um typical
    enable_dura_mater: bool = False  # Outermost, tough layer
    dura_mater_thickness_mm: float = 0.5  # ~0.3-0.5mm (500um typical)
    subarachnoid_space_mm: float = 2.0  # CSF space

    # === Root Entry/Exit Zones ===
    enable_root_entry_zones: bool = False  # Dorsal root entry points
    dorsal_root_diameter_mm: float = 1.0
    ventral_root_diameter_mm: float = 0.8
    root_spacing_mm: float = 10.0  # Segmental spacing
    root_angle_deg: float = 45.0  # Angle from midline

    # === Vascular Features ===
    enable_anterior_spinal_artery: bool = False
    anterior_spinal_artery_diameter_mm: float = 0.8  # 0.8-1.2mm, midline ventral
    enable_posterior_spinal_arteries: bool = False
    posterior_spinal_artery_diameter_mm: float = 0.4  # 0.4-0.5mm, paired dorsolateral

    # === Porosity ===
    scaffold_porosity: float = 0.6  # Target porosity
    pore_size_um: float = 100.0  # For cell infiltration

    # === Randomization ===
    position_noise: float = 0.0  # Variation in structures
    random_seed: int = 42

    # === Resolution ===
    resolution: int = 20  # Cylinder segments

    def __post_init__(self):
        if self.gray_matter_volume_ratio < 0.1 or self.gray_matter_volume_ratio > 0.5:
            raise ValueError("gray_matter_volume_ratio must be between 0.1 and 0.5")
        if self.num_guidance_channels < 0:
            raise ValueError("num_guidance_channels must be non-negative")
        if self.gray_matter_pattern not in ["butterfly", "h_shape", "simplified"]:
            raise ValueError("gray_matter_pattern must be 'butterfly', 'h_shape', or 'simplified'")


def _create_ellipse_cross_section(width: float, height: float, resolution: int = 32) -> list:
    """
    Create 2D ellipse points for cross-section.

    Args:
        width: Ellipse width (x-axis diameter)
        height: Ellipse height (y-axis diameter)
        resolution: Number of points

    Returns:
        List of (x, y) tuples forming ellipse
    """
    points = []
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        x = (width / 2) * np.cos(angle)
        y = (height / 2) * np.sin(angle)
        points.append((x, y))
    return points


def _create_butterfly_gray_matter(params: SpinalCordParams) -> m3d.Manifold:
    """
    Create anatomically accurate butterfly/H-shaped gray matter core.

    The gray matter consists of:
    - Dorsal horns: narrower, sensory processing
    - Ventral horns: wider, motor neurons
    - Central gray commissure connecting sides
    - Optional lateral horns (T1-L2)

    Uses white_matter_thickness_mm to define gray-white boundary.
    """
    length = params.length_mm
    res = max(params.resolution, 16)

    # Scale factor based on gray matter ratio and white matter thickness
    # Total cord radius minus white matter gives gray matter radius
    cord_radius = params.total_diameter_mm / 2
    gray_radius = cord_radius - params.white_matter_thickness_mm
    scale = gray_radius / cord_radius if cord_radius > 0 else np.sqrt(params.gray_matter_volume_ratio)

    # Dorsal horn dimensions (posterior, sensory)
    dh_w = params.dorsal_horn_width_mm * scale
    dh_h = params.dorsal_horn_height_mm * scale

    # Ventral horn dimensions (anterior, motor)
    vh_w = params.ventral_horn_width_mm * scale
    vh_h = params.ventral_horn_height_mm * scale

    # Central commissure dimensions
    comm_w = max(dh_w, vh_w) * 0.3
    comm_h = (dh_h + vh_h) * 0.15

    components = []

    # Create dorsal horns (top, sensory) - narrower
    # Positioned in the dorsal (positive Y) direction
    for side in [-1, 1]:  # Left and right
        # Main dorsal horn body as ellipsoid-like shape
        horn = m3d.Manifold.cylinder(
            length, dh_w / 2, dh_w / 2, res
        )
        # Scale to create elliptical cross-section
        horn = horn.scale([1.0, dh_h / dh_w, 1.0])
        # Position: lateral (X) and dorsal (Y)
        horn = horn.translate([side * dh_w * 0.8, dh_h * 0.4, 0])
        components.append(horn)

        # Add tapered tip extending dorsally
        tip = m3d.Manifold.cylinder(
            length, dh_w * 0.3, dh_w * 0.15, res
        )
        tip = tip.scale([1.0, dh_h / dh_w * 0.8, 1.0])
        tip = tip.translate([side * dh_w * 0.6, dh_h * 0.9, 0])
        components.append(tip)

    # Create ventral horns (bottom, motor) - wider
    # Positioned in the ventral (negative Y) direction
    for side in [-1, 1]:
        # Main ventral horn body
        horn = m3d.Manifold.cylinder(
            length, vh_w / 2, vh_w / 2, res
        )
        horn = horn.scale([1.0, vh_h / vh_w, 1.0])
        # Position: lateral and ventral
        horn = horn.translate([side * vh_w * 0.7, -vh_h * 0.35, 0])
        components.append(horn)

        # Ventral horn expansion (motor neuron pools)
        bulge = m3d.Manifold.cylinder(
            length, vh_w * 0.4, vh_w * 0.4, res
        )
        bulge = bulge.scale([1.0, vh_h / vh_w * 0.6, 1.0])
        bulge = bulge.translate([side * vh_w * 0.9, -vh_h * 0.2, 0])
        components.append(bulge)

    # Create central gray commissure (connecting bar)
    commissure = m3d.Manifold.cube([comm_w * 4, comm_h * 2, length], center=True)
    commissure = commissure.translate([0, 0, length / 2])
    components.append(commissure)

    # Add lateral horns if enabled (sympathetic, T1-L2)
    if params.lateral_horn_present:
        lh_size = params.lateral_horn_size_mm * scale
        for side in [-1, 1]:
            lat_horn = m3d.Manifold.cylinder(
                length, lh_size / 2, lh_size / 2, res
            )
            lat_horn = lat_horn.scale([1.0, 0.8, 1.0])
            # Position between dorsal and ventral horns
            lat_horn = lat_horn.translate([side * (dh_w + lh_size), 0, 0])
            components.append(lat_horn)

    # Union all gray matter components
    from ..core import batch_union
    gray_matter = batch_union(components)

    return gray_matter


def _create_h_shape_gray_matter(params: SpinalCordParams) -> m3d.Manifold:
    """
    Create H-shaped gray matter core (simpler, more symmetric).

    The H-shape consists of:
    - Dorsal horns: upper horizontal bars
    - Ventral horns: lower horizontal bars
    - Central vertical bar: connecting commissure
    """
    length = params.length_mm
    res = max(params.resolution, 16)

    # Scale factor based on gray matter ratio and white matter thickness
    cord_radius = params.total_diameter_mm / 2
    gray_radius = cord_radius - params.white_matter_thickness_mm
    scale = gray_radius / cord_radius if cord_radius > 0 else np.sqrt(params.gray_matter_volume_ratio)

    # Scaled horn dimensions
    dh_w = params.dorsal_horn_width_mm * scale
    dh_h = params.dorsal_horn_height_mm * scale
    vh_w = params.ventral_horn_width_mm * scale
    vh_h = params.ventral_horn_height_mm * scale

    components = []

    # Central vertical bar (commissure)
    bar_width = max(dh_w, vh_w) * 0.5
    bar_height = (dh_h + vh_h) * 0.8
    central_bar = m3d.Manifold.cube([bar_width, bar_height, length], center=True)
    central_bar = central_bar.translate([0, 0, length / 2])
    components.append(central_bar)

    # Dorsal horns (upper horizontal bars)
    for side in [-1, 1]:
        dorsal_horn = m3d.Manifold.cube([dh_w, dh_h * 0.6, length], center=True)
        dorsal_horn = dorsal_horn.translate([side * (dh_w * 0.6 + bar_width / 2), dh_h * 0.3, length / 2])
        components.append(dorsal_horn)

    # Ventral horns (lower horizontal bars)
    for side in [-1, 1]:
        ventral_horn = m3d.Manifold.cube([vh_w, vh_h * 0.6, length], center=True)
        ventral_horn = ventral_horn.translate([side * (vh_w * 0.6 + bar_width / 2), -vh_h * 0.3, length / 2])
        components.append(ventral_horn)

    # Add lateral horns if enabled
    if params.lateral_horn_present:
        lh_size = params.lateral_horn_size_mm * scale
        for side in [-1, 1]:
            lat_horn = m3d.Manifold.cube([lh_size, lh_size, length], center=True)
            lat_horn = lat_horn.translate([side * (dh_w + lh_size), 0, length / 2])
            components.append(lat_horn)

    # Union all components
    from ..core import batch_union
    h_shape = batch_union(components)

    return h_shape


def _create_simplified_gray_matter(params: SpinalCordParams) -> m3d.Manifold:
    """
    Create simplified elliptical gray matter core.

    Simple geometry for basic applications - just an elliptical cylinder.
    Size determined by gray_matter_volume_ratio.
    """
    length = params.length_mm
    res = max(params.resolution, 16)

    # Calculate gray matter radius from volume ratio
    cord_radius = params.total_diameter_mm / 2
    gray_radius = cord_radius - params.white_matter_thickness_mm

    # Create elliptical core (slightly taller than wide for anatomical accuracy)
    gray_matter = m3d.Manifold.cylinder(length, gray_radius, gray_radius, res)
    # Make it slightly elliptical (taller in Y direction)
    gray_matter = gray_matter.scale([1.0, 1.2, 1.0])

    return gray_matter


def _create_central_canal(params: SpinalCordParams) -> m3d.Manifold:
    """
    Create central canal through gray matter center.

    The central canal is lined with ependymal cells and contains CSF.
    Diameter: 200-300um typically.
    """
    canal_radius = params.central_canal_diameter_mm / 2
    # Make slightly longer to ensure clean subtraction
    canal = m3d.Manifold.cylinder(
        params.length_mm + 0.4,
        canal_radius,
        canal_radius,
        params.resolution
    )
    canal = canal.translate([0, 0, -0.2])
    return canal


def _create_anterior_median_fissure(params: SpinalCordParams) -> m3d.Manifold:
    """
    Create V-shaped anterior median fissure on ventral surface.

    This is a deep groove (3mm) on the ventral midline that partially
    divides the spinal cord. Contains the anterior spinal artery.
    """
    depth = params.anterior_fissure_depth_mm
    width = params.anterior_fissure_width_mm
    length = params.length_mm

    # Create V-shaped wedge
    # Build as triangular prism: wide at surface (ventral), tapers inward
    half_width = width / 2

    # Create cross-section points for the V-shape
    # The fissure opens ventrally (negative Y in our coordinate system)
    # We'll create a triangular prism

    # Use a thin box rotated and positioned to create V-shape
    # Alternative: create wedge using hull of cylinders

    # Simple approach: stack of progressively smaller boxes
    fissure_parts = []
    num_slices = 8
    for i in range(num_slices):
        frac = i / num_slices
        slice_width = width * (1 - frac * 0.8)  # Taper to 20% at tip
        slice_depth = depth / num_slices
        y_pos = -(params.total_diameter_mm / 2 - depth) - (i * slice_depth)

        slice_box = m3d.Manifold.cube([slice_width, slice_depth * 1.2, length + 0.4], center=True)
        slice_box = slice_box.translate([0, y_pos + slice_depth / 2, length / 2])
        fissure_parts.append(slice_box)

    from ..core import batch_union
    fissure = batch_union(fissure_parts)

    return fissure


def _create_white_matter_columns(params: SpinalCordParams, gray_matter: m3d.Manifold) -> m3d.Manifold:
    """
    Create white matter with distinct dorsal, lateral, and ventral columns.

    White matter columns (funiculi):
    - Dorsal columns: sensory (fasciculus gracilis/cuneatus)
    - Lateral columns: mixed (corticospinal, spinothalamic)
    - Ventral columns: motor tracts
    """
    cord_radius = params.total_diameter_mm / 2
    length = params.length_mm
    res = params.resolution

    # Create outer cord boundary
    outer_cord = m3d.Manifold.cylinder(length, cord_radius, cord_radius, res)

    if not params.enable_tract_columns:
        # Simple white matter: outer cord minus gray matter
        white_matter = outer_cord - gray_matter
        return white_matter

    # Create distinct column regions
    components = []

    # Angular extents for each column (in radians from dorsal midline)
    # Dorsal column: 0 to ~60 degrees on each side (top)
    # Lateral column: ~60 to ~150 degrees (sides)
    # Ventral column: ~150 to 180 degrees (bottom)

    dorsal_angle = np.radians(60)
    lateral_angle = np.radians(150)

    # Create column sectors as angular slices
    def create_column_sector(start_angle: float, end_angle: float,
                             inner_r: float, outer_r: float) -> m3d.Manifold:
        """Create a cylindrical sector for a white matter column."""
        # Create full cylinder and subtract wedges to create sector
        sector = m3d.Manifold.cylinder(length, outer_r, outer_r, res)

        # Subtract inner region
        if inner_r > 0:
            inner = m3d.Manifold.cylinder(length + 0.2, inner_r, inner_r, res)
            inner = inner.translate([0, 0, -0.1])
            sector = sector - inner

        # Create wedges to subtract the unwanted angular regions
        wedge_size = outer_r * 3

        # Subtract region before start_angle
        if start_angle > 0:
            wedge1 = m3d.Manifold.cube([wedge_size, wedge_size, length + 0.4], center=True)
            wedge1 = wedge1.translate([0, wedge_size / 2, length / 2])
            # Rotate to align with start angle
            wedge1 = wedge1.rotate([0, 0, np.degrees(start_angle / 2 - np.pi / 4)])
            sector = sector - wedge1

        return sector

    # Gray matter approximate radius
    gray_radius = cord_radius * np.sqrt(params.gray_matter_volume_ratio) * 1.5

    # Dorsal column (posterior funiculus) - top sector
    # Contains fasciculus gracilis (medial) and cuneatus (lateral)
    dorsal_outer = m3d.Manifold.cylinder(length, cord_radius, cord_radius, res)
    dorsal_outer = dorsal_outer.scale([
        params.dorsal_column_width_mm / (cord_radius * 2),
        1.0,
        1.0
    ])
    dorsal_outer = dorsal_outer.translate([0, cord_radius * 0.3, 0])
    components.append(dorsal_outer)

    # Lateral columns (bilateral)
    for side in [-1, 1]:
        lat_col = m3d.Manifold.cylinder(length,
                                        params.lateral_column_width_mm / 2,
                                        params.lateral_column_width_mm / 2,
                                        res)
        lat_col = lat_col.translate([side * cord_radius * 0.5, 0, 0])
        components.append(lat_col)

    # Ventral column (anterior funiculus) - bottom sector
    ventral_col = m3d.Manifold.cylinder(length, cord_radius, cord_radius, res)
    ventral_col = ventral_col.scale([
        params.ventral_column_width_mm / (cord_radius * 2),
        1.0,
        1.0
    ])
    ventral_col = ventral_col.translate([0, -cord_radius * 0.3, 0])
    components.append(ventral_col)

    # Union columns and intersect with outer boundary minus gray matter
    from ..core import batch_union
    all_columns = batch_union(components)

    # Intersect with valid white matter region
    white_matter_region = outer_cord - gray_matter
    white_matter = all_columns - gray_matter

    # Ensure we stay within cord boundary
    white_matter = outer_cord - gray_matter

    return white_matter


def _create_meningeal_layers(params: SpinalCordParams, cord_radius: float) -> list[m3d.Manifold]:
    """
    Create three concentric meningeal shells around the spinal cord.

    Meningeal layers (from inside to outside):
    - Pia mater: 12um, adherent to cord surface
    - Arachnoid mater: 38um, trabeculated space
    - Dura mater: 500um (0.5mm), tough outer layer

    Returns list of [pia, arachnoid, dura] manifolds.
    """
    length = params.length_mm
    res = params.resolution

    meninges = []
    current_radius = cord_radius

    # Pia mater (innermost)
    if params.enable_pia_mater:
        pia_thickness = params.pia_mater_thickness_um / 1000.0  # Convert to mm
        pia_outer_r = current_radius + pia_thickness

        pia_outer = m3d.Manifold.cylinder(length, pia_outer_r, pia_outer_r, res)
        pia_inner = m3d.Manifold.cylinder(length + 0.2, current_radius, current_radius, res)
        pia_inner = pia_inner.translate([0, 0, -0.1])
        pia = pia_outer - pia_inner
        meninges.append(pia)
        current_radius = pia_outer_r

    # Subarachnoid space (CSF-filled, we leave as gap)
    current_radius += params.subarachnoid_space_mm

    # Arachnoid mater (middle)
    if params.enable_arachnoid_mater:
        arach_thickness = params.arachnoid_thickness_um / 1000.0
        arach_outer_r = current_radius + arach_thickness

        arach_outer = m3d.Manifold.cylinder(length, arach_outer_r, arach_outer_r, res)
        arach_inner = m3d.Manifold.cylinder(length + 0.2, current_radius, current_radius, res)
        arach_inner = arach_inner.translate([0, 0, -0.1])
        arachnoid = arach_outer - arach_inner
        meninges.append(arachnoid)
        current_radius = arach_outer_r

    # Dura mater (outermost)
    if params.enable_dura_mater:
        dura_thickness = params.dura_mater_thickness_mm
        dura_outer_r = current_radius + dura_thickness

        dura_outer = m3d.Manifold.cylinder(length, dura_outer_r, dura_outer_r, res)
        dura_inner = m3d.Manifold.cylinder(length + 0.2, current_radius, current_radius, res)
        dura_inner = dura_inner.translate([0, 0, -0.1])
        dura = dura_outer - dura_inner
        meninges.append(dura)

    return meninges


def _create_root_entry_zones(params: SpinalCordParams, cord_radius: float,
                              outer_radius: float) -> tuple[list[m3d.Manifold], list[m3d.Manifold]]:
    """
    Create dorsal and ventral root channels through meninges.

    Nerve roots:
    - Dorsal roots: sensory afferents, 1mm diameter
    - Ventral roots: motor efferents, 0.8mm diameter
    - Angle: 45 degrees from midline
    - 31 pairs total (8C, 12T, 5L, 5S, 1Co)

    Returns tuple of (dorsal_roots, ventral_roots) channel lists.
    """
    length = params.length_mm
    spacing = params.root_spacing_mm
    angle = np.radians(params.root_angle_deg)

    dorsal_roots = []
    ventral_roots = []

    # Calculate number of root pairs that fit in segment
    num_pairs = int(length / spacing)

    # Channel extends from cord surface through all meninges
    channel_length = outer_radius - cord_radius + 2.0  # Extra length for clean cuts

    for i in range(num_pairs):
        z_pos = spacing / 2 + i * spacing

        if z_pos > length - spacing / 2:
            break

        # Dorsal roots (posterior, ±45° from dorsal midline)
        for side in [-1, 1]:
            # Dorsal root position: dorsal aspect of cord
            dr_radius = params.dorsal_root_diameter_mm / 2

            # Create channel cylinder
            dorsal_channel = m3d.Manifold.cylinder(
                channel_length, dr_radius, dr_radius, 8
            )

            # Rotate to angle from vertical (Y-axis)
            # Dorsal is +Y, so we angle in X-Y plane
            dorsal_channel = dorsal_channel.rotate([90, 0, 0])  # Point along Y
            dorsal_channel = dorsal_channel.rotate([0, 0, side * params.root_angle_deg])

            # Position at cord surface, dorsal aspect
            x_offset = cord_radius * np.sin(angle) * side
            y_offset = cord_radius * np.cos(angle)
            dorsal_channel = dorsal_channel.translate([x_offset, y_offset, z_pos])

            dorsal_roots.append(dorsal_channel)

        # Ventral roots (anterior, ±45° from ventral midline)
        for side in [-1, 1]:
            vr_radius = params.ventral_root_diameter_mm / 2

            ventral_channel = m3d.Manifold.cylinder(
                channel_length, vr_radius, vr_radius, 8
            )

            # Ventral is -Y
            ventral_channel = ventral_channel.rotate([-90, 0, 0])  # Point along -Y
            ventral_channel = ventral_channel.rotate([0, 0, side * params.root_angle_deg])

            x_offset = cord_radius * np.sin(angle) * side
            y_offset = -cord_radius * np.cos(angle)
            ventral_channel = ventral_channel.translate([x_offset, y_offset, z_pos])

            ventral_roots.append(ventral_channel)

    return dorsal_roots, ventral_roots


def _create_pore_network(params: SpinalCordParams, cord_radius: float) -> list[m3d.Manifold]:
    """
    Create microscopic pore network for nutrient exchange.

    Based on research:
    - AVOID 3-5 µm (inhibits neural differentiation, promotes glial)
    - Use >5 µm, recommend 10-50 µm for neural differentiation
    - Microscopic wall pores allow nutrient/waste exchange
    - Distributed throughout scaffold cross-section
    """
    if params.pore_size_um <= 0:
        return []

    pore_radius_mm = params.pore_size_um / 2000.0  # Convert µm to mm
    length = params.length_mm
    pores = []

    # Calculate pore density based on target porosity
    # Higher porosity = more pores
    pore_density = int(params.scaffold_porosity * 100)

    # Cap for performance
    actual_count = min(pore_density, 80)

    # Set random seed for reproducibility
    np.random.seed(params.random_seed + 1000)  # Different seed from channels

    # Distribute pores randomly throughout cross-section
    for _ in range(actual_count):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(pore_radius_mm, cord_radius - pore_radius_mm)
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Random Z position along length
        z = np.random.uniform(0, length)

        # Create small spherical pore
        pore = m3d.Manifold.sphere(pore_radius_mm, params.resolution)
        pore = pore.translate([x, y, z])
        pores.append(pore)

    return pores


def _create_vascular_channels(params: SpinalCordParams, cord_radius: float) -> list[m3d.Manifold]:
    """
    Create longitudinal vascular channels along the spinal cord.

    Vascular anatomy:
    - Anterior spinal artery (ASA): midline ventral, 0.8-1.2mm diameter
    - Posterior spinal arteries (PSA): paired dorsolateral, 0.4-0.5mm diameter
    """
    length = params.length_mm
    channels = []

    # Anterior spinal artery (single, midline ventral)
    if params.enable_anterior_spinal_artery:
        asa_radius = params.anterior_spinal_artery_diameter_mm / 2

        # Position in anterior median fissure (ventral midline)
        asa = m3d.Manifold.cylinder(
            length + 0.4, asa_radius, asa_radius, params.resolution
        )
        # Place at ventral surface, slightly embedded
        asa = asa.translate([0, -(cord_radius - asa_radius * 2), -0.2])
        channels.append(asa)

    # Posterior spinal arteries (paired, dorsolateral)
    if params.enable_posterior_spinal_arteries:
        psa_radius = params.posterior_spinal_artery_diameter_mm / 2

        for side in [-1, 1]:
            psa = m3d.Manifold.cylinder(
                length + 0.4, psa_radius, psa_radius, params.resolution
            )
            # Position dorsolateral (posterior-lateral)
            angle = np.radians(30)  # 30 degrees from dorsal midline
            x_pos = side * cord_radius * 0.7 * np.sin(angle)
            y_pos = cord_radius * 0.85  # Near dorsal surface
            psa = psa.translate([x_pos, y_pos, -0.2])
            channels.append(psa)

    return channels


def _create_guidance_channels(params: SpinalCordParams, cord_radius: float,
                               gray_width: float) -> list[m3d.Manifold]:
    """
    Create guidance channels for axon regeneration.

    Channels provide structural guidance for regenerating axons.
    Pattern options: radial, grid, random
    Region filtering: white_matter (default), gray_matter, or both

    Based on research:
    - White matter primary: longitudinal channels mimicking tracts
    - Guidance cues direct toward gray matter interfaces
    - Laminin coating recommended for axon guidance
    """
    if params.num_guidance_channels <= 0:
        return []

    channel_radius_mm = params.channel_diameter_um / 2000.0
    length = params.length_mm
    channels = []

    # Cap channel count for performance
    actual_count = min(params.num_guidance_channels, 48)

    # Set random seed for reproducibility
    np.random.seed(params.random_seed)

    # Define valid regions based on channel_region parameter
    gray_radius = gray_width / 2

    def is_valid_position(x: float, y: float) -> bool:
        """Check if position is valid based on channel_region setting."""
        dist = np.sqrt(x**2 + y**2)

        if params.channel_region == "white_matter":
            # Only in white matter (outside gray matter boundary)
            return gray_radius < dist < cord_radius
        elif params.channel_region == "gray_matter":
            # Only in gray matter (inside gray matter boundary)
            return dist < gray_radius
        else:  # "both"
            # Anywhere within cord
            return dist < cord_radius

    if params.channel_pattern == "radial":
        # Radial pattern - adjust radius based on region
        if params.channel_region == "gray_matter":
            channel_ring_radius = gray_radius * 0.7
        elif params.channel_region == "white_matter":
            channel_ring_radius = (cord_radius + gray_radius) / 2
        else:  # both
            channel_ring_radius = cord_radius * 0.6

        for i in range(actual_count):
            angle = 2 * np.pi * i / actual_count
            x = channel_ring_radius * np.cos(angle)
            y = channel_ring_radius * np.sin(angle)

            if params.position_noise > 0:
                x += np.random.uniform(-1, 1) * params.position_noise
                y += np.random.uniform(-1, 1) * params.position_noise

            if is_valid_position(x, y):
                channel = m3d.Manifold.cylinder(
                    length + 0.4, channel_radius_mm, channel_radius_mm, 8
                ).translate([x, y, -0.2])
                channels.append(channel)

    elif params.channel_pattern == "grid":
        # Grid pattern
        grid_size = int(np.sqrt(actual_count))
        spacing = (cord_radius * 2) / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size / 2 + 0.5) * spacing
                y = (j - grid_size / 2 + 0.5) * spacing

                if is_valid_position(x, y):
                    channel = m3d.Manifold.cylinder(
                        length + 0.4, channel_radius_mm, channel_radius_mm, 8
                    ).translate([x, y, -0.2])
                    channels.append(channel)

    else:  # random
        placed = 0
        attempts = 0
        max_attempts = actual_count * 10

        while placed < actual_count and attempts < max_attempts:
            angle = np.random.uniform(0, 2 * np.pi)

            # Generate radius based on region
            if params.channel_region == "gray_matter":
                r = np.random.uniform(channel_radius_mm, gray_radius - channel_radius_mm)
            elif params.channel_region == "white_matter":
                r = np.random.uniform(gray_radius + channel_radius_mm,
                                      cord_radius - channel_radius_mm)
            else:  # both
                r = np.random.uniform(channel_radius_mm, cord_radius - channel_radius_mm)

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            if is_valid_position(x, y):
                channel = m3d.Manifold.cylinder(
                    length + 0.4, channel_radius_mm, channel_radius_mm, 8
                ).translate([x, y, -0.2])
                channels.append(channel)
                placed += 1

            attempts += 1

    return channels


def generate_spinal_cord(params: SpinalCordParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a spinal cord scaffold with gray matter core and white matter channels.

    Creates a butterfly/H-shaped gray matter core in the center, surrounded by
    white matter region with parallel guidance channels for axon tracts.
    Optionally includes meningeal layers and vascular features.

    Args:
        params: SpinalCordParams with generation parameters

    Returns:
        Tuple of (manifold, stats_dict)

    Raises:
        ValueError: If parameters are invalid
    """
    from ..core import batch_union

    cord_radius = params.total_diameter_mm / 2
    gray_width = cord_radius * np.sqrt(params.gray_matter_volume_ratio)

    # === 1. Create gray matter core ===
    # Switch based on gray_matter_pattern parameter
    if params.gray_matter_pattern == "h_shape":
        gray_matter = _create_h_shape_gray_matter(params)
    elif params.gray_matter_pattern == "simplified":
        gray_matter = _create_simplified_gray_matter(params)
    else:  # default to "butterfly"
        gray_matter = _create_butterfly_gray_matter(params)

    # === 2. Subtract central canal from gray matter ===
    if params.enable_central_canal and params.central_canal_diameter_mm > 0:
        canal = _create_central_canal(params)
        gray_matter = gray_matter - canal

    # === 3. Create white matter (outer cord minus gray matter) ===
    outer_cord = m3d.Manifold.cylinder(
        params.length_mm, cord_radius, cord_radius, params.resolution
    )

    white_matter = _create_white_matter_columns(params, gray_matter)

    # === 4. Add anterior median fissure ===
    if params.enable_anterior_fissure and params.anterior_fissure_depth_mm > 0:
        fissure = _create_anterior_median_fissure(params)
        white_matter = white_matter - fissure

    # === 5. Create guidance channels in white matter ===
    channels = _create_guidance_channels(params, cord_radius, gray_width)

    if channels:
        all_channels = batch_union(channels)
        white_matter = white_matter - all_channels

    # === 6. Add vascular channels ===
    vascular = _create_vascular_channels(params, cord_radius)
    if vascular:
        all_vascular = batch_union(vascular)
        white_matter = white_matter - all_vascular

    # === 6.5. Add pore network for nutrient exchange ===
    pores = _create_pore_network(params, cord_radius)
    pore_count = 0
    if pores:
        all_pores = batch_union(pores)
        white_matter = white_matter - all_pores
        gray_matter = gray_matter - all_pores
        pore_count = len(pores)

    # === 7. Combine gray and white matter into cord ===
    spinal_cord = batch_union([gray_matter, white_matter])

    # === 8. Create meningeal layers ===
    meninges = _create_meningeal_layers(params, cord_radius)

    # Calculate outer radius after all meninges
    outer_radius = cord_radius
    if params.enable_pia_mater:
        outer_radius += params.pia_mater_thickness_um / 1000.0
    outer_radius += params.subarachnoid_space_mm
    if params.enable_arachnoid_mater:
        outer_radius += params.arachnoid_thickness_um / 1000.0
    if params.enable_dura_mater:
        outer_radius += params.dura_mater_thickness_mm

    # === 9. Create root entry/exit zones ===
    dorsal_roots = []
    ventral_roots = []
    if params.enable_root_entry_zones:
        dorsal_roots, ventral_roots = _create_root_entry_zones(
            params, cord_radius, outer_radius
        )

    # Subtract root channels from meninges
    if meninges and (dorsal_roots or ventral_roots):
        all_roots = batch_union(dorsal_roots + ventral_roots)
        meninges = [m - all_roots for m in meninges]

    # === 10. Combine all components ===
    all_components = [spinal_cord] + meninges
    final_scaffold = batch_union(all_components)

    # === Calculate statistics ===
    mesh = final_scaffold.to_mesh()
    volume = final_scaffold.volume() if hasattr(final_scaffold, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'total_diameter_mm': params.total_diameter_mm,
        'length_mm': params.length_mm,
        'gray_matter_volume_ratio': params.gray_matter_volume_ratio,
        'gray_matter_pattern': params.gray_matter_pattern,
        'dorsal_horn_width_mm': params.dorsal_horn_width_mm,
        'dorsal_horn_height_mm': params.dorsal_horn_height_mm,
        'ventral_horn_width_mm': params.ventral_horn_width_mm,
        'ventral_horn_height_mm': params.ventral_horn_height_mm,
        'lateral_horn_present': params.lateral_horn_present,
        'white_matter_thickness_mm': params.white_matter_thickness_mm,
        'enable_tract_columns': params.enable_tract_columns,
        'central_canal_diameter_mm': params.central_canal_diameter_mm if params.enable_central_canal else 0,
        'anterior_fissure_depth_mm': params.anterior_fissure_depth_mm if params.enable_anterior_fissure else 0,
        'num_guidance_channels': len(channels),
        'channel_diameter_um': params.channel_diameter_um,
        'channel_pattern': params.channel_pattern,
        'channel_region': params.channel_region,
        'pore_count': pore_count,
        'pore_size_um': params.pore_size_um,
        'meningeal_layers': {
            'pia_enabled': params.enable_pia_mater,
            'pia_thickness_um': params.pia_mater_thickness_um,
            'arachnoid_enabled': params.enable_arachnoid_mater,
            'arachnoid_thickness_um': params.arachnoid_thickness_um,
            'dura_enabled': params.enable_dura_mater,
            'dura_thickness_mm': params.dura_mater_thickness_mm,
            'subarachnoid_space_mm': params.subarachnoid_space_mm,
        },
        'root_zones': {
            'enabled': params.enable_root_entry_zones,
            'dorsal_root_count': len(dorsal_roots),
            'ventral_root_count': len(ventral_roots),
            'dorsal_diameter_mm': params.dorsal_root_diameter_mm,
            'ventral_diameter_mm': params.ventral_root_diameter_mm,
            'root_angle_deg': params.root_angle_deg,
        },
        'vascular': {
            'anterior_spinal_artery_enabled': params.enable_anterior_spinal_artery,
            'anterior_spinal_artery_diameter_mm': params.anterior_spinal_artery_diameter_mm,
            'posterior_spinal_arteries_enabled': params.enable_posterior_spinal_arteries,
            'posterior_spinal_artery_diameter_mm': params.posterior_spinal_artery_diameter_mm,
        },
        'scaffold_porosity': params.scaffold_porosity,
        'outer_radius_mm': outer_radius,
        'scaffold_type': 'spinal_cord'
    }

    return final_scaffold, stats


def generate_spinal_cord_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate a spinal cord scaffold from dictionary parameters.

    Args:
        params: Dictionary of parameters (will be unpacked into SpinalCordParams)

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle parameter name mappings for compatibility
    param_copy = params.copy()

    # Map legacy parameter names
    if 'cord_diameter_mm' in param_copy and 'total_diameter_mm' not in param_copy:
        param_copy['total_diameter_mm'] = param_copy.pop('cord_diameter_mm')
    if 'gray_matter_ratio' in param_copy and 'gray_matter_volume_ratio' not in param_copy:
        param_copy['gray_matter_volume_ratio'] = param_copy.pop('gray_matter_ratio')
    if 'channel_count' in param_copy and 'num_guidance_channels' not in param_copy:
        param_copy['num_guidance_channels'] = param_copy.pop('channel_count')
    if 'enable_white_matter_tracts' in param_copy and 'enable_tract_columns' not in param_copy:
        param_copy['enable_tract_columns'] = param_copy.pop('enable_white_matter_tracts')
    if 'pia_thickness_um' in param_copy and 'pia_mater_thickness_um' not in param_copy:
        param_copy['pia_mater_thickness_um'] = param_copy.pop('pia_thickness_um')

    # Filter to only valid SpinalCordParams fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(SpinalCordParams)}
    filtered_params = {k: v for k, v in param_copy.items() if k in valid_fields}

    return generate_spinal_cord(SpinalCordParams(**filtered_params))
