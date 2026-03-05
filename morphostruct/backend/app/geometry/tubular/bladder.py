"""
Bladder scaffold generator.

Generates multi-layer bladder wall structures with dome-shaped geometry
representing urothelium, lamina propria, and detrusor muscle layers.
Includes anatomically accurate features like rugae, trigone, and openings.
"""

import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BladderParams:
    """
    Parameters for bladder scaffold generation.

    Based on human urinary bladder anatomy:
    - Urothelium: transitional epithelium (innermost)
    - Lamina propria: connective tissue submucosa
    - Detrusor muscle: three smooth muscle layers
    - Serosa/Adventitia: outer covering
    """

    # === Basic Geometry ===
    dome_diameter_mm: float = 70.0  # Bladder diameter when moderately full (~50-100mm)
    wall_thickness_empty_mm: float = 3.3  # Wall thickness when empty (~3-5mm)
    dome_curvature: float = 0.6  # 0.5 = hemisphere, 1.0 = full sphere
    dome_height_mm: Optional[float] = None  # Auto-calculated if None

    # === Layer Configuration (detailed) ===
    # Urothelium (innermost epithelial layer)
    urothelium_thickness_um: float = 125.0  # ~100-150 um transitional epithelium
    enable_urothelium: bool = True

    # Lamina propria (submucosa)
    lamina_propria_thickness_um: float = 700.0  # ~600-800 um loose connective tissue
    enable_lamina_propria: bool = True

    # Detrusor muscle (smooth muscle)
    detrusor_thickness_mm: float = 2.25  # Major structural layer ~2.0-2.5mm
    enable_detrusor_muscle: bool = True
    detrusor_layer_count: int = 3  # Inner/middle/outer muscle layers (L-C-L)

    # Serosa (outer covering)
    serosa_thickness_um: float = 125.0  # ~50-200 um outer layer
    enable_serosa: bool = True

    # Legacy: simple layer count (overridden by detailed layers if enabled)
    layer_count: int = 3

    # === Rugae (Mucosal Folds) ===
    rugae_height_um: float = 750.0  # Fold height when contracted (~500-1000 um)
    rugae_spacing_mm: float = 2.0  # Distance between folds
    enable_urothelial_folds: bool = False  # Generate rugae structures
    fold_count: int = 12  # Number of radial folds
    fold_variance: float = 0.1  # Random variation in fold depth

    # === Compliance and Mechanics ===
    compliance_ml_per_cmH2O: float = 25.0  # Bladder compliance (design spec)
    max_capacity_ml: float = 500.0  # Target capacity

    # === Trigone Region ===
    trigone_marker: bool = False  # Mark triangular smooth muscle region at base
    trigone_width_mm: float = 25.0  # Width of trigone
    trigone_height_mm: float = 30.0  # Vertical extent of trigone

    # === Ureteral/Urethral Openings ===
    enable_ureteral_openings: bool = False  # Two ureteral orifices
    ureteral_opening_diameter_mm: float = 3.5  # ~3-4 mm
    ureteral_spacing_mm: float = 25.0  # Distance between ureters
    urethral_opening_diameter_mm: float = 6.0  # Internal urethral orifice (~5-8mm)
    enable_urethral_opening: bool = False

    # === Vascular Network ===
    enable_suburothelial_capillaries: bool = False
    capillary_diameter_um: float = 8.0  # ~5-10 um submucosal capillaries
    capillary_spacing_mm: float = 0.4  # Spacing in lamina propria (~400 um)

    # === Muscle Bundle Features ===
    muscle_bundle_diameter_mm: float = 1.0  # Detrusor muscle bundle size (0.5-2mm)
    muscle_bundle_spacing_mm: float = 1.5  # Spacing between bundles
    enable_muscle_bundles: bool = False  # Generate discrete bundles

    # === Nerve Innervation Markers ===
    enable_nerve_markers: bool = False  # Mark innervation points
    nerve_density_per_cm2: float = 50.0  # Nerve ending density

    # === Porosity ===
    scaffold_porosity: float = 0.6  # Target porosity
    pore_size_um: float = 100.0  # For cell infiltration
    enable_pore_network: bool = False  # Generate porous structure

    # === Randomization ===
    position_noise_mm: float = 0.0  # Random variation
    random_seed: int = 42

    # === Resolution ===
    resolution: int = 20  # Sphere/cylinder segments

    def __post_init__(self):
        if self.layer_count < 2 or self.layer_count > 4:
            raise ValueError("layer_count must be between 2 and 4")
        if self.dome_curvature < 0.3 or self.dome_curvature > 1.0:
            raise ValueError("dome_curvature must be between 0.3 and 1.0")

        # Auto-calculate dome height if not provided
        if self.dome_height_mm is None:
            self.dome_height_mm = self.dome_diameter_mm * self.dome_curvature


def _create_dome_layer(radius: float, dome_height: float, resolution: int) -> m3d.Manifold:
    """
    Create a hemispherical dome at given radius.

    Args:
        radius: Radius of the dome
        dome_height: Height of the dome (replaces dome_curvature calculation)
        resolution: Number of segments for sphere approximation

    Returns:
        Dome-shaped manifold
    """
    sphere = m3d.Manifold.sphere(radius, resolution)

    # Create cutting plane - keep everything above z=-(radius - dome_height)
    cut_z = -(radius - dome_height)

    # Create a large cube to cut with
    cutter = m3d.Manifold.cube([radius * 4, radius * 4, radius * 4], center=True)
    cutter = cutter.translate([0, 0, cut_z - radius * 2])

    # Subtract lower part
    dome = sphere - cutter

    return dome


def _create_detailed_layers(params: BladderParams, inner_radius: float, outer_radius: float) -> list:
    """
    Create anatomically accurate bladder wall layers as concentric shells.

    Layers from inside to outside:
    1. Urothelium (transitional epithelium): 100-150 um
    2. Lamina propria (loose connective tissue): 600-800 um
    3. Detrusor muscle (3 sub-layers L-C-L): 2.0-2.5 mm
    4. Serosa/Adventitia (outer covering): 50-200 um

    Args:
        params: BladderParams with layer configuration
        inner_radius: Inner radius of bladder wall
        outer_radius: Outer radius of bladder wall

    Returns:
        List of layer manifolds
    """
    layers = []

    # Convert layer thicknesses to mm
    urothelium_mm = params.urothelium_thickness_um / 1000.0
    lamina_propria_mm = params.lamina_propria_thickness_um / 1000.0
    detrusor_mm = params.detrusor_thickness_mm
    serosa_mm = params.serosa_thickness_um / 1000.0

    # Calculate total thickness from detailed layers
    total_detailed = urothelium_mm + lamina_propria_mm + detrusor_mm + serosa_mm

    # Scale factor to fit within wall thickness
    scale = params.wall_thickness_empty_mm / total_detailed

    urothelium_mm *= scale
    lamina_propria_mm *= scale
    detrusor_mm *= scale
    serosa_mm *= scale

    # Current radius tracker
    current_radius = inner_radius

    # 1. Urothelium layer (innermost)
    if params.enable_urothelium and urothelium_mm > 0:
        r_inner = current_radius
        r_outer = current_radius + urothelium_mm

        outer_dome = _create_dome_layer(r_outer, params.dome_height_mm, params.resolution)
        inner_dome = _create_dome_layer(r_inner, params.dome_height_mm, params.resolution)

        urothelium_layer = outer_dome - inner_dome
        layers.append(urothelium_layer)
        current_radius = r_outer

    # 2. Lamina propria layer (submucosa with capillaries)
    if params.enable_lamina_propria and lamina_propria_mm > 0:
        r_inner = current_radius
        r_outer = current_radius + lamina_propria_mm

        outer_dome = _create_dome_layer(r_outer, params.dome_height_mm, params.resolution)
        inner_dome = _create_dome_layer(r_inner, params.dome_height_mm, params.resolution)

        lamina_propria_layer = outer_dome - inner_dome
        layers.append(lamina_propria_layer)
        current_radius = r_outer

    # 3. Detrusor muscle layer (3 sub-layers: L-C-L pattern)
    if params.enable_detrusor_muscle and detrusor_mm > 0:
        sublayer_thickness = detrusor_mm / params.detrusor_layer_count

        for i in range(params.detrusor_layer_count):
            r_inner = current_radius
            r_outer = current_radius + sublayer_thickness

            outer_dome = _create_dome_layer(r_outer, params.dome_height_mm, params.resolution)
            inner_dome = _create_dome_layer(r_inner, params.dome_height_mm, params.resolution)

            detrusor_sublayer = outer_dome - inner_dome
            layers.append(detrusor_sublayer)
            current_radius = r_outer

    # 4. Serosa layer (outermost)
    if params.enable_serosa and serosa_mm > 0:
        r_inner = current_radius
        r_outer = current_radius + serosa_mm

        outer_dome = _create_dome_layer(r_outer, params.dome_height_mm, params.resolution)
        inner_dome = _create_dome_layer(r_inner, params.dome_height_mm, params.resolution)

        serosa_layer = outer_dome - inner_dome
        layers.append(serosa_layer)

    return layers


def _create_rugae(
    params: BladderParams,
    inner_radius: float,
    trigone_region: Optional[tuple] = None
) -> list:
    """
    Create rugae (mucosal folds) on the inner bladder surface.

    Rugae are sinusoidal folds present when bladder is contracted/empty.
    They flatten as the bladder fills. The trigone region remains smooth.

    Args:
        params: BladderParams with rugae configuration
        inner_radius: Inner radius of bladder
        trigone_region: Optional (center_z, width, height) of trigone to exclude

    Returns:
        List of fold manifolds to add to inner surface
    """
    if not params.enable_urothelial_folds:
        return []

    np.random.seed(params.random_seed)

    folds = []
    rugae_height_mm = params.rugae_height_um / 1000.0

    # Use pre-calculated dome height for fold placement
    cut_z = -(inner_radius - params.dome_height_mm)

    # Generate radial folds around the dome
    for i in range(params.fold_count):
        # Angular position of this fold
        theta = (2 * np.pi * i) / params.fold_count

        # Variation in fold depth
        variance_factor = 1.0 + np.random.uniform(-params.fold_variance, params.fold_variance)
        fold_height = rugae_height_mm * variance_factor

        # Create fold as a thin elongated box rotated into position
        # Fold runs from bottom to near top of dome
        fold_length = params.dome_height_mm * 0.8
        fold_width = params.rugae_spacing_mm * 0.3  # Thin fold

        # Create the fold shape
        fold = m3d.Manifold.cube([fold_width, fold_height, fold_length], center=True)

        # Position at inner radius, rotated around z-axis
        x_pos = (inner_radius - fold_height / 2) * np.cos(theta)
        y_pos = (inner_radius - fold_height / 2) * np.sin(theta)
        z_pos = cut_z + fold_length / 2 + params.dome_height_mm * 0.1

        # Rotate to align radially
        fold = fold.rotate([0, 0, np.degrees(theta)])
        fold = fold.translate([x_pos, y_pos, z_pos])

        # Check if in trigone region (skip if so)
        if trigone_region is not None:
            trigone_center_z, trigone_width, trigone_height = trigone_region
            # Trigone is at the base (lower z values)
            if z_pos < trigone_center_z + trigone_height / 2:
                # Check angular position - trigone spans front of bladder
                if abs(theta - np.pi) < np.pi / 3:  # Front 120 degrees
                    continue

        folds.append(fold)

    return folds


def _create_trigone(params: BladderParams, inner_radius: float, wall_thickness: float) -> tuple:
    """
    Create the trigone region: a smooth triangular area at the bladder base
    with ureteral and urethral openings.

    The trigone is bounded by:
    - Two ureteral orifices at the posterosuperior corners
    - One urethral orifice at the inferior apex

    Args:
        params: BladderParams with trigone and opening configuration
        inner_radius: Inner radius of bladder
        wall_thickness: Total wall thickness for openings

    Returns:
        Tuple of (trigone_marker, openings_list)
    """
    markers = []
    openings = []

    # Calculate base position (bottom of dome) using pre-calculated height
    base_z = -(inner_radius - params.dome_height_mm)

    # Trigone marker (thin triangular prism at base)
    if params.trigone_marker:
        # Create triangular cross-section for trigone
        # Vertices of triangle (in XY plane)
        half_width = params.trigone_width_mm / 2

        # Ureteral positions (top corners of triangle)
        ureteral_y = params.trigone_height_mm * 0.7
        urethral_y = 0  # Bottom apex

        # Create a thin triangular marker
        # Use a box approximation for the marker
        marker_thickness = 0.5  # mm
        marker = m3d.Manifold.cube([
            params.trigone_width_mm,
            params.trigone_height_mm,
            marker_thickness
        ], center=True)

        # Position at bladder base, on the front surface
        marker = marker.translate([0, inner_radius - marker_thickness / 2, base_z + params.trigone_height_mm / 2])
        markers.append(marker)

    # Ureteral openings (two paired orifices at posterosuperior corners)
    if params.enable_ureteral_openings:
        ureteral_radius = params.ureteral_opening_diameter_mm / 2
        half_spacing = params.ureteral_spacing_mm / 2

        # Position at top corners of trigone
        ureteral_z = base_z + params.trigone_height_mm * 0.7
        ureteral_y = inner_radius - wall_thickness / 2

        for x_sign in [-1, 1]:
            x_pos = x_sign * half_spacing

            # Create cylindrical opening through wall
            opening = m3d.Manifold.cylinder(
                wall_thickness * 1.5,  # Extend through full wall
                ureteral_radius,
                ureteral_radius,
                params.resolution
            )
            # Rotate to point radially inward
            opening = opening.rotate([90, 0, 0])
            opening = opening.translate([x_pos, ureteral_y + wall_thickness / 2, ureteral_z])
            openings.append(opening)

    # Urethral opening (single orifice at inferior apex)
    if params.enable_urethral_opening:
        urethral_radius = params.urethral_opening_diameter_mm / 2

        # Position at bottom apex of trigone
        urethral_z = base_z + params.trigone_height_mm * 0.1
        urethral_y = inner_radius - wall_thickness / 2

        # Create cylindrical opening through wall
        opening = m3d.Manifold.cylinder(
            wall_thickness * 1.5,
            urethral_radius,
            urethral_radius,
            params.resolution
        )
        # Rotate to point radially inward (downward at base)
        opening = opening.rotate([90, 0, 0])
        opening = opening.translate([0, urethral_y + wall_thickness / 2, urethral_z])
        openings.append(opening)

    return markers, openings


def _create_vascular_network(
    params: BladderParams,
    inner_radius: float,
    lamina_propria_start: float,
    lamina_propria_thickness: float
) -> list:
    """
    Create submucosal vascular network (capillary plexus) in the lamina propria.

    Capillaries are modeled as small cylindrical channels running through
    the connective tissue layer. Density is higher at the trigone region.

    Args:
        params: BladderParams with vascular configuration
        inner_radius: Inner radius of bladder
        lamina_propria_start: Starting radius of lamina propria layer
        lamina_propria_thickness: Thickness of lamina propria

    Returns:
        List of capillary channel manifolds
    """
    if not params.enable_suburothelial_capillaries:
        return []

    np.random.seed(params.random_seed + 100)  # Different seed for vascular

    capillaries = []
    capillary_radius = params.capillary_diameter_um / 1000.0 / 2  # Convert to mm radius
    spacing = params.capillary_spacing_mm

    # Position capillaries in the middle of lamina propria
    capillary_layer_radius = lamina_propria_start + lamina_propria_thickness / 2

    # Calculate dome coverage using pre-calculated height
    base_z = -(inner_radius - params.dome_height_mm)

    # Grid of capillaries around the dome surface
    # Use spherical coordinates for placement
    num_phi = int(np.pi * capillary_layer_radius / spacing)  # Elevation divisions

    for phi_idx in range(1, num_phi):  # Skip poles
        phi = np.pi * phi_idx / num_phi  # 0 to pi
        z = capillary_layer_radius * np.cos(phi)

        # Skip if below dome cut
        if z < base_z:
            continue

        # Circumference at this elevation
        ring_radius = capillary_layer_radius * np.sin(phi)
        num_theta = max(4, int(2 * np.pi * ring_radius / spacing))

        for theta_idx in range(num_theta):
            theta = 2 * np.pi * theta_idx / num_theta

            x = ring_radius * np.cos(theta)
            y = ring_radius * np.sin(theta)

            # Add small random variation
            noise = params.position_noise_mm
            if noise > 0:
                x += np.random.uniform(-noise, noise)
                y += np.random.uniform(-noise, noise)
                z += np.random.uniform(-noise, noise)

            # Create short capillary segment (radially oriented)
            capillary_length = lamina_propria_thickness * 0.8
            capillary = m3d.Manifold.cylinder(
                capillary_length,
                capillary_radius,
                capillary_radius,
                max(6, params.resolution // 3)
            )

            # Orient radially (pointing outward from center)
            # Calculate radial direction
            radial_dir = np.array([x, y, z])
            radial_length = np.linalg.norm(radial_dir)
            if radial_length > 0:
                radial_dir = radial_dir / radial_length

            # Calculate rotation to align cylinder with radial direction
            # Default cylinder is along z-axis
            z_axis = np.array([0, 0, 1])

            # Rotation axis is cross product
            rot_axis = np.cross(z_axis, radial_dir)
            rot_axis_length = np.linalg.norm(rot_axis)

            if rot_axis_length > 0.001:  # Not parallel to z
                rot_axis = rot_axis / rot_axis_length
                rot_angle = np.degrees(np.arccos(np.dot(z_axis, radial_dir)))

                # Apply rotation (manifold3d uses degrees)
                capillary = capillary.rotate([
                    rot_angle * rot_axis[0],
                    rot_angle * rot_axis[1],
                    rot_angle * rot_axis[2]
                ])

            # Position at capillary layer radius
            capillary = capillary.translate([x, y, z])
            capillaries.append(capillary)

    return capillaries


def _create_muscle_bundles(
    params: BladderParams,
    detrusor_start: float,
    detrusor_thickness: float
) -> list:
    """
    Create muscle bundle markers within the detrusor layer.

    The detrusor has a characteristic L-C-L (longitudinal-circular-longitudinal)
    fiber arrangement:
    - Inner layer: longitudinal fibers
    - Middle layer: circular fibers
    - Outer layer: longitudinal fibers

    Args:
        params: BladderParams with muscle bundle configuration
        detrusor_start: Starting radius of detrusor layer
        detrusor_thickness: Total thickness of detrusor

    Returns:
        List of muscle bundle marker manifolds
    """
    if not params.enable_muscle_bundles:
        return []

    np.random.seed(params.random_seed + 200)

    bundles = []
    bundle_radius = params.muscle_bundle_diameter_mm / 2
    spacing = params.muscle_bundle_spacing_mm

    sublayer_thickness = detrusor_thickness / params.detrusor_layer_count

    # Calculate dome parameters using pre-calculated height
    outer_radius = params.dome_diameter_mm / 2
    base_z = -(outer_radius - params.dome_height_mm)

    # Create bundles in each detrusor sublayer
    for sublayer_idx in range(params.detrusor_layer_count):
        # Determine orientation: L-C-L pattern
        # 0 = longitudinal (along meridians), 1 = circular (along parallels)
        if params.detrusor_layer_count == 3:
            orientations = ['longitudinal', 'circular', 'longitudinal']
        else:
            # Alternate for other layer counts
            orientations = ['longitudinal' if i % 2 == 0 else 'circular'
                          for i in range(params.detrusor_layer_count)]

        orientation = orientations[sublayer_idx]

        # Radius of this sublayer (middle of sublayer)
        sublayer_radius = detrusor_start + (sublayer_idx + 0.5) * sublayer_thickness

        # Number of bundles based on circumference and spacing
        if orientation == 'circular':
            # Circular bundles: rings at different elevations
            num_rings = int(params.dome_height_mm / spacing)

            for ring_idx in range(1, num_rings):
                # Z position of this ring
                z_frac = ring_idx / num_rings
                z = base_z + z_frac * params.dome_height_mm

                # Radius at this elevation (following dome shape)
                # For a sphere: r = sqrt(R^2 - z^2)
                local_radius = np.sqrt(max(0, sublayer_radius**2 - z**2))

                if local_radius < bundle_radius * 2:
                    continue

                # Create torus-like ring of bundles
                circumference = 2 * np.pi * local_radius
                num_bundles = int(circumference / spacing)

                for i in range(num_bundles):
                    theta = 2 * np.pi * i / num_bundles
                    x = local_radius * np.cos(theta)
                    y = local_radius * np.sin(theta)

                    # Small cylinder segment representing bundle
                    bundle_length = min(spacing * 0.7, 2.0)
                    bundle = m3d.Manifold.cylinder(
                        bundle_length,
                        bundle_radius,
                        bundle_radius,
                        max(6, params.resolution // 4)
                    )

                    # Rotate to tangential direction (circular)
                    bundle = bundle.rotate([0, 90, np.degrees(theta) + 90])
                    bundle = bundle.translate([x, y, z])
                    bundles.append(bundle)

        else:  # longitudinal
            # Longitudinal bundles: run from base to apex
            circumference = 2 * np.pi * sublayer_radius
            num_meridians = int(circumference / spacing)

            for i in range(num_meridians):
                theta = 2 * np.pi * i / num_meridians

                # Create bundle along meridian
                bundle_length = params.dome_height_mm * 0.6
                bundle = m3d.Manifold.cylinder(
                    bundle_length,
                    bundle_radius,
                    bundle_radius,
                    max(6, params.resolution // 4)
                )

                # Position along meridian
                x = sublayer_radius * np.cos(theta)
                y = sublayer_radius * np.sin(theta)
                z = base_z + params.dome_height_mm * 0.3

                # Rotate to align with meridian (vertical)
                bundle = bundle.translate([x, y, z])
                bundles.append(bundle)

    return bundles


def _create_pore_network(
    params: BladderParams,
    inner_radius: float,
    wall_thickness: float
) -> list:
    """
    Create interconnected pore network for tissue engineering scaffolds.

    Pores are distributed spherical voids throughout the wall thickness
    to allow cell infiltration and nutrient transport.

    Args:
        params: BladderParams with porosity configuration
        inner_radius: Inner radius of bladder
        wall_thickness: Total wall thickness

    Returns:
        List of pore (sphere) manifolds to subtract from scaffold
    """
    if not params.enable_pore_network or params.scaffold_porosity <= 0:
        return []

    np.random.seed(params.random_seed + 300)

    pores = []
    pore_radius = params.pore_size_um / 1000.0 / 2  # Convert to mm radius

    # Calculate required pore density to achieve target porosity
    # Volume of single pore
    pore_volume = (4/3) * np.pi * pore_radius**3

    # Approximate wall volume (dome shell)
    outer_radius = inner_radius + wall_thickness

    # Simplified volume estimate for dome shell
    shell_volume = (2/3) * np.pi * (outer_radius**3 - inner_radius**3) * params.dome_curvature

    # Number of pores needed for target porosity
    target_void_volume = shell_volume * params.scaffold_porosity
    num_pores = int(target_void_volume / pore_volume)

    # Cap at reasonable maximum
    num_pores = min(num_pores, 5000)

    if num_pores < 1:
        return []

    # Calculate dome parameters using pre-calculated height
    base_z = -(outer_radius - params.dome_height_mm)

    # Distribute pores randomly within the wall
    for _ in range(num_pores):
        # Random position in spherical shell
        # Use rejection sampling for dome shape
        while True:
            # Random radius within wall
            r = inner_radius + np.random.uniform(pore_radius, wall_thickness - pore_radius)

            # Random spherical angles
            phi = np.arccos(1 - 2 * np.random.uniform(0, 1))  # Elevation
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuth

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Check if within dome (above base)
            if z >= base_z + pore_radius:
                break

        # Create pore
        pore = m3d.Manifold.sphere(pore_radius, max(8, params.resolution // 2))
        pore = pore.translate([x, y, z])
        pores.append(pore)

    return pores


def _create_nerve_markers(
    params: BladderParams,
    inner_radius: float,
    lamina_propria_start: float,
    lamina_propria_thickness: float,
    detrusor_start: float,
    detrusor_thickness: float,
    is_trigone_region: bool = False
) -> list:
    """
    Create nerve innervation markers in the bladder wall.

    Based on bladder neurophysiology:
    - Nerve marker density: 1-2% of tissue area as nerve markers
    - Higher density in trigone region (1:2 smooth muscle cells per nerve vs body 1:6-7 ratio)
    - Nerve channel diameter: 20-40 µm for bladder nerve fibers
    - Nerve fiber types: A-delta (1-6 µm) and C-fibers (0.2-1.5 µm)
    - Distribution: hierarchical across layers (urothelium → lamina propria → detrusor)
    - Spacing between nerve channels: 200-500 µm (nerve regeneration distance)

    Args:
        params: BladderParams with nerve marker configuration
        inner_radius: Inner radius of bladder
        lamina_propria_start: Starting radius of lamina propria layer
        lamina_propria_thickness: Thickness of lamina propria
        detrusor_start: Starting radius of detrusor layer
        detrusor_thickness: Total thickness of detrusor
        is_trigone_region: Whether to use higher density for trigone

    Returns:
        List of nerve marker (small cylinder) manifolds
    """
    if not params.enable_nerve_markers:
        return []

    np.random.seed(params.random_seed + 400)

    nerve_markers = []

    # Nerve fiber diameter: 20-40 µm average for bladder nerves
    nerve_diameter_um = 30.0  # Average of range
    nerve_radius = (nerve_diameter_um / 1000.0) / 2  # Convert to mm radius

    # Spacing between nerve channels: 200-500 µm
    nerve_spacing_mm = 0.35  # Average 350 µm

    # Adjust density based on region
    if is_trigone_region or params.trigone_marker:
        # Trigone has 3x higher nerve density (1:2 vs 1:6 ratio)
        density_factor = 3.0
    else:
        density_factor = 1.0

    # Calculate surface area to determine nerve count
    outer_radius = params.dome_diameter_mm / 2

    # Approximate surface area (hemispherical cap formula)
    # A = 2πRh for spherical cap, convert to cm²
    surface_area_cm2 = (2 * np.pi * outer_radius * params.dome_height_mm) / 100.0

    # Calculate number of nerve markers from density
    num_nerves = int(params.nerve_density_per_cm2 * surface_area_cm2 * density_factor)

    # Cap at reasonable maximum
    num_nerves = min(num_nerves, 3000)

    if num_nerves < 1:
        return []

    # Calculate dome parameters using pre-calculated height
    base_z = -(outer_radius - params.dome_height_mm)

    # Distribute nerve markers in lamina propria and detrusor layers
    # 40% in lamina propria (submucosal plexus), 60% in detrusor (myenteric plexus)
    num_lamina = int(num_nerves * 0.4)
    num_detrusor = num_nerves - num_lamina

    # Create nerve markers in lamina propria
    for _ in range(num_lamina):
        # Random position within lamina propria layer
        while True:
            # Random radius within lamina propria
            r = lamina_propria_start + np.random.uniform(0, lamina_propria_thickness)

            # Random spherical angles
            phi = np.arccos(1 - 2 * np.random.uniform(0, 1))  # Elevation
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuth

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Check if within dome (above base)
            if z >= base_z + nerve_radius:
                break

        # Create short cylindrical nerve marker
        nerve_length = nerve_spacing_mm * 0.5  # Short segment
        nerve = m3d.Manifold.cylinder(
            nerve_length,
            nerve_radius,
            nerve_radius,
            max(6, params.resolution // 4)
        )

        # Orient radially (pointing outward from center)
        radial_dir = np.array([x, y, z])
        radial_length = np.linalg.norm(radial_dir)
        if radial_length > 0:
            radial_dir = radial_dir / radial_length

        # Calculate rotation to align cylinder with radial direction
        z_axis = np.array([0, 0, 1])
        rot_axis = np.cross(z_axis, radial_dir)
        rot_axis_length = np.linalg.norm(rot_axis)

        if rot_axis_length > 0.001:
            rot_axis = rot_axis / rot_axis_length
            rot_angle = np.degrees(np.arccos(np.dot(z_axis, radial_dir)))

            nerve = nerve.rotate([
                rot_angle * rot_axis[0],
                rot_angle * rot_axis[1],
                rot_angle * rot_axis[2]
            ])

        nerve = nerve.translate([x, y, z])
        nerve_markers.append(nerve)

    # Create nerve markers in detrusor layer
    for _ in range(num_detrusor):
        # Random position within detrusor layer
        while True:
            # Random radius within detrusor
            r = detrusor_start + np.random.uniform(0, detrusor_thickness)

            # Random spherical angles
            phi = np.arccos(1 - 2 * np.random.uniform(0, 1))  # Elevation
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuth

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Check if within dome (above base)
            if z >= base_z + nerve_radius:
                break

        # Create short cylindrical nerve marker
        nerve_length = nerve_spacing_mm * 0.6  # Slightly longer in muscle layer
        nerve = m3d.Manifold.cylinder(
            nerve_length,
            nerve_radius,
            nerve_radius,
            max(6, params.resolution // 4)
        )

        # Orient radially or tangentially (mixed orientation in detrusor)
        if np.random.random() < 0.7:  # 70% radial
            radial_dir = np.array([x, y, z])
        else:  # 30% tangential (along muscle fibers)
            # Tangent direction perpendicular to radial
            tangent_theta = theta + np.pi / 2
            radial_dir = np.array([np.cos(tangent_theta), np.sin(tangent_theta), 0])

        radial_length = np.linalg.norm(radial_dir)
        if radial_length > 0:
            radial_dir = radial_dir / radial_length

        # Calculate rotation to align cylinder with direction
        z_axis = np.array([0, 0, 1])
        rot_axis = np.cross(z_axis, radial_dir)
        rot_axis_length = np.linalg.norm(rot_axis)

        if rot_axis_length > 0.001:
            rot_axis = rot_axis / rot_axis_length
            rot_angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, radial_dir), -1.0, 1.0)))

            nerve = nerve.rotate([
                rot_angle * rot_axis[0],
                rot_angle * rot_axis[1],
                rot_angle * rot_axis[2]
            ])

        nerve = nerve.translate([x, y, z])
        nerve_markers.append(nerve)

    return nerve_markers


def generate_bladder(params: BladderParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a bladder scaffold with multi-layer dome-shaped wall.

    Creates a hemispherical dome structure with layered walls representing
    the urothelium (inner lining), lamina propria, detrusor muscle, and serosa.

    Optionally includes:
    - Rugae (mucosal folds) on inner surface
    - Trigone region with ureteral/urethral openings
    - Vascular network in submucosa
    - Muscle bundle markers in detrusor
    - Pore network for tissue engineering

    Args:
        params: BladderParams with generation parameters

    Returns:
        Tuple of (manifold, stats_dict)

    Raises:
        ValueError: If parameters are invalid
    """
    outer_radius = params.dome_diameter_mm / 2
    inner_radius = outer_radius - params.wall_thickness_empty_mm

    if inner_radius <= 0:
        raise ValueError("Wall thickness exceeds radius")

    from ..core import batch_union

    # Create detailed anatomical layers
    layers = _create_detailed_layers(params, inner_radius, outer_radius)

    # Combine all layers
    bladder = batch_union(layers)

    # Calculate layer boundaries for feature placement
    urothelium_mm = params.urothelium_thickness_um / 1000.0
    lamina_propria_mm = params.lamina_propria_thickness_um / 1000.0
    detrusor_mm = params.detrusor_thickness_mm
    serosa_mm = params.serosa_thickness_um / 1000.0

    total_detailed = urothelium_mm + lamina_propria_mm + detrusor_mm + serosa_mm
    scale = params.wall_thickness_empty_mm / total_detailed

    urothelium_mm *= scale
    lamina_propria_mm *= scale
    detrusor_mm *= scale
    serosa_mm *= scale

    lamina_propria_start = inner_radius + urothelium_mm
    detrusor_start = lamina_propria_start + lamina_propria_mm

    # Add rugae (mucosal folds) - use pre-calculated dome_height
    base_z = -(outer_radius - params.dome_height_mm)

    trigone_region = None
    if params.trigone_marker:
        trigone_region = (base_z + params.trigone_height_mm / 2,
                         params.trigone_width_mm,
                         params.trigone_height_mm)

    rugae = _create_rugae(params, inner_radius, trigone_region)
    if rugae:
        rugae_combined = batch_union(rugae)
        bladder = bladder + rugae_combined

    # Add trigone markers and openings
    trigone_markers, openings = _create_trigone(
        params, inner_radius, params.wall_thickness_empty_mm
    )

    if trigone_markers:
        markers_combined = batch_union(trigone_markers)
        bladder = bladder + markers_combined

    if openings:
        openings_combined = batch_union(openings)
        bladder = bladder - openings_combined

    # Add vascular network
    capillaries = _create_vascular_network(
        params, inner_radius, lamina_propria_start, lamina_propria_mm
    )
    if capillaries:
        capillaries_combined = batch_union(capillaries)
        bladder = bladder - capillaries_combined  # Subtract to create channels

    # Add muscle bundles
    bundles = _create_muscle_bundles(params, detrusor_start, detrusor_mm)
    if bundles:
        bundles_combined = batch_union(bundles)
        bladder = bladder + bundles_combined  # Add as markers

    # Add pore network
    pores = _create_pore_network(params, inner_radius, params.wall_thickness_empty_mm)
    if pores:
        pores_combined = batch_union(pores)
        bladder = bladder - pores_combined  # Subtract to create pores

    # Add nerve markers
    nerve_markers = _create_nerve_markers(
        params, inner_radius, lamina_propria_start, lamina_propria_mm,
        detrusor_start, detrusor_mm, params.trigone_marker
    )
    if nerve_markers:
        nerve_markers_combined = batch_union(nerve_markers)
        bladder = bladder + nerve_markers_combined  # Add as markers

    # Calculate statistics
    mesh = bladder.to_mesh()
    volume = bladder.volume() if hasattr(bladder, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'dome_diameter_mm': params.dome_diameter_mm,
        'wall_thickness_empty_mm': params.wall_thickness_empty_mm,
        'layer_count': len(layers),
        'configured_layer_count': params.layer_count,
        'dome_height_mm': params.dome_height_mm,
        'dome_curvature': params.dome_curvature,
        'max_capacity_ml': params.max_capacity_ml,
        'layer_thicknesses_um': {
            'urothelium': params.urothelium_thickness_um,
            'lamina_propria': params.lamina_propria_thickness_um,
            'detrusor': params.detrusor_thickness_mm * 1000,
            'serosa': params.serosa_thickness_um
        },
        'layer_thicknesses_mm_scaled': {
            'urothelium': urothelium_mm,
            'lamina_propria': lamina_propria_mm,
            'detrusor': detrusor_mm,
            'serosa': serosa_mm
        },
        'features_enabled': {
            'rugae': params.enable_urothelial_folds,
            'trigone': params.trigone_marker,
            'ureteral_openings': params.enable_ureteral_openings,
            'urethral_opening': params.enable_urethral_opening,
            'vascular_network': params.enable_suburothelial_capillaries,
            'muscle_bundles': params.enable_muscle_bundles,
            'pore_network': params.enable_pore_network,
            'nerve_markers': params.enable_nerve_markers
        },
        'feature_counts': {
            'rugae': len(rugae),
            'capillaries': len(capillaries),
            'muscle_bundles': len(bundles),
            'pores': len(pores),
            'openings': len(openings),
            'nerve_markers': len(nerve_markers)
        },
        'rugae_height_um': params.rugae_height_um,
        'rugae_spacing_mm': params.rugae_spacing_mm,
        'compliance_ml_per_cmH2O': params.compliance_ml_per_cmH2O,
        'scaffold_porosity': params.scaffold_porosity,
        'scaffold_type': 'bladder'
    }

    return bladder, stats


def generate_bladder_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate a bladder scaffold from dictionary parameters.

    Args:
        params: Dictionary of parameters (will be unpacked into BladderParams)

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle parameter name mappings for compatibility
    param_copy = params.copy()

    # Map legacy parameter names
    if 'diameter_mm' in param_copy and 'dome_diameter_mm' not in param_copy:
        param_copy['dome_diameter_mm'] = param_copy.pop('diameter_mm')
    if 'wall_thickness_mm' in param_copy and 'wall_thickness_empty_mm' not in param_copy:
        param_copy['wall_thickness_empty_mm'] = param_copy.pop('wall_thickness_mm')

    # Map additional legacy names
    if 'epithelial_thickness_um' in param_copy and 'urothelium_thickness_um' not in param_copy:
        param_copy['urothelium_thickness_um'] = param_copy.pop('epithelial_thickness_um')
    if 'muscle_layer_thickness_mm' in param_copy and 'detrusor_thickness_mm' not in param_copy:
        param_copy['detrusor_thickness_mm'] = param_copy.pop('muscle_layer_thickness_mm')
    if 'enable_rugae' in param_copy and 'enable_urothelial_folds' not in param_copy:
        param_copy['enable_urothelial_folds'] = param_copy.pop('enable_rugae')
    if 'trigone_length_mm' in param_copy and 'trigone_height_mm' not in param_copy:
        param_copy['trigone_height_mm'] = param_copy.pop('trigone_length_mm')
    if 'ureteral_orifice_diameter_mm' in param_copy and 'ureteral_opening_diameter_mm' not in param_copy:
        param_copy['ureteral_opening_diameter_mm'] = param_copy.pop('ureteral_orifice_diameter_mm')
    if 'urethral_orifice_diameter_mm' in param_copy and 'urethral_opening_diameter_mm' not in param_copy:
        param_copy['urethral_opening_diameter_mm'] = param_copy.pop('urethral_orifice_diameter_mm')
    if 'vascular_channel_diameter_um' in param_copy and 'capillary_diameter_um' not in param_copy:
        param_copy['capillary_diameter_um'] = param_copy.pop('vascular_channel_diameter_um')
    if 'vascular_spacing_um' in param_copy and 'capillary_spacing_mm' not in param_copy:
        param_copy['capillary_spacing_mm'] = param_copy.pop('vascular_spacing_um') / 1000.0
    if 'pore_diameter_um' in param_copy and 'pore_size_um' not in param_copy:
        param_copy['pore_size_um'] = param_copy.pop('pore_diameter_um')
    if 'porosity_percent' in param_copy and 'scaffold_porosity' not in param_copy:
        param_copy['scaffold_porosity'] = param_copy.pop('porosity_percent') / 100.0
    if 'muscle_orientation' in param_copy:
        param_copy.pop('muscle_orientation')  # Handled internally as L-C-L

    # Filter to only valid BladderParams fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(BladderParams)}
    filtered_params = {k: v for k, v in param_copy.items() if k in valid_fields}

    return generate_bladder(BladderParams(**filtered_params))
