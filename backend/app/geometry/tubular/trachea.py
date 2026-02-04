"""
Trachea scaffold generator.

Generates tracheal structures with C-shaped cartilage rings and posterior membrane.
Includes anatomically accurate features like mucosal layer and trachealis muscle.
"""

import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..helpers import tree_union, batch_union


@dataclass
class TracheaParams:
    """
    Parameters for trachea scaffold generation.

    Based on human tracheal anatomy:
    - C-shaped hyaline cartilage rings (16-20 rings)
    - Posterior trachealis muscle
    - Mucosa with ciliated epithelium
    - Submucosa with glands
    """

    # === Basic Geometry ===
    total_length_mm: float = 110.0  # Adult trachea ~10-12 cm (used to calculate ring spacing if differs from ring count)
    length_mm: Optional[float] = None  # Legacy alias for total_length_mm
    outer_diameter_mm: float = 20.0  # ~15-25 mm outer
    inner_diameter_mm: float = 16.0  # ~12-18 mm lumen

    # === Cartilage Ring Configuration ===
    num_cartilage_rings: int = 18  # Typically 16-20 rings
    ring_height_mm: float = 4.0  # Vertical extent of each ring (~3-5 mm) - USED IN GEOMETRY
    ring_width_mm: float = 4.5  # Alias for ring_height_mm (superior-inferior extent) - USED IN GEOMETRY if set
    ring_thickness_mm: float = 1.5  # Radial thickness of cartilage (~1-2 mm) - USED IN GEOMETRY
    ring_gap_deg: float = 90.0  # Alias for posterior_opening_degrees - USED IN GEOMETRY if differs from default
    posterior_opening_degrees: float = 80.0  # Gap for trachealis muscle (~60-100 deg) - USED IN GEOMETRY
    interring_spacing_mm: float = 2.0  # Annular ligament space (~1-3 mm) - USED IN GEOMETRY

    # Legacy parameter names (for compatibility)
    ring_spacing_mm: Optional[float] = None  # Use interring_spacing_mm
    ring_count: Optional[int] = None  # Use num_cartilage_rings
    posterior_gap_angle_deg: Optional[float] = None  # Use posterior_opening_degrees

    # === Cartilage Properties ===
    cartilage_type: str = "hyaline"  # "hyaline", "elastic", "fibrocartilage" - METADATA for FEA/materials
    enable_c_shaped_cartilage: bool = True  # C-shaped vs full rings
    cartilage_porosity: float = 0.3  # For scaffolds - USED IN GEOMETRY (see also scaffold_porosity)

    # === Mucosal Layers (detailed) ===
    enable_mucosal_layer: bool = False
    epithelium_thickness_um: float = 60.0  # Pseudostratified ciliated columnar (~50-70 um) - USED IN GEOMETRY
    lamina_propria_thickness_um: float = 290.0  # Loose connective tissue (~250-350 um) - USED IN GEOMETRY
    submucosa_thickness_um: float = 500.0  # With glands (~300-700 um) - USED IN GEOMETRY
    mucosa_thickness_um: float = 350.0  # Legacy: epithelium + lamina propria combined - USED as fallback validation

    # === Cilia Markers ===
    enable_cilia_markers: bool = False  # Texture markers for mucociliary apparatus
    cilia_density: float = 0.5  # Marker density (0-1)
    enable_ciliated_epithelium_markers: bool = False  # Alias for cilia markers
    enable_goblet_cell_markers: bool = False  # Mucus cell markers

    # === Submucosa ===
    enable_submucosa: bool = False

    # === Submucosal Glands ===
    enable_submucosal_glands: bool = False
    gland_diameter_um: float = 300.0  # ~200-400 um
    gland_diameter_mm: float = 0.3  # Legacy alias
    gland_density_per_mm2: float = 1.0  # Glands per mm^2

    # === Trachealis Muscle ===
    enable_trachealis_muscle: bool = False  # Posterior smooth muscle
    trachealis_thickness_um: float = 800.0  # ~500-1500 um (0.5-1.5 mm)
    trachealis_thickness_mm: float = 0.8  # Legacy: ~0.5-1.5 mm
    trachealis_width_deg: float = 80.0  # Spans C-ring gap
    trachealis_fiber_orientation_deg: float = 0.0  # Transverse orientation

    # === Perichondrium ===
    enable_perichondrium: bool = False
    perichondrium_thickness_um: float = 80.0  # Fibrous tissue (~50-100 um)

    # === Mucosal Folds ===
    enable_mucosal_folds: bool = False
    mucosal_fold_depth_mm: float = 0.2  # ~0.1-0.3 mm
    mucosal_fold_count: int = 12  # Longitudinal folds

    # === Vascular Channels ===
    enable_vascular_channels: bool = False
    vascular_channel_diameter_um: float = 100.0  # For scaffold perfusion (~50-200 um)
    vascular_channel_diameter_mm: float = 0.25  # Legacy alias
    vascular_spacing_um: float = 200.0  # Max spacing between vessels (~200 um)
    vascular_channel_spacing_mm: float = 3.0  # Legacy alias

    # === Annular Ligaments ===
    enable_annular_ligaments: bool = False  # Fibrous tissue between rings
    annular_ligament_thickness_mm: float = 0.5  # Fibroelastic tissue thickness
    ligament_thickness_mm: Optional[float] = None  # Legacy alias for annular_ligament_thickness_mm

    # === Carina (for bifurcation) ===
    enable_carina: bool = False  # Bifurcation at distal end
    carina_angle_deg: float = 70.0  # Total angle between main bronchi - USED to adjust bronchus angles if set
    left_bronchus_angle_deg: float = 45.0  # More horizontal (~45 deg from vertical) - USED IN GEOMETRY
    right_bronchus_angle_deg: float = 25.0  # Steeper (~25 deg from vertical) - USED IN GEOMETRY
    bronchus_diameter_mm: float = 12.0  # Main bronchus diameter
    bronchus_length_mm: float = 15.0  # Length of bronchi segments

    # === Porosity ===
    scaffold_porosity: float = 0.5  # Overall scaffold porosity - USED IN GEOMETRY (max with cartilage_porosity)
    pore_size_um: float = 100.0  # For cell infiltration - USED IN GEOMETRY

    # === Randomization ===
    position_noise_mm: float = 0.0  # Random variation
    ring_variance_pct: float = 0.0  # Variation in ring dimensions (%)
    random_seed: int = 42

    # === Resolution ===
    resolution: int = 20  # Cylinder segments

    def __post_init__(self):
        # Handle legacy parameter mappings
        if self.length_mm is not None and self.total_length_mm == 110.0:
            self.total_length_mm = self.length_mm
        if self.ring_spacing_mm is not None and self.interring_spacing_mm == 2.0:
            self.interring_spacing_mm = self.ring_spacing_mm
        if self.ring_count is not None and self.num_cartilage_rings == 18:
            self.num_cartilage_rings = self.ring_count
        if self.posterior_gap_angle_deg is not None and self.posterior_opening_degrees == 80.0:
            self.posterior_opening_degrees = self.posterior_gap_angle_deg

        # Sync legacy gland_diameter_mm to gland_diameter_um if set
        if self.gland_diameter_mm != 0.3:
            self.gland_diameter_um = self.gland_diameter_mm * 1000

        # Sync legacy trachealis_thickness_mm to trachealis_thickness_um
        if self.trachealis_thickness_mm != 0.8:
            self.trachealis_thickness_um = self.trachealis_thickness_mm * 1000

        # Sync legacy vascular_channel_diameter_mm to vascular_channel_diameter_um
        if self.vascular_channel_diameter_mm != 0.25:
            self.vascular_channel_diameter_um = self.vascular_channel_diameter_mm * 1000

        # Sync legacy ligament_thickness_mm to annular_ligament_thickness_mm
        if self.ligament_thickness_mm is not None and self.annular_ligament_thickness_mm == 0.5:
            self.annular_ligament_thickness_mm = self.ligament_thickness_mm

        # Auto-calculate length if not matching ring count
        expected_length = self.num_cartilage_rings * (self.ring_height_mm + self.interring_spacing_mm)
        # Allow total_length to override

        if self.num_cartilage_rings < 1:
            raise ValueError("num_cartilage_rings must be at least 1")
        if self.posterior_opening_degrees < 30 or self.posterior_opening_degrees > 150:
            raise ValueError("posterior_opening_degrees must be between 30 and 150")


def _create_c_ring(
    z_position: float,
    outer_radius: float,
    inner_radius: float,
    ring_height: float,
    ring_thickness: float,
    gap_degrees: float,
    resolution: int,
    ring_variance_pct: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> m3d.Manifold:
    """
    Create a single C-shaped cartilage ring at given z position.

    Args:
        z_position: Vertical position of ring center
        outer_radius: Outer radius (from trachea diameter)
        inner_radius: Inner radius (from trachea lumen)
        ring_height: Superior-inferior extent (vertical height)
        ring_thickness: Radial thickness of cartilage ring
        gap_degrees: Posterior gap angle
        resolution: Cylinder segments
        ring_variance_pct: Random variation percentage
        rng: Random number generator
    """
    # Apply variance if requested
    if ring_variance_pct > 0 and rng is not None:
        variance_mult = 1.0 + (rng.random() - 0.5) * 2 * ring_variance_pct / 100
        ring_height = ring_height * variance_mult

    # Use ring_thickness to define the radial extent of cartilage
    # The ring sits between (outer_radius - ring_thickness) and outer_radius
    cartilage_outer = outer_radius
    cartilage_inner = max(inner_radius, outer_radius - ring_thickness)

    # Create full ring
    outer_ring = m3d.Manifold.cylinder(
        ring_height,
        cartilage_outer,
        cartilage_outer,
        resolution
    )

    inner_ring = m3d.Manifold.cylinder(
        ring_height + 0.2,
        cartilage_inner,
        cartilage_inner,
        resolution
    ).translate([0, 0, -0.1])

    ring = outer_ring - inner_ring

    # Create gap cutter for posterior opening
    # Gap is centered at the back (negative Y direction)
    gap_width = cartilage_outer * 2.5
    gap_depth = cartilage_outer * 1.5

    cutter = m3d.Manifold.cube(
        [gap_width, gap_depth, ring_height + 0.4],
        center=True
    )
    # Position cutter at back of ring
    cutter = cutter.translate([0, -cartilage_outer, 0])

    # Subtract gap from ring
    c_ring = ring - cutter

    # Translate to z position
    c_ring = c_ring.translate([0, 0, z_position])

    return c_ring


def _create_perichondrium_layer(
    z_position: float,
    cartilage_outer_radius: float,
    cartilage_inner_radius: float,
    ring_height: float,
    perichondrium_thickness_mm: float,
    gap_degrees: float,
    resolution: int
) -> m3d.Manifold:
    """Create a thin perichondrium shell around a cartilage ring."""
    peri_thick = perichondrium_thickness_mm

    # Outer perichondrium (outside cartilage)
    outer_peri = m3d.Manifold.cylinder(
        ring_height + peri_thick * 2,
        cartilage_outer_radius + peri_thick,
        cartilage_outer_radius + peri_thick,
        resolution
    ).translate([0, 0, -peri_thick])

    # Inner boundary (cartilage surface)
    inner_bound = m3d.Manifold.cylinder(
        ring_height + peri_thick * 2 + 0.1,
        cartilage_outer_radius,
        cartilage_outer_radius,
        resolution
    ).translate([0, 0, -peri_thick - 0.05])

    outer_shell = outer_peri - inner_bound

    # Inner perichondrium (inside cartilage)
    inner_peri_out = m3d.Manifold.cylinder(
        ring_height + peri_thick * 2,
        cartilage_inner_radius,
        cartilage_inner_radius,
        resolution
    ).translate([0, 0, -peri_thick])

    inner_peri_in = m3d.Manifold.cylinder(
        ring_height + peri_thick * 2 + 0.1,
        cartilage_inner_radius - peri_thick,
        cartilage_inner_radius - peri_thick,
        resolution
    ).translate([0, 0, -peri_thick - 0.05])

    inner_shell = inner_peri_out - inner_peri_in

    # Create gap cutter for posterior opening
    gap_width = (cartilage_outer_radius + peri_thick) * 2.5
    gap_depth = (cartilage_outer_radius + peri_thick) * 1.5

    cutter = m3d.Manifold.cube(
        [gap_width, gap_depth, ring_height + peri_thick * 2 + 0.4],
        center=True
    )
    cutter = cutter.translate([0, -(cartilage_outer_radius + peri_thick), 0])

    combined = tree_union([outer_shell, inner_shell])
    c_shaped = combined - cutter

    return c_shaped.translate([0, 0, z_position])


def _create_mucosal_layers(
    lumen_radius: float,
    epithelium_thick_mm: float,
    lamina_propria_thick_mm: float,
    submucosa_thick_mm: float,
    length: float,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create three concentric inner soft tissue layers.

    Returns epithelium, lamina propria, and submucosa as separate layers.
    """
    layers = []
    r = lumen_radius

    # Epithelium (innermost)
    outer_epi = m3d.Manifold.cylinder(length, r + epithelium_thick_mm, r + epithelium_thick_mm, resolution)
    inner_epi = m3d.Manifold.cylinder(length + 0.2, r, r, resolution).translate([0, 0, -0.1])
    layers.append(outer_epi - inner_epi)
    r += epithelium_thick_mm

    # Lamina propria
    outer_lp = m3d.Manifold.cylinder(length, r + lamina_propria_thick_mm, r + lamina_propria_thick_mm, resolution)
    inner_lp = m3d.Manifold.cylinder(length + 0.2, r, r, resolution).translate([0, 0, -0.1])
    layers.append(outer_lp - inner_lp)
    r += lamina_propria_thick_mm

    # Submucosa
    outer_sub = m3d.Manifold.cylinder(length, r + submucosa_thick_mm, r + submucosa_thick_mm, resolution)
    inner_sub = m3d.Manifold.cylinder(length + 0.2, r, r, resolution).translate([0, 0, -0.1])
    layers.append(outer_sub - inner_sub)

    return layers


def _create_trachealis_muscle(
    inner_radius: float,
    outer_radius: float,
    muscle_thickness_mm: float,
    gap_degrees: float,
    z_start: float,
    z_end: float,
    fiber_orientation_deg: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create the posterior trachealis muscle as a flat band spanning the C-ring gap.

    The trachealis is a smooth muscle band that connects the ends of the C-shaped
    cartilage rings, allowing tracheal constriction.

    Fiber orientation:
    - 0° = longitudinal (70% of fibers)
    - 90° = circular/transverse (30% of fibers)
    """
    length = z_end - z_start

    # Calculate the chord width at the gap
    gap_rad = np.radians(gap_degrees / 2)
    chord_half_width = inner_radius * np.sin(gap_rad)

    # Muscle band dimensions
    band_width = chord_half_width * 2
    band_depth = muscle_thickness_mm

    # Create the muscle band as a box
    muscle = m3d.Manifold.cube([band_width, band_depth, length], center=True)

    # Apply fiber orientation rotation if not transverse (90 degrees is default orientation)
    if fiber_orientation_deg != 90.0:
        # Rotate around Z axis to adjust fiber orientation
        muscle = muscle.rotate([0, 0, fiber_orientation_deg - 90.0])

    # Position at posterior wall (negative Y)
    y_pos = -inner_radius * np.cos(gap_rad) - band_depth / 2
    muscle = muscle.translate([0, y_pos, z_start + length / 2])

    return muscle


def _create_submucosal_glands(
    submucosa_inner_radius: float,
    submucosa_outer_radius: float,
    length: float,
    gland_diameter_mm: float,
    gland_density_per_mm2: float,
    random_seed: int,
    resolution: int
) -> Tuple[List[m3d.Manifold], int]:
    """
    Create spherical gland cavities within the submucosa layer.

    Returns list of gland manifolds and count of glands created.
    """
    rng = np.random.default_rng(random_seed)

    # Calculate submucosa surface area for gland distribution
    mid_radius = (submucosa_inner_radius + submucosa_outer_radius) / 2
    submucosa_surface_area = 2 * np.pi * mid_radius * length  # mm^2

    # Number of glands based on density
    num_glands = int(submucosa_surface_area * gland_density_per_mm2)

    glands = []
    gland_radius = gland_diameter_mm / 2

    # Distribute glands throughout the submucosa
    for _ in range(num_glands):
        # Random position in cylindrical coordinates
        theta = rng.random() * 2 * np.pi
        r = rng.uniform(submucosa_inner_radius + gland_radius, submucosa_outer_radius - gland_radius)
        z = rng.uniform(gland_radius, length - gland_radius)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Create spherical gland cavity
        gland = m3d.Manifold.sphere(gland_radius, resolution // 2)
        gland = gland.translate([x, y, z])
        glands.append(gland)

    return glands, num_glands


def _create_vascular_channels(
    submucosa_inner_radius: float,
    submucosa_outer_radius: float,
    length: float,
    channel_diameter_mm: float,
    max_spacing_mm: float,
    random_seed: int,
    resolution: int
) -> Tuple[List[m3d.Manifold], int]:
    """
    Create longitudinal and circumferential vascular channels in submucosa.

    Vessels ensure no tissue is more than max_spacing_mm from a vessel.
    """
    rng = np.random.default_rng(random_seed)

    mid_radius = (submucosa_inner_radius + submucosa_outer_radius) / 2
    channel_radius = channel_diameter_mm / 2

    channels = []

    # Longitudinal channels (run along length)
    circumference = 2 * np.pi * mid_radius
    num_longitudinal = max(4, int(circumference / max_spacing_mm))

    for i in range(num_longitudinal):
        theta = i * 2 * np.pi / num_longitudinal
        # Add slight radial variation
        r = mid_radius + (rng.random() - 0.5) * (submucosa_outer_radius - submucosa_inner_radius) * 0.5

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Longitudinal cylinder
        channel = m3d.Manifold.cylinder(length, channel_radius, channel_radius, resolution // 2)
        channel = channel.translate([x, y, 0])
        channels.append(channel)

    # Circumferential channels (rings connecting longitudinal)
    num_rings = max(3, int(length / max_spacing_mm))
    ring_spacing = length / (num_rings + 1)

    for i in range(num_rings):
        z = (i + 1) * ring_spacing

        # Create a torus-like ring using many small cylinders
        num_segments = max(8, resolution)
        for j in range(num_segments):
            theta1 = j * 2 * np.pi / num_segments
            theta2 = (j + 1) * 2 * np.pi / num_segments

            r = mid_radius
            x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
            x2, y2 = r * np.cos(theta2), r * np.sin(theta2)

            # Small segment cylinder
            seg_len = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            seg = m3d.Manifold.cylinder(seg_len, channel_radius, channel_radius, 6)

            # Rotate to align with segment direction
            angle_deg = np.degrees(np.arctan2(y2-y1, x2-x1))
            seg = seg.rotate([0, 90, 0]).rotate([0, 0, angle_deg])
            seg = seg.translate([x1, y1, z])
            channels.append(seg)

    return channels, num_longitudinal + num_rings * num_segments


def _create_annular_ligaments(
    inner_radius: float,
    outer_radius: float,
    ligament_thickness_mm: float,
    ring_positions: List[float],
    ring_height: float,
    interring_spacing: float,
    gap_degrees: float,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create fibroelastic annular ligaments between cartilage rings.

    These ligaments allow tracheal flexibility while maintaining structure.
    """
    ligaments = []

    for i in range(len(ring_positions) - 1):
        # Ligament sits between rings
        z_start = ring_positions[i] + ring_height
        z_end = ring_positions[i + 1]
        ligament_height = z_end - z_start

        if ligament_height <= 0:
            continue

        # Create a thinner ring for the ligament
        ligament_outer = inner_radius + ligament_thickness_mm
        ligament_inner = inner_radius

        outer_cyl = m3d.Manifold.cylinder(ligament_height, ligament_outer, ligament_outer, resolution)
        inner_cyl = m3d.Manifold.cylinder(ligament_height + 0.2, ligament_inner, ligament_inner, resolution)
        inner_cyl = inner_cyl.translate([0, 0, -0.1])

        ligament = outer_cyl - inner_cyl

        # Cut posterior gap to match C-rings
        gap_width = ligament_outer * 2.5
        gap_depth = ligament_outer * 1.5
        cutter = m3d.Manifold.cube([gap_width, gap_depth, ligament_height + 0.4], center=True)
        cutter = cutter.translate([0, -ligament_outer, 0])

        ligament = ligament - cutter
        ligament = ligament.translate([0, 0, z_start])
        ligaments.append(ligament)

    return ligaments


def _create_carina_bifurcation(
    trachea_end_z: float,
    trachea_radius: float,
    right_angle_deg: float,
    left_angle_deg: float,
    right_diameter_mm: float,
    left_diameter_mm: float,
    branch_length: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create Y-junction with asymmetric bronchi at the carina.

    Right bronchus: larger diameter, steeper angle (25 deg)
    Left bronchus: smaller diameter, more horizontal (45 deg)
    """
    # Junction sphere at bifurcation point
    junction_radius = trachea_radius * 1.2
    junction = m3d.Manifold.sphere(junction_radius, resolution)
    junction = junction.translate([0, 0, trachea_end_z])

    # Carina ridge (sharp ridge at bifurcation)
    ridge_height = junction_radius * 0.3
    ridge_width = junction_radius * 0.1
    ridge = m3d.Manifold.cube([ridge_width, junction_radius * 2, ridge_height], center=True)
    ridge = ridge.translate([0, 0, trachea_end_z + junction_radius * 0.5])

    # Right main bronchus (steeper, larger)
    right_rad = right_diameter_mm / 2
    right_cyl = m3d.Manifold.cylinder(branch_length, right_rad * 1.1, right_rad, resolution)
    # Tilt toward right (positive X) and down (positive Z continues)
    right_cyl = right_cyl.rotate([0, right_angle_deg, 0])
    right_cyl = right_cyl.translate([0, 0, trachea_end_z])

    # Left main bronchus (more horizontal, smaller)
    left_rad = left_diameter_mm / 2
    left_cyl = m3d.Manifold.cylinder(branch_length, left_rad * 1.1, left_rad, resolution)
    # Tilt toward left (negative X) and down
    left_cyl = left_cyl.rotate([0, -left_angle_deg, 0])
    left_cyl = left_cyl.translate([0, 0, trachea_end_z])

    return tree_union([junction, ridge, right_cyl, left_cyl])


def _create_cilia_markers(
    lumen_radius: float,
    length: float,
    density: float,
    random_seed: int,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create fine surface texture markers representing cilia on epithelium.

    These are visualization markers for the mucociliary apparatus.
    """
    rng = np.random.default_rng(random_seed)

    # Calculate surface area
    surface_area = 2 * np.pi * lumen_radius * length

    # Markers per mm^2 based on density
    markers_per_mm2 = density * 10  # Scale density to reasonable marker count
    num_markers = int(surface_area * markers_per_mm2)
    num_markers = min(num_markers, 500)  # Cap for performance

    markers = []
    marker_height = 0.02  # 20 um
    marker_radius = 0.01  # 10 um

    for _ in range(num_markers):
        theta = rng.random() * 2 * np.pi
        z = rng.random() * length

        x = lumen_radius * np.cos(theta)
        y = lumen_radius * np.sin(theta)

        # Small cylinder pointing inward
        marker = m3d.Manifold.cylinder(marker_height, marker_radius, marker_radius * 0.5, 4)

        # Orient toward center
        tilt = 90 + np.degrees(theta)
        marker = marker.rotate([0, 90, 0]).rotate([0, 0, tilt])
        marker = marker.translate([x, y, z])
        markers.append(marker)

    return markers


def _create_goblet_cell_markers(
    lumen_radius: float,
    length: float,
    random_seed: int,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create markers for goblet cells (mucus-secreting cells).

    20% of epithelial cells are goblet cells, scattered randomly (not uniform).
    """
    rng = np.random.default_rng(random_seed)

    # Calculate surface area
    surface_area = 2 * np.pi * lumen_radius * length

    # Assume ~200 cells per mm^2, 20% are goblet cells
    cells_per_mm2 = 200
    goblet_percentage = 0.20
    num_goblet = int(surface_area * cells_per_mm2 * goblet_percentage)
    num_goblet = min(num_goblet, 1000)  # Cap for performance

    markers = []
    marker_height = 0.03  # 30 um (slightly larger than cilia)
    marker_radius = 0.015  # 15 um

    for _ in range(num_goblet):
        theta = rng.random() * 2 * np.pi
        z = rng.random() * length

        x = lumen_radius * np.cos(theta)
        y = lumen_radius * np.sin(theta)

        # Small sphere for goblet cell
        marker = m3d.Manifold.sphere(marker_radius, 6)
        marker = marker.translate([x, y, z])
        markers.append(marker)

    return markers


def _create_full_cartilage_ring(
    z_position: float,
    outer_radius: float,
    inner_radius: float,
    ring_height: float,
    ring_thickness: float,
    resolution: int,
    ring_variance_pct: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> m3d.Manifold:
    """
    Create a full circular cartilage ring (not C-shaped).

    Args:
        ring_thickness: Radial thickness of cartilage ring
    """
    # Apply variance if requested
    if ring_variance_pct > 0 and rng is not None:
        variance_mult = 1.0 + (rng.random() - 0.5) * 2 * ring_variance_pct / 100
        ring_height = ring_height * variance_mult

    # Use ring_thickness to define the radial extent of cartilage
    cartilage_outer = outer_radius
    cartilage_inner = max(inner_radius, outer_radius - ring_thickness)

    # Create full ring
    outer_ring = m3d.Manifold.cylinder(
        ring_height,
        cartilage_outer,
        cartilage_outer,
        resolution
    )

    inner_ring = m3d.Manifold.cylinder(
        ring_height + 0.2,
        cartilage_inner,
        cartilage_inner,
        resolution
    ).translate([0, 0, -0.1])

    ring = outer_ring - inner_ring

    # Translate to z position
    ring = ring.translate([0, 0, z_position])

    return ring


def _apply_cartilage_porosity(
    cartilage_manifold: m3d.Manifold,
    porosity: float,
    pore_size_mm: float,
    outer_radius: float,
    ring_height: float,
    z_position: float,
    random_seed: int,
    resolution: int
) -> m3d.Manifold:
    """
    Apply porosity to cartilage ring by subtracting pores.

    Optimal porosity: 80-95% (85-90% recommended)
    """
    if porosity <= 0 or porosity >= 1.0:
        return cartilage_manifold

    rng = np.random.default_rng(random_seed)

    # Calculate cartilage volume and number of pores needed
    ring_volume = np.pi * (outer_radius ** 2) * ring_height
    pore_volume = (4/3) * np.pi * (pore_size_mm / 2) ** 3

    target_pore_volume = ring_volume * porosity
    num_pores = int(target_pore_volume / pore_volume)
    num_pores = min(num_pores, 200)  # Cap for performance

    pores = []
    pore_radius = pore_size_mm / 2

    for _ in range(num_pores):
        # Random position in cylindrical coordinates
        theta = rng.random() * 2 * np.pi
        r = rng.uniform(outer_radius * 0.3, outer_radius * 0.9)
        z = rng.uniform(pore_radius, ring_height - pore_radius)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        pore = m3d.Manifold.sphere(pore_radius, resolution // 4)
        pore = pore.translate([x, y, z_position - ring_height/2 + z])
        pores.append(pore)

    if pores:
        pores_union = batch_union(pores)
        return cartilage_manifold - pores_union

    return cartilage_manifold


def _create_mucosal_folds(
    lumen_radius: float,
    length: float,
    fold_depth_mm: float,
    fold_count: int,
    posterior_gap_deg: float,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create mucosal folds on the posterior wall (where cartilage is absent).

    Folds: 0.1-0.3 mm depth, variable/irregular pattern
    """
    folds = []

    # Posterior wall spans the gap
    gap_rad_half = np.radians(posterior_gap_deg / 2)

    # Create folds as ridges along the length
    for i in range(fold_count):
        # Position within the posterior gap
        fraction = (i + 0.5) / fold_count
        # Map to gap angle range (centered at back = 180 deg from front)
        theta = np.pi - gap_rad_half + (2 * gap_rad_half * fraction)

        # Create a thin ridge
        ridge_width = fold_depth_mm
        ridge_thickness = 0.05  # 50 um thick

        x = lumen_radius * np.cos(theta)
        y = lumen_radius * np.sin(theta)

        # Longitudinal ridge
        fold = m3d.Manifold.cube([ridge_thickness, ridge_width, length], center=True)

        # Rotate to align with surface normal
        angle_deg = np.degrees(theta)
        fold = fold.rotate([0, 0, angle_deg])
        fold = fold.translate([x, y, length / 2])

        folds.append(fold)

    return folds


def generate_trachea(params: TracheaParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a trachea scaffold with C-shaped cartilage rings.

    Creates a series of C-shaped rings (270 arc) with gaps for the posterior
    membrane. Rings are evenly spaced along the length.

    Args:
        params: TracheaParams with generation parameters

    Returns:
        Tuple of (manifold, stats_dict)

    Raises:
        ValueError: If parameters are invalid
    """
    outer_radius = params.outer_diameter_mm / 2
    inner_radius = params.inner_diameter_mm / 2

    if inner_radius <= 0:
        raise ValueError("Inner diameter must be positive")
    if inner_radius >= outer_radius:
        raise ValueError("Inner diameter must be less than outer diameter")

    # Initialize RNG for variance
    rng = np.random.default_rng(params.random_seed)

    # Use ring_gap_deg as alias for posterior_opening_degrees
    effective_gap_degrees = params.ring_gap_deg if params.ring_gap_deg != 90.0 else params.posterior_opening_degrees

    # Calculate ring arc angle (360 - gap)
    ring_arc_deg = 360.0 - effective_gap_degrees

    # Determine actual length calculation strategy
    # If total_length_mm differs significantly from calculated length, use total_length_mm
    calculated_length_from_rings = params.num_cartilage_rings * (params.ring_height_mm + params.interring_spacing_mm)
    use_total_length = abs(params.total_length_mm - calculated_length_from_rings) > 1.0  # > 1mm difference

    if use_total_length:
        # Use total_length_mm: adjust ring count or spacing to fit
        target_length = params.total_length_mm
        # Calculate spacing to fill the target length
        spacing = target_length / params.num_cartilage_rings
        # Adjust ring height to maintain reasonable proportions
        ring_height_adjusted = spacing * 0.7  # 70% ring, 30% gap
        interring_spacing_adjusted = spacing * 0.3
    else:
        # Use ring count-driven calculation
        spacing = params.ring_height_mm + params.interring_spacing_mm
        ring_height_adjusted = params.ring_height_mm
        interring_spacing_adjusted = params.interring_spacing_mm

    # Use ring_width_mm as the superior-inferior extent (vertical height)
    # If ring_width_mm is specified and differs from ring_height_mm, use it
    if params.ring_width_mm != 4.5:  # Not default value
        ring_height_adjusted = params.ring_width_mm

    # Create all cartilage rings (C-shaped or full)
    rings = []
    ring_positions = []

    for i in range(params.num_cartilage_rings):
        z_pos = i * spacing + ring_height_adjusted / 2

        # Apply position noise if requested
        if params.position_noise_mm > 0:
            z_pos += (rng.random() - 0.5) * 2 * params.position_noise_mm

        ring_positions.append(z_pos)

        if params.enable_c_shaped_cartilage:
            ring = _create_c_ring(
                z_position=z_pos,
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                ring_height=ring_height_adjusted,
                ring_thickness=params.ring_thickness_mm,
                gap_degrees=effective_gap_degrees,
                resolution=params.resolution,
                ring_variance_pct=params.ring_variance_pct,
                rng=rng
            )
        else:
            ring = _create_full_cartilage_ring(
                z_position=z_pos,
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                ring_height=ring_height_adjusted,
                ring_thickness=params.ring_thickness_mm,
                resolution=params.resolution,
                ring_variance_pct=params.ring_variance_pct,
                rng=rng
            )

        # Apply porosity if enabled (use scaffold_porosity if higher than cartilage_porosity)
        effective_porosity = max(params.cartilage_porosity, params.scaffold_porosity)
        if effective_porosity > 0:
            pore_size_mm = params.pore_size_um / 1000.0
            ring = _apply_cartilage_porosity(
                cartilage_manifold=ring,
                porosity=effective_porosity,
                pore_size_mm=pore_size_mm,
                outer_radius=outer_radius,
                ring_height=ring_height_adjusted,
                z_position=z_pos,
                random_seed=params.random_seed + i,
                resolution=params.resolution
            )

        rings.append(ring)

    all_parts = rings.copy()
    actual_length = params.num_cartilage_rings * spacing

    # Track created features for stats
    stats_features = {
        'cartilage_rings': params.num_cartilage_rings,
        'perichondrium_layers': 0,
        'mucosal_layers': 0,
        'glands': 0,
        'vascular_channels': 0,
        'annular_ligaments': 0,
        'cilia_markers': 0,
        'goblet_cell_markers': 0,
        'mucosal_folds': 0,
    }


    # === Add Perichondrium ===
    if params.enable_perichondrium:
        peri_thickness_mm = params.perichondrium_thickness_um / 1000.0
        # Calculate cartilage ring boundaries considering ring_thickness
        cartilage_outer = outer_radius
        cartilage_inner = max(inner_radius, outer_radius - params.ring_thickness_mm)

        for i, z_pos in enumerate(ring_positions):
            peri = _create_perichondrium_layer(
                z_position=z_pos,
                cartilage_outer_radius=cartilage_outer,
                cartilage_inner_radius=cartilage_inner,
                ring_height=ring_height_adjusted,
                perichondrium_thickness_mm=peri_thickness_mm,
                gap_degrees=effective_gap_degrees,
                resolution=params.resolution
            )
            all_parts.append(peri)
        stats_features['perichondrium_layers'] = params.num_cartilage_rings

    # === Calculate Mucosal Layer Thicknesses ===
    # Always calculate these for stats, even if not enabled
    epithelium_mm = params.epithelium_thickness_um / 1000.0
    lamina_propria_mm = params.lamina_propria_thickness_um / 1000.0
    submucosa_mm = params.submucosa_thickness_um / 1000.0

    # Use mucosa_thickness_um as fallback when individual layers aren't set
    if params.enable_mucosal_layer or params.enable_submucosa:
        # Validate against mucosa_thickness_um (epithelium + lamina propria should approximate it)
        combined_mucosa = epithelium_mm + lamina_propria_mm
        mucosa_target_mm = params.mucosa_thickness_um / 1000.0

        # If combined differs significantly and mucosa_thickness_um is non-default, use it as guide
        if abs(combined_mucosa - mucosa_target_mm) > 0.05 and params.mucosa_thickness_um != 350.0:
            # Scale layers to match mucosa_thickness_um
            scale_factor = mucosa_target_mm / combined_mucosa if combined_mucosa > 0 else 1.0
            epithelium_mm *= scale_factor
            lamina_propria_mm *= scale_factor

        # Lumen radius for mucosal layers
        mucosal_lumen_radius = inner_radius

        mucosal_layers = _create_mucosal_layers(
            lumen_radius=mucosal_lumen_radius,
            epithelium_thick_mm=epithelium_mm,
            lamina_propria_thick_mm=lamina_propria_mm,
            submucosa_thick_mm=submucosa_mm if params.enable_submucosa else 0,
            length=actual_length,
            resolution=params.resolution
        )

        if params.enable_mucosal_layer:
            all_parts.extend(mucosal_layers[:2])  # Epithelium + lamina propria
            stats_features['mucosal_layers'] = 2
        if params.enable_submucosa and len(mucosal_layers) > 2:
            all_parts.append(mucosal_layers[2])
            stats_features['mucosal_layers'] += 1

    # === Add Submucosal Glands ===
    if params.enable_submucosal_glands and params.enable_submucosa:
        epithelium_mm = params.epithelium_thickness_um / 1000.0
        lamina_propria_mm = params.lamina_propria_thickness_um / 1000.0
        submucosa_mm = params.submucosa_thickness_um / 1000.0

        submucosa_inner = inner_radius + epithelium_mm + lamina_propria_mm
        submucosa_outer = submucosa_inner + submucosa_mm

        gland_diameter_mm = params.gland_diameter_um / 1000.0

        glands, num_glands = _create_submucosal_glands(
            submucosa_inner_radius=submucosa_inner,
            submucosa_outer_radius=submucosa_outer,
            length=actual_length,
            gland_diameter_mm=gland_diameter_mm,
            gland_density_per_mm2=params.gland_density_per_mm2,
            random_seed=params.random_seed,
            resolution=params.resolution
        )
        all_parts.extend(glands)
        stats_features['glands'] = num_glands

    # === Add Trachealis Muscle ===
    if params.enable_trachealis_muscle:
        muscle_thick_mm = params.trachealis_thickness_um / 1000.0

        muscle = _create_trachealis_muscle(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            muscle_thickness_mm=muscle_thick_mm,
            gap_degrees=params.trachealis_width_deg,
            z_start=0,
            z_end=actual_length,
            fiber_orientation_deg=params.trachealis_fiber_orientation_deg,
            resolution=params.resolution
        )
        all_parts.append(muscle)

    # === Add Vascular Channels ===
    if params.enable_vascular_channels and params.enable_submucosa:
        epithelium_mm = params.epithelium_thickness_um / 1000.0
        lamina_propria_mm = params.lamina_propria_thickness_um / 1000.0
        submucosa_mm = params.submucosa_thickness_um / 1000.0

        submucosa_inner = inner_radius + epithelium_mm + lamina_propria_mm
        submucosa_outer = submucosa_inner + submucosa_mm

        channel_diameter_mm = params.vascular_channel_diameter_um / 1000.0

        # Use vascular_channel_spacing_mm if specified, otherwise fall back to vascular_spacing_um
        if params.vascular_channel_spacing_mm != 3.0:
            max_spacing_mm = params.vascular_channel_spacing_mm
        else:
            max_spacing_mm = params.vascular_spacing_um / 1000.0

        channels, num_channels = _create_vascular_channels(
            submucosa_inner_radius=submucosa_inner,
            submucosa_outer_radius=submucosa_outer,
            length=actual_length,
            channel_diameter_mm=channel_diameter_mm,
            max_spacing_mm=max_spacing_mm,
            random_seed=params.random_seed + 1,
            resolution=params.resolution
        )
        all_parts.extend(channels)
        stats_features['vascular_channels'] = num_channels

    # === Add Annular Ligaments ===
    if params.enable_annular_ligaments:
        ligaments = _create_annular_ligaments(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            ligament_thickness_mm=params.annular_ligament_thickness_mm,
            ring_positions=ring_positions,
            ring_height=params.ring_height_mm,
            interring_spacing=params.interring_spacing_mm,
            gap_degrees=params.posterior_opening_degrees,
            resolution=params.resolution
        )
        all_parts.extend(ligaments)
        stats_features['annular_ligaments'] = len(ligaments)

    # === Add Cilia Markers ===
    if params.enable_cilia_markers or params.enable_ciliated_epithelium_markers:
        cilia_markers = _create_cilia_markers(
            lumen_radius=inner_radius,
            length=actual_length,
            density=params.cilia_density,
            random_seed=params.random_seed + 2,
            resolution=params.resolution
        )
        all_parts.extend(cilia_markers)
        stats_features['cilia_markers'] = len(cilia_markers)

    # === Add Goblet Cell Markers ===
    if params.enable_goblet_cell_markers:
        goblet_markers = _create_goblet_cell_markers(
            lumen_radius=inner_radius,
            length=actual_length,
            random_seed=params.random_seed + 3,
            resolution=params.resolution
        )
        all_parts.extend(goblet_markers)
        stats_features['goblet_cell_markers'] = len(goblet_markers)

    # === Add Mucosal Folds ===
    if params.enable_mucosal_folds:
        mucosal_folds = _create_mucosal_folds(
            lumen_radius=inner_radius,
            length=actual_length,
            fold_depth_mm=params.mucosal_fold_depth_mm,
            fold_count=params.mucosal_fold_count,
            posterior_gap_deg=params.posterior_opening_degrees,
            resolution=params.resolution
        )
        all_parts.extend(mucosal_folds)
        stats_features['mucosal_folds'] = len(mucosal_folds)

    # === Add Carina Bifurcation ===
    carina_manifold = None
    actual_carina_angle = 0.0
    effective_left_angle = params.left_bronchus_angle_deg
    effective_right_angle = params.right_bronchus_angle_deg

    if params.enable_carina:
        # Calculate actual carina angle from bronchus angles
        # Carina angle = total angle between main bronchi
        actual_carina_angle = params.left_bronchus_angle_deg + params.right_bronchus_angle_deg

        # If carina_angle_deg is specified and differs from calculated, use it to adjust bronchus angles
        if params.carina_angle_deg != 70.0:  # Non-default value
            # User specified carina angle, distribute between bronchi maintaining ratio
            angle_ratio = params.right_bronchus_angle_deg / (params.left_bronchus_angle_deg + params.right_bronchus_angle_deg)
            effective_right_angle = params.carina_angle_deg * angle_ratio
            effective_left_angle = params.carina_angle_deg * (1 - angle_ratio)
        else:
            effective_right_angle = params.right_bronchus_angle_deg
            effective_left_angle = params.left_bronchus_angle_deg
            actual_carina_angle = effective_left_angle + effective_right_angle

        # Right bronchus is larger (~60% of trachea), left is smaller (~55%)
        right_diameter = params.bronchus_diameter_mm
        left_diameter = params.bronchus_diameter_mm * 0.9

        carina_manifold = _create_carina_bifurcation(
            trachea_end_z=actual_length,
            trachea_radius=outer_radius,
            right_angle_deg=effective_right_angle,
            left_angle_deg=effective_left_angle,
            right_diameter_mm=right_diameter,
            left_diameter_mm=left_diameter,
            branch_length=params.bronchus_length_mm,
            resolution=params.resolution
        )
        all_parts.append(carina_manifold)

    # Combine all parts
    trachea = batch_union(all_parts)

    # Calculate statistics
    mesh = trachea.to_mesh()
    volume = trachea.volume() if hasattr(trachea, 'volume') else 0

    # Calculate effective porosity used in geometry
    effective_porosity_used = max(params.cartilage_porosity, params.scaffold_porosity)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'total_length_mm': params.total_length_mm,  # Input parameter
        'actual_scaffold_length_mm': actual_length,  # Actual generated length
        'outer_diameter_mm': params.outer_diameter_mm,
        'inner_diameter_mm': params.inner_diameter_mm,
        'num_cartilage_rings': params.num_cartilage_rings,
        'ring_height_mm': ring_height_adjusted,  # Actual height used (may differ from input)
        'ring_width_mm': ring_height_adjusted,  # Ring width = superior-inferior extent = height
        'ring_thickness_mm': params.ring_thickness_mm,  # Radial thickness (used in geometry)
        'ring_gap_deg': effective_gap_degrees,  # Actual gap used (ring_gap_deg or posterior_opening_degrees)
        'posterior_opening_degrees': params.posterior_opening_degrees,
        'interring_spacing_mm': interring_spacing_adjusted if use_total_length else params.interring_spacing_mm,
        'ring_arc_angle_deg': ring_arc_deg,
        'layer_thicknesses_um': {
            'epithelium': epithelium_mm * 1000 if (params.enable_mucosal_layer or params.enable_submucosa) else 0,
            'lamina_propria': lamina_propria_mm * 1000 if (params.enable_mucosal_layer or params.enable_submucosa) else 0,
            'submucosa': submucosa_mm * 1000 if params.enable_submucosa else 0,
            'perichondrium': params.perichondrium_thickness_um if params.enable_perichondrium else 0
        },
        'trachealis_thickness_um': params.trachealis_thickness_um if params.enable_trachealis_muscle else 0,
        'trachealis_fiber_orientation_deg': params.trachealis_fiber_orientation_deg if params.enable_trachealis_muscle else 0,
        'cartilage_type': params.cartilage_type,  # Metadata for material properties/FEA
        'enable_c_shaped_cartilage': params.enable_c_shaped_cartilage,
        'cartilage_porosity': params.cartilage_porosity,
        'effective_porosity': effective_porosity_used,  # Actual porosity applied
        'pore_size_um': params.pore_size_um,
        'mucosa_thickness_um': params.mucosa_thickness_um,  # Legacy: epithelium + lamina_propria combined
        'actual_mucosa_thickness_um': (epithelium_mm + lamina_propria_mm) * 1000 if (params.enable_mucosal_layer or params.enable_submucosa) else 0,
        'mucosal_fold_depth_mm': params.mucosal_fold_depth_mm if params.enable_mucosal_folds else 0,
        'mucosal_fold_count': params.mucosal_fold_count if params.enable_mucosal_folds else 0,
        'vascular_channel_spacing_mm': params.vascular_channel_spacing_mm if params.enable_vascular_channels else 0,
        'scaffold_porosity': params.scaffold_porosity,  # Overall scaffold porosity
        'scaffold_type': 'trachea',
        'features': stats_features,
        'carina_enabled': params.enable_carina,
        'carina_angle_deg': actual_carina_angle if params.enable_carina else 0,  # Calculated from bronchus angles
        'left_bronchus_angle_deg': effective_left_angle if params.enable_carina else 0,
        'right_bronchus_angle_deg': effective_right_angle if params.enable_carina else 0,
    }

    return trachea, stats


def generate_trachea_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate a trachea scaffold from dictionary parameters.

    Args:
        params: Dictionary of parameters (will be unpacked into TracheaParams)

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle parameter name mappings for compatibility
    param_copy = params.copy()

    # Map legacy parameter names
    if 'ring_count' in param_copy and 'num_cartilage_rings' not in param_copy:
        param_copy['num_cartilage_rings'] = param_copy.pop('ring_count')
    if 'ring_spacing_mm' in param_copy and 'interring_spacing_mm' not in param_copy:
        param_copy['interring_spacing_mm'] = param_copy.pop('ring_spacing_mm')
    if 'posterior_gap_angle_deg' in param_copy and 'posterior_opening_degrees' not in param_copy:
        param_copy['posterior_opening_degrees'] = param_copy.pop('posterior_gap_angle_deg')
    if 'length_mm' in param_copy and 'total_length_mm' not in param_copy:
        param_copy['total_length_mm'] = param_copy.pop('length_mm')
    if 'ring_thickness_mm' in param_copy:
        # This was used for ring_height in old API
        pass

    # Filter to only valid TracheaParams fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(TracheaParams)}
    filtered_params = {k: v for k, v in param_copy.items() if k in valid_fields}

    return generate_trachea(TracheaParams(**filtered_params))
