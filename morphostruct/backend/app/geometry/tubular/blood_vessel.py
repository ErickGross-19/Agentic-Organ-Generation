"""
Blood vessel scaffold generator.

Generates tri-layer vascular walls (intima, media, adventitia) with optional bifurcation.
Includes biologically realistic parameters for vessel wall microstructure.
"""

import manifold3d as m3d
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BloodVesselParams:
    """
    Parameters for blood vessel scaffold generation.

    Based on human vascular anatomy with tri-layer wall structure:
    - Intima: endothelial cells on basement membrane
    - Media: smooth muscle cells with elastic laminae
    - Adventitia: collagen-rich connective tissue
    """

    # === Basic Geometry ===
    inner_diameter_mm: float = 3.0  # Lumen diameter (small artery ~3mm)
    wall_thickness_mm: float = 0.5  # Total wall thickness
    length_mm: float = 50.0  # Vessel segment length

    # === Layer Thicknesses (anatomically accurate) ===
    # These override layer_ratios if provided
    intima_thickness_um: float = 50.0  # ~50 um for healthy artery
    media_thickness_um: float = 400.0  # Major structural layer
    adventitia_thickness_um: float = 200.0  # Outer connective tissue

    # Legacy: layer ratios (used if layer thicknesses not matching wall)
    layer_ratios: list[float] = None  # [intima, media, adventitia]

    # === Elastic Laminae (Media Layer) ===
    elastic_lamina_thickness_um: float = 1.8  # Internal elastic lamina thickness
    external_elastic_lamina_thickness_um: float = 2.0  # External elastic lamina thickness
    num_elastic_laminae: int = 10  # Number of elastic sheets in media
    enable_elastic_laminae: bool = False  # Generate laminae structures
    fenestration_count: int = 55  # Number of fenestrations per lamina (40-70)
    fenestration_diameter_um: float = 2.5  # Fenestration hole diameter (2-3um)

    # === Smooth Muscle Cells (Media) ===
    smc_length_um: float = 175.0  # Smooth muscle cell length
    smc_orientation_angle_deg: float = 0.0  # Helical angle (0=circumferential)
    enable_smc_alignment: bool = False  # Generate aligned SMC structures
    smc_density_per_mm2: float = 500.0  # SMC markers per mm^2 in media

    # === Endothelial Features (Intima) ===
    enable_endothelial_texture: bool = False  # Inner surface texture
    endothelial_cell_size_um: float = 30.0  # Typical EC size (20-30um)
    endothelial_cell_density_per_mm2: float = 2500.0  # Cells/mm^2 (2000-3000)
    endothelial_bump_height_um: float = 3.0  # Height of cobblestone bumps

    # === Adventitial Features ===
    enable_vasa_vasorum: bool = False  # Small vessels in adventitia
    vasa_vasorum_diameter_um: float = 75.0  # Microvessels feeding wall (50-100um)
    vasa_vasorum_spacing_um: float = 500.0  # Distance between vasa vasorum
    vasa_vasorum_density_per_mm2: float = 30.0  # Density (20-40/mm^2)

    # === Bifurcation ===
    enable_bifurcation: bool = False  # Enable Y-bifurcation
    bifurcation_angle_deg: Optional[float] = None  # Y-bifurcation angle (36 deg typical)
    daughter_diameter_ratio: float = 0.794  # Murray's law: cube root of 0.5

    # === Porosity and Permeability ===
    scaffold_porosity: float = 0.5  # Target porosity (0-1)
    pore_size_um: float = 100.0  # For cell infiltration
    enable_radial_pores: bool = False  # Transmural pores
    enable_pore_network: bool = False  # Interconnected pore system

    # === Vessel Geometry Variations ===
    taper_ratio: float = 0.0  # Diameter reduction along length (0-0.3)
    vessel_taper: float = 0.0  # Alias for taper_ratio
    tortuosity_index: float = 0.0  # Waviness/curvature (0-1)
    vessel_tortuosity: float = 0.0  # Alias for tortuosity_index
    tortuosity_wavelength_mm: float = 5.0  # Wave period for tortuosity
    tortuosity_amplitude_ratio: float = 0.15  # Amplitude as fraction of wavelength (10-20%)

    # === Mechanical Properties (for FEA export) ===
    target_compliance_percent_per_100mmHg: float = 6.0  # Compliance target
    burst_pressure_mmHg: float = 2000.0  # Design specification

    # === Randomization ===
    position_noise: float = 0.0  # Random variation in structures
    random_seed: int = 42

    # === Resolution ===
    resolution: int = 16  # Cylinder segments

    def __post_init__(self):
        # Sync alias parameters
        if self.taper_ratio == 0 and self.vessel_taper != 0:
            self.taper_ratio = self.vessel_taper
        elif self.vessel_taper == 0 and self.taper_ratio != 0:
            self.vessel_taper = self.taper_ratio

        if self.tortuosity_index == 0 and self.vessel_tortuosity != 0:
            self.tortuosity_index = self.vessel_tortuosity
        elif self.vessel_tortuosity == 0 and self.tortuosity_index != 0:
            self.vessel_tortuosity = self.tortuosity_index

        # Enable bifurcation if angle is set
        if self.bifurcation_angle_deg is not None and self.bifurcation_angle_deg > 0:
            self.enable_bifurcation = True

        if self.layer_ratios is None:
            # Calculate ratios from explicit thicknesses
            total_um = self.intima_thickness_um + self.media_thickness_um + self.adventitia_thickness_um
            if total_um > 0:
                self.layer_ratios = [
                    self.intima_thickness_um / total_um,
                    self.media_thickness_um / total_um,
                    self.adventitia_thickness_um / total_um
                ]
            else:
                self.layer_ratios = [0.1, 0.5, 0.4]

        if len(self.layer_ratios) != 3:
            raise ValueError("layer_ratios must have exactly 3 values")

        # Normalize layer_ratios if they don't sum to 1
        ratio_sum = sum(self.layer_ratios)
        if not np.isclose(ratio_sum, 1.0):
            self.layer_ratios = [r / ratio_sum for r in self.layer_ratios]


def _make_elastic_lamina(
    radius: float,
    thickness_mm: float,
    z_start: float,
    z_end: float,
    num_fenestrations: int,
    fenestration_diameter_mm: float,
    resolution: int,
    rng: np.random.Generator,
    position_noise: float = 0.0
) -> m3d.Manifold:
    """
    Create an elastic lamina (IEL or EEL) with fenestrations.

    Elastic laminae are thin cylindrical sheets with small holes (fenestrations)
    that allow diffusion of nutrients and signaling molecules.

    Args:
        radius: Outer radius of the lamina in mm
        thickness_mm: Thickness of the lamina in mm
        z_start: Start position along vessel axis
        z_end: End position along vessel axis
        num_fenestrations: Number of holes in the lamina
        fenestration_diameter_mm: Diameter of each hole in mm
        resolution: Circular resolution for cylinders
        rng: Random number generator for fenestration placement
        position_noise: Random variation (0-1 scale, 0.05-0.15 typical)

    Returns:
        Manifold representing the lamina with fenestrations
    """
    height = z_end - z_start
    inner_radius = radius - thickness_mm

    # Create the base lamina as a thin cylindrical shell
    outer_cyl = m3d.Manifold.cylinder(height, radius, radius, resolution)
    inner_cyl = m3d.Manifold.cylinder(height + 0.01, inner_radius, inner_radius, resolution)
    lamina = outer_cyl - inner_cyl.translate([0, 0, -0.005])

    # Add fenestrations (small cylindrical holes)
    if num_fenestrations > 0 and fenestration_diameter_mm > 0:
        base_fenestration_radius = fenestration_diameter_mm / 2
        # Distribute fenestrations around and along the lamina
        for i in range(num_fenestrations):
            # Random angle and height for each fenestration
            angle = rng.uniform(0, 2 * np.pi)
            z_pos = rng.uniform(0.1 * height, 0.9 * height)

            # Position at the middle of the lamina wall
            mid_radius = (radius + inner_radius) / 2
            x = mid_radius * np.cos(angle)
            y = mid_radius * np.sin(angle)

            # Apply position noise to fenestration size (biomimetic variation)
            size_variation = 1.0 + rng.uniform(-1, 1) * position_noise * 0.3
            fenestration_radius = base_fenestration_radius * size_variation

            # Create a small cylinder for the fenestration, oriented radially
            fenestration = m3d.Manifold.cylinder(
                thickness_mm * 3,  # Long enough to cut through
                fenestration_radius,
                fenestration_radius,
                8  # Lower resolution for small holes
            )

            # Rotate to radial orientation (pointing outward from center)
            fenestration = fenestration.rotate([0, 90, np.degrees(angle)])
            fenestration = fenestration.translate([x, y, z_pos])

            lamina = lamina - fenestration

    return lamina.translate([0, 0, z_start])


def _make_smc_alignment_markers(
    inner_radius: float,
    outer_radius: float,
    z_start: float,
    z_end: float,
    orientation_angle_deg: float,
    density_per_mm2: float,
    smc_length_um: float,
    resolution: int,
    rng: np.random.Generator,
    position_noise: float = 0.0
) -> m3d.Manifold:
    """
    Create elongated ellipsoid markers showing smooth muscle cell alignment.

    SMCs in the media layer are oriented circumferentially (0 deg) or with
    a slight helical pitch (up to 30 deg).

    Args:
        inner_radius: Inner radius of media layer
        outer_radius: Outer radius of media layer
        z_start: Start position along vessel axis
        z_end: End position along vessel axis
        orientation_angle_deg: Helical pitch angle (0=circumferential)
        density_per_mm2: Number of markers per mm^2
        smc_length_um: Length of SMC markers
        resolution: Resolution for ellipsoids
        rng: Random number generator
        position_noise: Random variation (0-1 scale, 0.05-0.15 typical)

    Returns:
        Manifold with SMC alignment markers
    """
    height = z_end - z_start
    mid_radius = (inner_radius + outer_radius) / 2
    circumference = 2 * np.pi * mid_radius

    # Calculate surface area and number of markers
    surface_area_mm2 = circumference * height
    num_markers = int(surface_area_mm2 * density_per_mm2 / 100)  # Reduced for performance
    num_markers = min(num_markers, 200)  # Cap for performance

    if num_markers == 0:
        return m3d.Manifold()

    base_smc_length_mm = smc_length_um / 1000.0
    base_smc_width_mm = base_smc_length_mm / 5.0  # SMCs are elongated ~5:1 ratio

    markers = []
    for i in range(num_markers):
        # Random position in media layer
        angle = rng.uniform(0, 2 * np.pi)
        z_pos = z_start + rng.uniform(0.1, 0.9) * height
        r = rng.uniform(inner_radius + 0.01, outer_radius - 0.01)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Apply position noise to SMC size (biological variation)
        size_variation = 1.0 + rng.uniform(-1, 1) * position_noise * 0.25
        smc_length_mm = base_smc_length_mm * size_variation
        smc_width_mm = base_smc_width_mm * size_variation

        # Create elongated ellipsoid (approximated with scaled sphere)
        marker = m3d.Manifold.sphere(smc_width_mm / 2, 8)
        marker = marker.scale([smc_length_mm / smc_width_mm, 1.0, 1.0])

        # Apply position noise to orientation angle (alignment variability)
        angle_variation = rng.uniform(-1, 1) * position_noise * 15.0  # ±15 deg max
        varied_orientation = orientation_angle_deg + angle_variation

        # Rotate to circumferential direction then apply helical pitch
        # First align along circumference (tangent to circle)
        tangent_angle = angle + np.pi / 2
        marker = marker.rotate([0, 0, np.degrees(tangent_angle)])

        # Apply helical pitch with variation
        if varied_orientation != 0:
            marker = marker.rotate([varied_orientation, 0, 0])

        marker = marker.translate([x, y, z_pos])
        markers.append(marker)

    if markers:
        from ..core import batch_union
        return batch_union(markers)
    return m3d.Manifold()


def _make_endothelial_texture(
    lumen_radius: float,
    z_start: float,
    z_end: float,
    cell_size_um: float,
    bump_height_um: float,
    density_per_mm2: float,
    resolution: int,
    rng: np.random.Generator,
    position_noise: float = 0.0
) -> m3d.Manifold:
    """
    Create cobblestone endothelial texture on inner lumen surface.

    Endothelial cells form a cobblestone pattern on the vessel's inner surface.
    Model as hexagonal/irregular surface bumps.

    Args:
        lumen_radius: Inner radius of vessel lumen
        z_start: Start position along vessel axis
        z_end: End position along vessel axis
        cell_size_um: Diameter of endothelial cells
        bump_height_um: Height of surface bumps
        density_per_mm2: Cell density
        resolution: Resolution for geometry
        rng: Random number generator
        position_noise: Random variation (0-1 scale, 0.05-0.15 typical)

    Returns:
        Manifold with endothelial texture bumps
    """
    height = z_end - z_start
    circumference = 2 * np.pi * lumen_radius

    # Calculate surface area and number of cells
    surface_area_mm2 = circumference * height
    num_cells = int(surface_area_mm2 * density_per_mm2 / 1000)  # Reduced for performance
    num_cells = min(num_cells, 300)  # Cap for performance

    if num_cells == 0:
        return m3d.Manifold()

    base_cell_radius_mm = cell_size_um / 2000.0
    base_bump_height_mm = bump_height_um / 1000.0

    bumps = []
    for i in range(num_cells):
        # Random position on inner surface
        angle = rng.uniform(0, 2 * np.pi)
        z_pos = z_start + rng.uniform(0.05, 0.95) * height

        x = lumen_radius * np.cos(angle)
        y = lumen_radius * np.sin(angle)

        # Apply position noise to cell size (natural variation)
        size_variation = 1.0 + rng.uniform(-1, 1) * position_noise * 0.3
        cell_radius_mm = base_cell_radius_mm * size_variation
        bump_height_mm = base_bump_height_mm * size_variation

        # Create flattened sphere for cobblestone appearance
        bump = m3d.Manifold.sphere(cell_radius_mm, 6)
        bump = bump.scale([1.0, 1.0, bump_height_mm / cell_radius_mm])

        # Rotate so bump faces inward (toward center)
        bump = bump.rotate([0, 90, np.degrees(angle)])
        bump = bump.translate([x, y, z_pos])
        bumps.append(bump)

    if bumps:
        from ..core import batch_union
        return batch_union(bumps)
    return m3d.Manifold()


def _make_vasa_vasorum(
    inner_radius: float,
    outer_radius: float,
    z_start: float,
    z_end: float,
    vessel_diameter_um: float,
    spacing_um: float,
    density_per_mm2: float,
    wall_thickness_mm: float,
    resolution: int,
    rng: np.random.Generator,
    position_noise: float = 0.0
) -> list[m3d.Manifold]:
    """
    Create vasa vasorum channels in adventitia layer.

    Vasa vasorum are small blood vessels that supply the outer wall of larger
    vessels (>1mm wall thickness). They run through the adventitia.
    Spacing follows biological nutrient diffusion limits (300-500 µm).

    Args:
        inner_radius: Inner radius of adventitia
        outer_radius: Outer radius of adventitia
        z_start: Start position along vessel axis
        z_end: End position along vessel axis
        vessel_diameter_um: Diameter of vasa vasorum channels
        spacing_um: Distance between vasa vasorum (300-500 µm typical)
        density_per_mm2: Vasa vasorum density (alternative to spacing)
        wall_thickness_mm: Total wall thickness (vasa only for >1mm)
        resolution: Resolution for cylinders
        rng: Random number generator
        position_noise: Random variation (0-1 scale, 0.05-0.15 typical)

    Returns:
        List of manifolds representing channels (to subtract from adventitia)
    """
    # Vasa vasorum only present in vessels with wall thickness > 1mm
    if wall_thickness_mm < 1.0:
        return []

    height = z_end - z_start
    mid_radius = (inner_radius + outer_radius) / 2
    circumference = 2 * np.pi * mid_radius
    adventitia_thickness = outer_radius - inner_radius

    # Calculate number of channels based on spacing or density
    if spacing_um > 0:
        # Use spacing-based distribution
        spacing_mm = spacing_um / 1000.0
        num_circumferential = max(1, int(circumference / spacing_mm))
        num_axial = max(1, int(height / spacing_mm))
        num_channels = num_circumferential * num_axial
    else:
        # Use density-based distribution
        surface_area_mm2 = circumference * height
        num_channels = int(density_per_mm2 * surface_area_mm2)

    num_channels = min(num_channels, 50)  # Cap for performance

    if num_channels == 0:
        return []

    channel_radius_mm = vessel_diameter_um / 2000.0

    channels = []

    # Create grid-based positions with position noise
    angle_step = 2 * np.pi / num_circumferential
    z_step = height / num_axial

    for i in range(num_circumferential):
        for j in range(num_axial):
            # Base grid position
            base_angle = i * angle_step
            base_z = z_start + (j + 0.5) * z_step

            # Apply position noise (0-1 scale)
            angle_noise = rng.uniform(-1, 1) * position_noise * angle_step
            z_noise = rng.uniform(-1, 1) * position_noise * z_step * 0.5

            angle = base_angle + angle_noise
            z_pos = base_z + z_noise

            # Prefer denser distribution at media-adventitia border
            # (higher density in inner 60% of adventitia thickness)
            r_base = inner_radius + adventitia_thickness * 0.5
            r_noise = rng.uniform(-1, 1) * position_noise * adventitia_thickness * 0.2
            r = np.clip(r_base + r_noise,
                       inner_radius + adventitia_thickness * 0.3,
                       outer_radius - adventitia_thickness * 0.2)

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Create channel running along vessel length (axial orientation)
            # Variable length for each channel with noise
            base_length = height * 0.3
            length_noise = rng.uniform(-1, 1) * position_noise * height * 0.1
            channel_length = np.clip(base_length + length_noise, height * 0.2, height * 0.4)

            channel = m3d.Manifold.cylinder(channel_length, channel_radius_mm,
                                             channel_radius_mm, 8)
            channel = channel.translate([x, y, z_pos - channel_length / 2])
            channels.append(channel)

            # Some channels run radially (perpendicular) - with stochastic variation
            radial_threshold = 0.6 + rng.uniform(-1, 1) * position_noise * 0.2
            if rng.random() > radial_threshold:
                radial_channel = m3d.Manifold.cylinder(
                    adventitia_thickness * 0.8,
                    channel_radius_mm * 0.7,
                    channel_radius_mm * 0.7,
                    6
                )
                radial_channel = radial_channel.rotate([0, 90, np.degrees(angle)])
                radial_channel = radial_channel.translate([
                    (inner_radius + adventitia_thickness * 0.4) * np.cos(angle),
                    (inner_radius + adventitia_thickness * 0.4) * np.sin(angle),
                    z_pos
                ])
                channels.append(radial_channel)

    return channels


def _make_radial_pores(
    inner_radius: float,
    outer_radius: float,
    z_start: float,
    z_end: float,
    pore_size_um: float,
    porosity: float,
    resolution: int,
    rng: np.random.Generator,
    position_noise: float = 0.0
) -> list[m3d.Manifold]:
    """
    Create transmural (radial) pores through vessel wall.

    Pores allow cell infiltration and nutrient diffusion through the scaffold.

    Args:
        inner_radius: Inner wall radius
        outer_radius: Outer wall radius
        z_start: Start position along vessel axis
        z_end: End position along vessel axis
        pore_size_um: Diameter of pores
        porosity: Target porosity (0-1)
        resolution: Resolution for geometry
        rng: Random number generator
        position_noise: Random variation (0-1 scale, 0.05-0.15 typical)

    Returns:
        List of manifolds representing pores (to subtract)
    """
    height = z_end - z_start
    wall_thickness = outer_radius - inner_radius
    mid_radius = (inner_radius + outer_radius) / 2
    circumference = 2 * np.pi * mid_radius

    # Calculate surface area and number of pores based on porosity
    surface_area_mm2 = circumference * height
    base_pore_radius_mm = pore_size_um / 2000.0
    pore_area_mm2 = np.pi * base_pore_radius_mm ** 2

    # Estimate pore count from porosity (simplified model)
    num_pores = int(surface_area_mm2 * porosity / (pore_area_mm2 * 10))
    num_pores = min(num_pores, 100)  # Cap for performance

    if num_pores == 0:
        return []

    pores = []
    for i in range(num_pores):
        angle = rng.uniform(0, 2 * np.pi)
        z_pos = z_start + rng.uniform(0.05, 0.95) * height

        # Apply position noise to pore size (natural variation)
        size_variation = 1.0 + rng.uniform(-1, 1) * position_noise * 0.3
        pore_radius_mm = base_pore_radius_mm * size_variation

        # Pore runs radially through wall
        pore = m3d.Manifold.cylinder(
            wall_thickness * 1.5,  # Longer to ensure full penetration
            pore_radius_mm,
            pore_radius_mm,
            8
        )

        # Rotate to radial orientation
        pore = pore.rotate([0, 90, np.degrees(angle)])
        pore = pore.translate([
            (inner_radius - wall_thickness * 0.1) * np.cos(angle),
            (inner_radius - wall_thickness * 0.1) * np.sin(angle),
            z_pos
        ])
        pores.append(pore)

    return pores


def _make_pore_network(
    inner_radius: float,
    outer_radius: float,
    z_start: float,
    z_end: float,
    pore_size_um: float,
    porosity: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create interconnected pore network throughout vessel wall.

    Unlike radial pores (straight through-holes), pore networks create
    a 3D interconnected system of spherical voids for cell infiltration
    and nutrient transport.

    Args:
        inner_radius: Inner wall radius (mm)
        outer_radius: Outer wall radius (mm)
        z_start: Start position along vessel axis
        z_end: End position along vessel axis
        pore_size_um: Diameter of pores
        porosity: Target porosity (0-1)
        resolution: Resolution for geometry
        rng: Random number generator

    Returns:
        List of spherical pore manifolds to subtract from scaffold
    """
    height = z_end - z_start
    wall_thickness = outer_radius - inner_radius
    pore_radius_mm = pore_size_um / 2000.0  # Convert um to mm radius

    # Calculate volume of vessel wall (cylindrical shell)
    wall_volume = np.pi * height * (outer_radius**2 - inner_radius**2)

    # Volume of single pore
    pore_volume = (4/3) * np.pi * pore_radius_mm**3

    # Number of pores needed for target porosity
    target_void_volume = wall_volume * porosity
    num_pores = int(target_void_volume / pore_volume)

    # Cap at reasonable maximum for performance
    num_pores = min(num_pores, 2000)

    if num_pores < 1:
        return []

    pores = []
    for _ in range(num_pores):
        # Random position within the cylindrical wall
        # Radius: uniform within wall, avoiding edges
        r = inner_radius + pore_radius_mm + rng.uniform(0, 1) * (wall_thickness - 2 * pore_radius_mm)

        # Angle: uniform around circumference
        theta = rng.uniform(0, 2 * np.pi)

        # Z position: uniform along length, avoiding edges
        z = z_start + pore_radius_mm + rng.uniform(0, 1) * (height - 2 * pore_radius_mm)

        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Create spherical pore
        pore = m3d.Manifold.sphere(pore_radius_mm, max(8, resolution // 2))
        pore = pore.translate([x, y, z])
        pores.append(pore)

    return pores


def _create_tapered_cylinder(
    height: float,
    radius_bottom: float,
    radius_top: float,
    resolution: int
) -> m3d.Manifold:
    """Create a tapered cylinder (truncated cone)."""
    return m3d.Manifold.cylinder(height, radius_bottom, radius_top, resolution)


def _create_tortuous_vessel_segment(
    length: float,
    inner_radius: float,
    wall_thickness: float,
    tortuosity_index: float,
    wavelength_mm: float,
    amplitude_ratio: float,
    num_segments: int,
    resolution: int
) -> m3d.Manifold:
    """
    Create a tortuous (sinusoidal) vessel segment.

    The vessel follows a sinusoidal path instead of being straight.

    Args:
        length: Total length of vessel
        inner_radius: Inner lumen radius
        wall_thickness: Total wall thickness
        tortuosity_index: Degree of tortuosity (0-1)
        wavelength_mm: Wavelength of sinusoidal path
        amplitude_ratio: Amplitude as fraction of wavelength
        num_segments: Number of discrete segments
        resolution: Circular resolution

    Returns:
        Manifold of tortuous vessel
    """
    if tortuosity_index <= 0:
        # Straight vessel
        outer_radius = inner_radius + wall_thickness
        outer = m3d.Manifold.cylinder(length, outer_radius, outer_radius, resolution)
        inner = m3d.Manifold.cylinder(length + 0.2, inner_radius, inner_radius, resolution)
        return outer - inner.translate([0, 0, -0.1])

    # Sinusoidal path parameters
    amplitude = wavelength_mm * amplitude_ratio * tortuosity_index
    outer_radius = inner_radius + wall_thickness

    segments = []
    segment_length = length / num_segments

    for i in range(num_segments):
        z_start = i * segment_length
        z_mid = z_start + segment_length / 2
        z_end = z_start + segment_length

        # Calculate positions along sinusoidal path
        x_start = amplitude * np.sin(2 * np.pi * z_start / wavelength_mm)
        x_end = amplitude * np.sin(2 * np.pi * z_end / wavelength_mm)

        # Create segment as cylinder
        seg_outer = m3d.Manifold.cylinder(segment_length * 1.1, outer_radius,
                                           outer_radius, resolution)
        seg_inner = m3d.Manifold.cylinder(segment_length * 1.2, inner_radius,
                                           inner_radius, resolution)
        segment = seg_outer - seg_inner.translate([0, 0, -0.05])

        # Calculate rotation angle to align segment with path
        dx = x_end - x_start
        dz = segment_length
        angle = np.arctan2(dx, dz)

        # Rotate and position segment
        segment = segment.rotate([0, np.degrees(angle), 0])
        segment = segment.translate([x_start, 0, z_start])
        segments.append(segment)

    if segments:
        from ..core import batch_union
        return batch_union(segments)
    return m3d.Manifold()


def generate_blood_vessel(params: BloodVesselParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a blood vessel scaffold with tri-layer wall structure.

    Creates concentric tubes representing intima, media, and adventitia layers.
    Optionally adds microstructural features: elastic laminae, SMC alignment,
    endothelial texture, vasa vasorum, bifurcation, tortuosity, and taper.

    Args:
        params: BloodVesselParams with generation parameters

    Returns:
        Tuple of (manifold, stats_dict)

    Raises:
        ValueError: If wall thickness exceeds radius or layer ratios invalid
    """
    # Initialize random generator
    rng = np.random.default_rng(params.random_seed)

    inner_radius = params.inner_diameter_mm / 2
    outer_radius = inner_radius + params.wall_thickness_mm

    if inner_radius <= 0:
        raise ValueError("Inner diameter must be positive")

    # Calculate layer boundaries
    intima_thickness = params.wall_thickness_mm * params.layer_ratios[0]
    media_thickness = params.wall_thickness_mm * params.layer_ratios[1]
    adventitia_thickness = params.wall_thickness_mm * params.layer_ratios[2]

    r0 = inner_radius  # Lumen inner wall
    r1 = r0 + intima_thickness
    r2 = r1 + media_thickness
    r3 = r2 + adventitia_thickness

    # Apply taper if specified
    taper = params.taper_ratio if params.taper_ratio > 0 else params.vessel_taper
    taper_factor = 1.0 - min(taper, 0.3)  # Cap at 30% taper

    # Determine vessel length and segment count
    vessel_length = params.length_mm

    # Check if tortuous path is needed
    tortuosity = params.tortuosity_index if params.tortuosity_index > 0 else params.vessel_tortuosity
    use_tortuosity = tortuosity > 0

    def create_vessel_segment(
        length: float,
        r_inner: float,
        r_intima: float,
        r_media: float,
        r_outer: float,
        offset_z: float = 0,
        apply_taper: bool = False,
        taper_mult: float = 1.0
    ) -> m3d.Manifold:
        """Create a tri-layer vessel segment with optional taper."""

        if apply_taper and taper_mult != 1.0:
            # Tapered vessel
            r_inner_top = r_inner * taper_mult
            r_intima_top = r_intima * taper_mult
            r_media_top = r_media * taper_mult
            r_outer_top = r_outer * taper_mult
        else:
            r_inner_top = r_inner
            r_intima_top = r_intima
            r_media_top = r_media
            r_outer_top = r_outer

        # Intima layer (innermost)
        intima_outer = _create_tapered_cylinder(length, r_intima, r_intima_top, params.resolution)
        intima_inner = _create_tapered_cylinder(length + 0.2, r_inner, r_inner_top, params.resolution)
        intima = intima_outer - intima_inner.translate([0, 0, -0.1])

        # Media layer (middle)
        media_outer = _create_tapered_cylinder(length, r_media, r_media_top, params.resolution)
        media_inner = _create_tapered_cylinder(length + 0.2, r_intima, r_intima_top, params.resolution)
        media = media_outer - media_inner.translate([0, 0, -0.1])

        # Adventitia layer (outermost)
        adventitia_outer = _create_tapered_cylinder(length, r_outer, r_outer_top, params.resolution)
        adventitia_inner = _create_tapered_cylinder(length + 0.2, r_media, r_media_top, params.resolution)
        adventitia = adventitia_outer - adventitia_inner.translate([0, 0, -0.1])

        # Combine all layers
        from ..core import batch_union
        vessel = batch_union([intima, media, adventitia])

        if offset_z != 0:
            vessel = vessel.translate([0, 0, offset_z])

        return vessel

    # Create main vessel (with or without tortuosity)
    if use_tortuosity and not params.enable_bifurcation and params.bifurcation_angle_deg is None:
        # Create tortuous vessel
        main_vessel = _create_tortuous_vessel_segment(
            length=vessel_length,
            inner_radius=r0,
            wall_thickness=params.wall_thickness_mm,
            tortuosity_index=tortuosity,
            wavelength_mm=params.tortuosity_wavelength_mm,
            amplitude_ratio=params.tortuosity_amplitude_ratio,
            num_segments=max(10, int(vessel_length / params.tortuosity_wavelength_mm * 4)),
            resolution=params.resolution
        )
    elif params.enable_bifurcation or params.bifurcation_angle_deg is not None:
        # Create vessel with Murray's law bifurcation
        trunk_length = vessel_length * 0.67

        # Create trunk (with taper applied to full trunk)
        trunk = create_vessel_segment(
            trunk_length, r0, r1, r2, r3,
            apply_taper=taper > 0,
            taper_mult=1.0 - taper * 0.67  # Partial taper for trunk
        )

        # Bifurcation parameters
        bif_angle = params.bifurcation_angle_deg if params.bifurcation_angle_deg else 75.0
        daughter_ratio = params.daughter_diameter_ratio  # Murray's law: 0.794

        # Daughter vessel dimensions (Murray's law scaling)
        d_r0 = r0 * daughter_ratio
        d_r1 = r1 * daughter_ratio
        d_r2 = r2 * daughter_ratio
        d_r3 = r3 * daughter_ratio

        branch_length = vessel_length * 0.4
        angle_rad = np.radians(bif_angle / 2)

        # Branch 1 (rotated positive)
        branch1 = create_vessel_segment(branch_length, d_r0, d_r1, d_r2, d_r3)
        branch1 = branch1.rotate([0, np.degrees(angle_rad), 0])
        branch1 = branch1.translate([
            branch_length / 2 * np.sin(angle_rad),
            0,
            trunk_length
        ])

        # Branch 2 (rotated negative)
        branch2 = create_vessel_segment(branch_length, d_r0, d_r1, d_r2, d_r3)
        branch2 = branch2.rotate([0, -np.degrees(angle_rad), 0])
        branch2 = branch2.translate([
            -branch_length / 2 * np.sin(angle_rad),
            0,
            trunk_length
        ])

        # Create transition region at bifurcation (smooth junction)
        junction_height = r3 * 0.5
        junction = m3d.Manifold.sphere(r3 * 1.2, params.resolution)
        junction = junction.scale([1.0, 1.0, 0.5])
        junction = junction.translate([0, 0, trunk_length])

        # Combine trunk, junction, and branches
        from ..core import batch_union
        main_vessel = batch_union([trunk, junction, branch1, branch2])

        # Carve out lumen through junction
        lumen_carve = m3d.Manifold.cylinder(
            trunk_length + junction_height + 0.1,
            r0, r0 * daughter_ratio, params.resolution
        )
        main_vessel = main_vessel - lumen_carve.translate([0, 0, -0.05])
    else:
        # Simple straight vessel with optional taper
        main_vessel = create_vessel_segment(
            vessel_length, r0, r1, r2, r3,
            apply_taper=taper > 0,
            taper_mult=taper_factor
        )

    # === Add microstructural features ===
    features_to_add = []
    features_to_subtract = []

    # Elastic laminae (IEL and EEL)
    if params.enable_elastic_laminae:
        # Internal elastic lamina at intima-media boundary
        iel_thickness_mm = params.elastic_lamina_thickness_um / 1000.0
        iel = _make_elastic_lamina(
            radius=r1,
            thickness_mm=iel_thickness_mm,
            z_start=0,
            z_end=vessel_length,
            num_fenestrations=params.fenestration_count,
            fenestration_diameter_mm=params.fenestration_diameter_um / 1000.0,
            resolution=params.resolution,
            rng=rng,
            position_noise=params.position_noise
        )
        features_to_add.append(iel)

        # External elastic lamina at media-adventitia boundary
        eel_thickness_mm = params.external_elastic_lamina_thickness_um / 1000.0
        eel = _make_elastic_lamina(
            radius=r2,
            thickness_mm=eel_thickness_mm,
            z_start=0,
            z_end=vessel_length,
            num_fenestrations=params.fenestration_count,
            fenestration_diameter_mm=params.fenestration_diameter_um / 1000.0,
            resolution=params.resolution,
            rng=rng,
            position_noise=params.position_noise
        )
        features_to_add.append(eel)

        # Additional elastic laminae within media (if num_elastic_laminae > 2)
        if params.num_elastic_laminae > 2:
            num_internal = params.num_elastic_laminae - 2
            for i in range(num_internal):
                # Distribute evenly through media layer
                fraction = (i + 1) / (num_internal + 1)
                lamina_radius = r1 + media_thickness * fraction
                lamina = _make_elastic_lamina(
                    radius=lamina_radius,
                    thickness_mm=iel_thickness_mm * 0.8,  # Slightly thinner internal laminae
                    z_start=0,
                    z_end=vessel_length,
                    num_fenestrations=int(params.fenestration_count * 0.7),
                    fenestration_diameter_mm=params.fenestration_diameter_um / 1000.0,
                    resolution=params.resolution,
                    rng=rng,
                    position_noise=params.position_noise
                )
                features_to_add.append(lamina)

    # SMC alignment markers
    if params.enable_smc_alignment:
        smc_markers = _make_smc_alignment_markers(
            inner_radius=r1,
            outer_radius=r2,
            z_start=0,
            z_end=vessel_length,
            orientation_angle_deg=params.smc_orientation_angle_deg,
            density_per_mm2=params.smc_density_per_mm2,
            smc_length_um=params.smc_length_um,
            resolution=params.resolution,
            rng=rng,
            position_noise=params.position_noise
        )
        if smc_markers.num_vert() > 0:
            features_to_add.append(smc_markers)

    # Endothelial texture
    if params.enable_endothelial_texture:
        endo_texture = _make_endothelial_texture(
            lumen_radius=r0,
            z_start=0,
            z_end=vessel_length,
            cell_size_um=params.endothelial_cell_size_um,
            bump_height_um=params.endothelial_bump_height_um,
            density_per_mm2=params.endothelial_cell_density_per_mm2,
            resolution=params.resolution,
            rng=rng,
            position_noise=params.position_noise
        )
        if endo_texture.num_vert() > 0:
            features_to_add.append(endo_texture)

    # Vasa vasorum channels (subtracted from adventitia)
    if params.enable_vasa_vasorum:
        vasa_channels = _make_vasa_vasorum(
            inner_radius=r2,
            outer_radius=r3,
            z_start=0,
            z_end=vessel_length,
            vessel_diameter_um=params.vasa_vasorum_diameter_um,
            spacing_um=params.vasa_vasorum_spacing_um,
            density_per_mm2=params.vasa_vasorum_density_per_mm2,
            wall_thickness_mm=params.wall_thickness_mm,
            resolution=params.resolution,
            rng=rng,
            position_noise=params.position_noise
        )
        features_to_subtract.extend(vasa_channels)

    # Radial pores for porosity
    if params.enable_radial_pores and params.scaffold_porosity > 0:
        pores = _make_radial_pores(
            inner_radius=r0,
            outer_radius=r3,
            z_start=0,
            z_end=vessel_length,
            pore_size_um=params.pore_size_um,
            porosity=params.scaffold_porosity,
            resolution=params.resolution,
            rng=rng,
            position_noise=params.position_noise
        )
        features_to_subtract.extend(pores)

    # Interconnected pore network
    if params.enable_pore_network and params.scaffold_porosity > 0:
        pore_network = _make_pore_network(
            inner_radius=r0,
            outer_radius=r3,
            z_start=0,
            z_end=vessel_length,
            pore_size_um=params.pore_size_um,
            porosity=params.scaffold_porosity,
            resolution=params.resolution,
            rng=rng
        )
        features_to_subtract.extend(pore_network)

    # Combine all features
    vessel = main_vessel

    if features_to_add:
        from ..core import batch_union
        all_additions = batch_union(features_to_add)
        vessel = vessel + all_additions

    if features_to_subtract:
        from ..core import batch_union
        all_subtractions = batch_union(features_to_subtract)
        vessel = vessel - all_subtractions

    # Calculate statistics
    mesh = vessel.to_mesh()
    volume = vessel.volume() if hasattr(vessel, 'volume') else 0

    # Calculate actual tortuosity index (path length / straight length)
    actual_tortuosity = 1.0
    if use_tortuosity and tortuosity > 0:
        # Approximate calculation
        wavelength = params.tortuosity_wavelength_mm
        amplitude = wavelength * params.tortuosity_amplitude_ratio * tortuosity
        num_waves = vessel_length / wavelength
        # Arc length approximation for sinusoid
        actual_tortuosity = 1.0 + (2 * np.pi * amplitude / wavelength) ** 2 / 4 * num_waves / (vessel_length / wavelength)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'inner_diameter_mm': params.inner_diameter_mm,
        'wall_thickness_mm': params.wall_thickness_mm,
        'length_mm': params.length_mm,
        'layer_thicknesses_mm': {
            'intima': intima_thickness,
            'media': media_thickness,
            'adventitia': adventitia_thickness
        },
        'layer_thicknesses_um': {
            'intima': params.intima_thickness_um,
            'media': params.media_thickness_um,
            'adventitia': params.adventitia_thickness_um
        },
        # Elastic laminae
        'has_elastic_laminae': params.enable_elastic_laminae,
        'elastic_laminae_count': params.num_elastic_laminae if params.enable_elastic_laminae else 0,
        'iel_thickness_um': params.elastic_lamina_thickness_um,
        'eel_thickness_um': params.external_elastic_lamina_thickness_um,
        'fenestration_count_per_lamina': params.fenestration_count if params.enable_elastic_laminae else 0,
        # SMC alignment
        'has_smc_alignment': params.enable_smc_alignment,
        'smc_orientation_deg': params.smc_orientation_angle_deg,
        # Endothelial texture
        'has_endothelial_texture': params.enable_endothelial_texture,
        'endothelial_cell_size_um': params.endothelial_cell_size_um,
        # Vasa vasorum
        'has_vasa_vasorum': params.enable_vasa_vasorum,
        'vasa_vasorum_diameter_um': params.vasa_vasorum_diameter_um if params.enable_vasa_vasorum else 0,
        'vasa_vasorum_spacing_um': params.vasa_vasorum_spacing_um if params.enable_vasa_vasorum else 0,
        # Bifurcation
        'has_bifurcation': params.enable_bifurcation or params.bifurcation_angle_deg is not None,
        'bifurcation_angle_deg': params.bifurcation_angle_deg,
        'daughter_diameter_ratio': params.daughter_diameter_ratio,
        # Geometry variations
        'taper_ratio': taper,
        'vessel_taper': taper,
        'tortuosity_index': tortuosity,
        'vessel_tortuosity': tortuosity,
        'actual_tortuosity_index': actual_tortuosity,
        # Porosity
        'scaffold_porosity': params.scaffold_porosity,
        'has_radial_pores': params.enable_radial_pores,
        'has_pore_network': params.enable_pore_network,
        'pore_size_um': params.pore_size_um,
        # Mechanical properties (FEA design specs)
        'target_compliance_percent_per_100mmHg': params.target_compliance_percent_per_100mmHg,
        'burst_pressure_mmHg': params.burst_pressure_mmHg,
        # Randomization
        'position_noise': params.position_noise,
        'random_seed': params.random_seed,
        'scaffold_type': 'blood_vessel'
    }

    return vessel, stats


def generate_blood_vessel_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate a blood vessel scaffold from dictionary parameters.

    Args:
        params: Dictionary of parameters (will be unpacked into BloodVesselParams)

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle parameter name mappings for compatibility
    param_copy = params.copy()

    # Map frontend parameter names to dataclass field names if needed
    if 'intima_thickness_um' not in param_copy and 'layer_ratios' not in param_copy:
        # Use defaults
        pass

    # Filter to only valid BloodVesselParams fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(BloodVesselParams)}
    filtered_params = {k: v for k, v in param_copy.items() if k in valid_fields}

    return generate_blood_vessel(BloodVesselParams(**filtered_params))
