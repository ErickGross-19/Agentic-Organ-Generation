"""
Nerve conduit scaffold generator.

Generates multi-channel nerve guidance conduits with internal microchannels for axon regeneration.
Includes biologically realistic parameters for nerve fascicle architecture.

Biological basis:
- Fascicles: 300-500μm diameter bundles containing nerve fibers
- Perineurium: 15-30μm connective tissue sheath around each fascicle
- Epineurium: 100-150μm outer connective tissue layer
- Microgrooves: 20μm wide, 3μm deep longitudinal channels for axon guidance
- Growth factor reservoirs: 300-500μm spherical cavities for drug delivery
"""

import manifold3d as m3d
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class NerveConduitParams:
    """
    Parameters for nerve conduit scaffold generation.

    Based on peripheral nerve anatomy with fascicular organization:
    - Endoneurium: surrounds individual axons
    - Perineurium: surrounds fascicles
    - Epineurium: outermost connective tissue
    """

    # === Basic Geometry ===
    outer_diameter_mm: float = 2.5  # Conduit outer diameter
    inner_diameter_mm: float = 2.0  # Lumen diameter (derived from wall thickness)
    wall_thickness_mm: float = 0.4  # Conduit wall thickness
    length_mm: float = 30.0  # Total conduit length

    # === Channel Structure ===
    num_channels: int = 50  # Number of guidance channels
    channel_diameter_um: float = 150.0  # Individual channel diameter (100-300um for axons)
    channel_spacing_um: float = 100.0  # Minimum spacing between channels

    # === Fascicle Architecture ===
    fascicle_diameter_um: float = 400.0  # Fascicle bundle diameter (300-500μm typical)
    num_fascicles: int = 4  # Number of fascicle compartments
    enable_fascicle_chambers: bool = False  # Create fascicle compartments
    fascicle_spacing_um: float = 150.0  # Separation between fascicles (100-200μm)

    # === Perineurium Layer ===
    perineurium_thickness_um: float = 20.0  # Perineurial sheath (15-30μm typical)
    enable_perineurium: bool = False  # Add perineurium around fascicles

    # === Epineurium Layer ===
    epineurium_thickness_um: float = 125.0  # Outer connective tissue (100-150μm)
    enable_epineurium: bool = False  # Add epineurial sheath

    # === Schwann Cell Guidance Features ===
    ridge_width_um: float = 5.0  # Microridge width for Schwann cells
    groove_width_um: float = 20.0  # Microgroove width (20μm recommended)
    groove_depth_um: float = 3.0  # Groove depth (3μm recommended)
    groove_spacing_um: float = 25.0  # Spacing between grooves (groove + ridge)
    enable_microgrooves: bool = False  # Surface microgrooves for cell alignment
    num_microgrooves: int = 0  # Auto-calculated if 0

    # === Inner Surface Features ===
    enable_guidance_channels: bool = True  # Internal guidance channels
    guidance_channel_diameter_um: float = 50.0  # Smaller guidance microchannels
    guidance_channel_pattern: str = "linear"  # "linear" or "spiral"

    # === Wall Porosity ===
    wall_porosity: float = 0.75  # Target wall porosity for nutrient diffusion
    pore_size_um: float = 50.0  # Wall pore size
    inner_surface_porosity: float = 0.3  # Inner surface porosity

    # === Growth Factor Delivery ===
    enable_growth_factor_reservoirs: bool = False  # Drug delivery reservoirs
    reservoir_diameter_um: float = 400.0  # Reservoir size (300-500μm spherical cavities)
    reservoir_count: int = 4  # Number of reservoirs per ring
    reservoir_spacing_mm: float = 2.5  # Axial spacing between reservoir rings (2-3mm)
    reservoir_rings: int = 0  # Auto-calculated based on length if 0

    # === Suture Features ===
    enable_biodegradable_suture_holes: bool = False  # Pre-made suture holes
    suture_hole_diameter_mm: float = 0.5  # Suture hole size (500μm)
    suture_hole_count: int = 4  # Number per end (4-6 typical)
    suture_hole_distance_from_end_mm: float = 2.5  # Distance from end (2-3mm)

    # === Geometry Variations ===
    taper_ratio: float = 1.0  # End diameter / start diameter (1.0 = no taper)
    enable_flared_ends: bool = False  # Flared ends for easier anastomosis
    flare_angle_deg: float = 15.0  # Flare angle (15° typical)
    flare_length_mm: float = 1.5  # Length of flared section (1-2mm)

    # === Randomization ===
    position_noise_um: float = 0.0  # Random variation in channel positions
    channel_variance: float = 0.0  # Variation in channel diameter (0-0.5)
    random_seed: int = 42

    # === Resolution ===
    resolution: int = 12  # Cylinder segments

    def __post_init__(self):
        if self.num_channels < 1:
            raise ValueError("num_channels must be at least 1")
        if self.channel_diameter_um < 50 or self.channel_diameter_um > 500:
            raise ValueError("channel_diameter_um must be between 50 and 500")
        # Derive inner_diameter from outer and wall if not explicitly matching
        expected_inner = self.outer_diameter_mm - 2 * self.wall_thickness_mm
        if abs(self.inner_diameter_mm - expected_inner) > 0.01:
            self.inner_diameter_mm = expected_inner
        # Calculate ridge_width from groove parameters (30 µm optimal)
        calculated_ridge = self.groove_spacing_um - self.groove_width_um
        if abs(self.ridge_width_um - calculated_ridge) > 0.1:
            self.ridge_width_um = calculated_ridge


# =============================================================================
# Helper Functions for Nerve Conduit Features
# =============================================================================


def _create_fascicle_chambers(
    params: NerveConduitParams,
    conduit: m3d.Manifold,
    inner_radius: float,
    length: float
) -> Tuple[m3d.Manifold, List[Tuple[float, float]]]:
    """
    Create fascicle chambers with optional perineurium.

    Fascicles are arranged in hexagonal packing within the lumen.
    Each fascicle is a parallel cylindrical channel through the conduit.

    Args:
        params: Generation parameters
        conduit: Current conduit manifold
        inner_radius: Inner radius of conduit in mm
        length: Length of conduit in mm

    Returns:
        Tuple of (modified conduit, list of fascicle center positions)
    """
    fascicle_radius_mm = params.fascicle_diameter_um / 2000.0
    spacing_mm = params.fascicle_spacing_um / 1000.0

    # Calculate positions for fascicles in hexagonal arrangement
    fascicle_positions = []

    if params.num_fascicles == 1:
        # Single central fascicle
        fascicle_positions.append((0.0, 0.0))
    elif params.num_fascicles <= 7:
        # Center + ring arrangement (like honeycomb)
        # Add central fascicle
        fascicle_positions.append((0.0, 0.0))

        # Ring of fascicles around center
        ring_radius = fascicle_radius_mm * 2 + spacing_mm
        n_ring = min(params.num_fascicles - 1, 6)
        for i in range(n_ring):
            angle = 2 * math.pi * i / 6  # Hexagonal angles
            x = ring_radius * math.cos(angle)
            y = ring_radius * math.sin(angle)
            fascicle_positions.append((x, y))
    else:
        # Multiple rings for more fascicles
        fascicle_positions.append((0.0, 0.0))
        ring_radius = fascicle_radius_mm * 2 + spacing_mm

        # First ring (6 fascicles)
        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = ring_radius * math.cos(angle)
            y = ring_radius * math.sin(angle)
            fascicle_positions.append((x, y))

        # Second ring if needed (12 fascicles)
        if params.num_fascicles > 7:
            ring_radius_2 = ring_radius * 2
            n_outer = min(params.num_fascicles - 7, 12)
            for i in range(n_outer):
                angle = 2 * math.pi * i / 12 + math.pi / 12  # Offset
                x = ring_radius_2 * math.cos(angle)
                y = ring_radius_2 * math.sin(angle)
                fascicle_positions.append((x, y))

    # Trim to requested number
    fascicle_positions = fascicle_positions[:params.num_fascicles]

    # Create fascicle channels
    channels = []
    for (fx, fy) in fascicle_positions:
        # Each fascicle is a cylindrical void
        channel = m3d.Manifold.cylinder(
            length + 0.4,
            fascicle_radius_mm,
            fascicle_radius_mm,
            params.resolution
        ).translate([fx, fy, -0.2])
        channels.append(channel)

    # Union all channels and subtract
    if channels:
        from ..core import batch_union
        all_channels = batch_union(channels)
        conduit = conduit - all_channels

    # Add perineurium shells if enabled
    if params.enable_perineurium:
        perineurium_thickness_mm = params.perineurium_thickness_um / 1000.0
        perineurium_shells = []

        for (fx, fy) in fascicle_positions:
            # Create thin shell around each fascicle
            outer_shell = m3d.Manifold.cylinder(
                length + 0.2,
                fascicle_radius_mm + perineurium_thickness_mm,
                fascicle_radius_mm + perineurium_thickness_mm,
                params.resolution
            ).translate([fx, fy, -0.1])

            inner_shell = m3d.Manifold.cylinder(
                length + 0.4,
                fascicle_radius_mm,
                fascicle_radius_mm,
                params.resolution
            ).translate([fx, fy, -0.2])

            shell = outer_shell - inner_shell
            perineurium_shells.append(shell)

        if perineurium_shells:
            from ..core import batch_union
            all_perineurium = batch_union(perineurium_shells)
            conduit = conduit + all_perineurium

    return conduit, fascicle_positions


def _create_epineurium(
    params: NerveConduitParams,
    conduit: m3d.Manifold,
    outer_radius: float,
    length: float
) -> m3d.Manifold:
    """
    Add epineurium outer layer to the conduit.

    The epineurium is the outermost connective tissue layer that surrounds
    the entire nerve bundle. Thickness is typically 100-150μm.

    Args:
        params: Generation parameters
        conduit: Current conduit manifold
        outer_radius: Current outer radius in mm
        length: Length of conduit in mm

    Returns:
        Modified conduit with epineurium layer
    """
    epineurium_mm = params.epineurium_thickness_um / 1000.0
    new_outer_radius = outer_radius + epineurium_mm

    # Create epineurium shell
    outer_cylinder = m3d.Manifold.cylinder(
        length,
        new_outer_radius,
        new_outer_radius,
        params.resolution * 2
    )

    inner_cylinder = m3d.Manifold.cylinder(
        length + 0.2,
        outer_radius,
        outer_radius,
        params.resolution * 2
    ).translate([0, 0, -0.1])

    epineurium_shell = outer_cylinder - inner_cylinder

    return conduit + epineurium_shell


def _create_microgrooves(
    params: NerveConduitParams,
    conduit: m3d.Manifold,
    inner_radius: float,
    length: float
) -> m3d.Manifold:
    """
    Create longitudinal microgrooves on the inner surface with ridges between.

    Microgrooves guide Schwann cells and axons along the conduit length.
    Ridges between grooves provide structural support and additional guidance.
    Typical dimensions: 20μm groove width, 3μm depth, 5μm ridge width.

    Args:
        params: Generation parameters
        conduit: Current conduit manifold
        inner_radius: Inner radius in mm
        length: Length of conduit in mm

    Returns:
        Modified conduit with microgrooves and ridges
    """
    groove_width_mm = params.groove_width_um / 1000.0
    groove_depth_mm = params.groove_depth_um / 1000.0
    groove_spacing_mm = params.groove_spacing_um / 1000.0
    ridge_width_mm = params.ridge_width_um / 1000.0  # Width of ridge between grooves

    # Calculate number of grooves based on circumference
    circumference = 2 * math.pi * inner_radius
    if params.num_microgrooves > 0:
        num_grooves = params.num_microgrooves
    else:
        # Calculate based on groove + ridge pattern
        pattern_width = groove_width_mm + ridge_width_mm
        num_grooves = int(circumference / pattern_width)

    if num_grooves < 4:
        num_grooves = 4  # Minimum for meaningful guidance

    # Create grooves as thin cylinders extruded along the length
    grooves = []
    for i in range(num_grooves):
        # Distribute grooves evenly around circumference
        angle = 2 * math.pi * i / num_grooves

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Groove center position on inner wall
        # Position groove slightly inward from inner surface
        groove_radial_position = inner_radius - groove_depth_mm / 2

        # Use a small cylinder to approximate groove
        groove_cyl = m3d.Manifold.cylinder(
            length + 0.2,
            groove_width_mm / 2,
            groove_width_mm / 2,
            max(6, params.resolution // 2)
        ).translate([
            groove_radial_position * cos_a,
            groove_radial_position * sin_a,
            -0.1
        ])
        grooves.append(groove_cyl)

    # Union all grooves and subtract from conduit
    if grooves:
        from ..core import batch_union
        all_grooves = batch_union(grooves)
        conduit = conduit - all_grooves

    return conduit


def _create_growth_factor_reservoirs(
    params: NerveConduitParams,
    conduit: m3d.Manifold,
    outer_radius: float,
    inner_radius: float,
    length: float
) -> m3d.Manifold:
    """
    Create spherical growth factor reservoirs in the conduit wall.

    Reservoirs allow controlled release of neurotrophic factors (NGF, BDNF, etc.).
    Typical size: 300-500μm diameter, spaced 2-3mm apart axially.

    Args:
        params: Generation parameters
        conduit: Current conduit manifold
        outer_radius: Outer radius in mm
        inner_radius: Inner radius in mm
        length: Length of conduit in mm

    Returns:
        Modified conduit with reservoir cavities
    """
    reservoir_radius_mm = params.reservoir_diameter_um / 2000.0
    wall_midpoint = (outer_radius + inner_radius) / 2

    # Calculate number of axial rings
    if params.reservoir_rings > 0:
        num_rings = params.reservoir_rings
    else:
        # Auto-calculate based on length and spacing
        usable_length = length - 4.0  # Leave 2mm at each end
        num_rings = max(1, int(usable_length / params.reservoir_spacing_mm))

    reservoirs = []
    for ring_idx in range(num_rings):
        # Axial position of this ring
        z_pos = 2.0 + ring_idx * params.reservoir_spacing_mm
        if z_pos > length - 2.0:
            break

        # Place reservoirs circumferentially
        for i in range(params.reservoir_count):
            angle = 2 * math.pi * i / params.reservoir_count
            # Offset alternating rings for better distribution
            if ring_idx % 2 == 1:
                angle += math.pi / params.reservoir_count

            x = wall_midpoint * math.cos(angle)
            y = wall_midpoint * math.sin(angle)

            # Create spherical reservoir
            sphere = m3d.Manifold.sphere(
                reservoir_radius_mm,
                max(8, params.resolution)
            ).translate([x, y, z_pos])
            reservoirs.append(sphere)

    # Subtract all reservoirs from conduit
    if reservoirs:
        from ..core import batch_union
        all_reservoirs = batch_union(reservoirs)
        conduit = conduit - all_reservoirs

    return conduit


def _create_suture_holes(
    params: NerveConduitParams,
    conduit: m3d.Manifold,
    outer_radius: float,
    length: float
) -> m3d.Manifold:
    """
    Create suture holes near conduit ends for surgical attachment.

    Holes are placed circumferentially at each end for suturing the conduit
    to nerve stumps. Typical: 0.5mm diameter, 4-6 per end, 2-3mm from end.

    Args:
        params: Generation parameters
        conduit: Current conduit manifold
        outer_radius: Outer radius in mm
        length: Length of conduit in mm

    Returns:
        Modified conduit with suture holes
    """
    hole_radius = params.suture_hole_diameter_mm / 2
    dist_from_end = params.suture_hole_distance_from_end_mm

    holes = []

    # Create holes at both ends
    for end_z in [dist_from_end, length - dist_from_end]:
        for i in range(params.suture_hole_count):
            angle = 2 * math.pi * i / params.suture_hole_count

            # Radial direction for hole
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Create horizontal cylinder through wall
            # Rotate cylinder to point radially
            hole_length = outer_radius * 2.5  # Ensure it goes through

            hole = m3d.Manifold.cylinder(
                hole_length,
                hole_radius,
                hole_radius,
                params.resolution
            )

            # Rotate to point radially outward (along X initially, rotate around Z)
            # Cylinder is along Z by default, need to rotate 90° around Y
            # Then rotate around Z to correct angle
            hole = hole.rotate([0, 90, 0])  # Now along X
            hole = hole.rotate([0, 0, math.degrees(angle)])  # Rotate to angle
            hole = hole.translate([0, 0, end_z])

            holes.append(hole)

    # Subtract all holes
    if holes:
        from ..core import batch_union
        all_holes = batch_union(holes)
        conduit = conduit - all_holes

    return conduit


def _create_spiral_channels(
    params: NerveConduitParams,
    length: float,
    channel_radius_mm: float,
    helix_radius: float,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create spiral guidance channels that wrap helically around the conduit axis.

    Spiral patterns significantly enhance cellular activity and axon guidance
    compared to linear channels.

    Args:
        params: Generation parameters
        length: Conduit length in mm
        channel_radius_mm: Radius of each channel in mm
        helix_radius: Radius of the helix path in mm
        rng: Random number generator for reproducibility

    Returns:
        List of channel manifolds
    """
    channels = []
    num_spirals = max(1, params.num_channels // 3)  # Fewer spirals than linear channels
    pitch = length / 2.0  # Two complete turns per conduit length

    for i in range(num_spirals):
        # Phase offset for each spiral
        phase_offset = 2 * math.pi * i / num_spirals

        # Create spiral path as series of small cylindrical segments
        num_segments = int(length * 20)  # 20 segments per mm for smooth spiral
        segment_height = length / num_segments

        spiral_segments = []
        for seg in range(num_segments):
            z = seg * segment_height
            angle = (2 * math.pi * z / pitch) + phase_offset

            # Apply position noise if enabled
            noise_x = noise_y = 0.0
            if params.position_noise_um > 0:
                noise_x = (rng.random() - 0.5) * 2 * params.position_noise_um / 1000.0
                noise_y = (rng.random() - 0.5) * 2 * params.position_noise_um / 1000.0

            x = helix_radius * math.cos(angle) + noise_x
            y = helix_radius * math.sin(angle) + noise_y

            # Adjust channel diameter with variance
            local_radius = channel_radius_mm
            if params.channel_variance > 0:
                variance_factor = 1.0 + (rng.random() - 0.5) * 2 * params.channel_variance
                local_radius = channel_radius_mm * variance_factor

            # Create small segment
            segment = m3d.Manifold.cylinder(
                segment_height * 1.2,  # Overlap slightly
                local_radius,
                local_radius,
                params.resolution
            ).translate([x, y, z])
            spiral_segments.append(segment)

        # Union all segments into single spiral
        if spiral_segments:
            from ..core import batch_union
            spiral = batch_union(spiral_segments)
            channels.append(spiral)

    return channels


def _create_wall_pores(
    params: NerveConduitParams,
    conduit: m3d.Manifold,
    outer_radius: float,
    inner_radius: float,
    length: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create porous structure in the conduit wall for nutrient diffusion.

    Uses bi-layered asymmetric pore distribution:
    - Inner wall: 10-40 µm pores (60-80% porosity)
    - Outer wall: 50-60 µm pores (lower porosity)

    Args:
        params: Generation parameters
        conduit: Current conduit manifold
        outer_radius: Outer radius in mm
        inner_radius: Inner radius in mm
        length: Length of conduit in mm
        rng: Random number generator

    Returns:
        Modified conduit with porous wall structure
    """
    wall_thickness = outer_radius - inner_radius
    mid_radius = (inner_radius + outer_radius) / 2

    # Calculate target number of pores based on surface area and porosity
    inner_surface_area = 2 * math.pi * inner_radius * length
    target_inner_porosity = params.inner_surface_porosity
    pore_size_mm = params.pore_size_um / 1000.0
    pore_area = math.pi * (pore_size_mm / 2) ** 2

    # Inner wall pores (10-40 µm range, use lower end of pore_size_um)
    inner_pore_radius = min(params.pore_size_um, 40.0) / 2000.0
    num_inner_pores = int(inner_surface_area * target_inner_porosity / pore_area)

    # Outer wall pores (50-60 µm range, use full pore_size_um)
    outer_pore_radius = params.pore_size_um / 2000.0
    target_outer_porosity = params.wall_porosity - target_inner_porosity
    num_outer_pores = int(inner_surface_area * target_outer_porosity / pore_area)

    pores = []

    # Inner wall pores
    for _ in range(num_inner_pores):
        # Random position along circumference and length
        angle = rng.random() * 2 * math.pi
        z_pos = rng.random() * length
        radial_pos = inner_radius + rng.random() * wall_thickness * 0.3  # Inner 30% of wall

        x = radial_pos * math.cos(angle)
        y = radial_pos * math.sin(angle)

        pore = m3d.Manifold.sphere(
            inner_pore_radius,
            max(6, params.resolution // 2)
        ).translate([x, y, z_pos])
        pores.append(pore)

    # Outer wall pores
    for _ in range(num_outer_pores):
        angle = rng.random() * 2 * math.pi
        z_pos = rng.random() * length
        radial_pos = mid_radius + rng.random() * wall_thickness * 0.5  # Outer 50% of wall

        x = radial_pos * math.cos(angle)
        y = radial_pos * math.sin(angle)

        pore = m3d.Manifold.sphere(
            outer_pore_radius,
            max(6, params.resolution // 2)
        ).translate([x, y, z_pos])
        pores.append(pore)

    # Subtract all pores from conduit
    if pores:
        from ..core import batch_union
        all_pores = batch_union(pores)
        conduit = conduit - all_pores

    return conduit


def _create_flared_ends(
    params: NerveConduitParams,
    outer_radius: float,
    inner_radius: float,
    length: float
) -> Tuple[m3d.Manifold, float]:
    """
    Create conduit with flared ends for easier anastomosis.

    Flared ends help with surgical insertion of nerve stumps.
    Typical: 15° angle, 1-2mm flare length.

    Args:
        params: Generation parameters
        outer_radius: Base outer radius in mm
        inner_radius: Base inner radius in mm
        length: Total length in mm

    Returns:
        Tuple of (conduit with flared ends, new effective outer radius)
    """
    flare_angle_rad = math.radians(params.flare_angle_deg)
    flare_length = params.flare_length_mm
    main_length = length - 2 * flare_length

    # Calculate flared radii
    flare_increase = flare_length * math.tan(flare_angle_rad)
    flared_outer_radius = outer_radius + flare_increase
    flared_inner_radius = inner_radius + flare_increase

    # Create three sections: bottom flare, main body, top flare
    segments = []

    # Bottom flare (z=0 to z=flare_length)
    # Outer cone: larger at z=0, regular at z=flare_length
    bottom_outer = m3d.Manifold.cylinder(
        flare_length,
        flared_outer_radius,  # bottom (z=0)
        outer_radius,          # top (z=flare_length)
        params.resolution * 2
    )
    bottom_inner = m3d.Manifold.cylinder(
        flare_length + 0.1,
        flared_inner_radius,
        inner_radius,
        params.resolution * 2
    ).translate([0, 0, -0.05])
    bottom_flare = bottom_outer - bottom_inner
    segments.append(bottom_flare)

    # Main body (z=flare_length to z=flare_length+main_length)
    main_outer = m3d.Manifold.cylinder(
        main_length,
        outer_radius,
        outer_radius,
        params.resolution * 2
    ).translate([0, 0, flare_length])
    main_inner = m3d.Manifold.cylinder(
        main_length + 0.1,
        inner_radius,
        inner_radius,
        params.resolution * 2
    ).translate([0, 0, flare_length - 0.05])
    main_body = main_outer - main_inner
    segments.append(main_body)

    # Top flare (z=flare_length+main_length to z=length)
    top_outer = m3d.Manifold.cylinder(
        flare_length,
        outer_radius,          # bottom
        flared_outer_radius,   # top (z=length)
        params.resolution * 2
    ).translate([0, 0, flare_length + main_length])
    top_inner = m3d.Manifold.cylinder(
        flare_length + 0.1,
        inner_radius,
        flared_inner_radius,
        params.resolution * 2
    ).translate([0, 0, flare_length + main_length - 0.05])
    top_flare = top_outer - top_inner
    segments.append(top_flare)

    # Union all segments
    from ..core import batch_union
    conduit = batch_union(segments)

    return conduit, flared_outer_radius


def generate_nerve_conduit(params: NerveConduitParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a nerve conduit scaffold with internal guidance microchannels.

    Creates a hollow outer tube with multiple internal parallel channels to guide
    axon regeneration. Supports comprehensive biological features:
    - Fascicle chambers with perineurium
    - Epineurium outer layer
    - Microgrooves for Schwann cell guidance
    - Growth factor reservoirs
    - Suture holes for surgical attachment
    - Flared ends for easier anastomosis
    - Spiral guidance channels for enhanced cellular activity
    - Bi-layered porous wall structure
    - Diameter tapering along length

    Args:
        params: NerveConduitParams with generation parameters

    Returns:
        Tuple of (manifold, stats_dict)

    Raises:
        ValueError: If channels don't fit within outer diameter
    """
    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(params.random_seed)

    # Apply taper ratio to calculate end dimensions
    outer_radius_start = params.outer_diameter_mm / 2
    outer_radius_end = outer_radius_start * params.taper_ratio

    inner_radius_start = outer_radius_start - params.wall_thickness_mm
    inner_radius_end = outer_radius_end - params.wall_thickness_mm

    # Use mid-length dimensions for most calculations
    outer_radius = (outer_radius_start + outer_radius_end) / 2
    inner_radius = (inner_radius_start + inner_radius_end) / 2

    channel_radius_mm = params.channel_diameter_um / 2000.0  # Convert um to mm
    guidance_radius_mm = params.guidance_channel_diameter_um / 2000.0  # Smaller guidance channels
    length = params.length_mm

    if inner_radius <= 0:
        raise ValueError("Wall thickness exceeds radius")

    # Check if channels fit
    if params.num_channels == 1:
        max_channel_radius = inner_radius * 0.8
    else:
        ring_radius = inner_radius * 0.5
        max_channel_radius = ring_radius * 0.4

    if channel_radius_mm > max_channel_radius:
        raise ValueError(
            f"Channels too large: {params.channel_diameter_um}um "
            f"exceeds max {max_channel_radius * 2000:.0f}um for {params.num_channels} channels"
        )

    # Track effective dimensions (may change with epineurium/flare)
    effective_outer_radius = outer_radius

    # ==========================================================================
    # STEP 1: Create base conduit (with or without flared ends)
    # ==========================================================================
    if params.enable_flared_ends:
        conduit, effective_outer_radius = _create_flared_ends(
            params, outer_radius, inner_radius, length
        )
    else:
        # Cylindrical conduit with optional taper
        if params.taper_ratio != 1.0:
            # Tapered conduit (outer and inner diameters change along length)
            outer = m3d.Manifold.cylinder(
                length,
                outer_radius_start,  # bottom radius
                outer_radius_end,     # top radius
                params.resolution * 2
            )
            inner = m3d.Manifold.cylinder(
                length + 0.2,
                inner_radius_start,
                inner_radius_end,
                params.resolution * 2
            ).translate([0, 0, -0.1])
            conduit = outer - inner
        else:
            # Standard non-tapered cylindrical conduit
            outer = m3d.Manifold.cylinder(
                length,
                outer_radius,
                outer_radius,
                params.resolution * 2
            )
            inner = m3d.Manifold.cylinder(
                length + 0.2,
                inner_radius,
                inner_radius,
                params.resolution * 2
            ).translate([0, 0, -0.1])
            conduit = outer - inner

    # ==========================================================================
    # STEP 2: Add epineurium outer layer if enabled
    # ==========================================================================
    if params.enable_epineurium:
        conduit = _create_epineurium(params, conduit, outer_radius, length)
        effective_outer_radius = outer_radius + params.epineurium_thickness_um / 1000.0

    # ==========================================================================
    # STEP 3: Create fascicle chambers if enabled
    # ==========================================================================
    fascicle_positions = []
    if params.enable_fascicle_chambers:
        conduit, fascicle_positions = _create_fascicle_chambers(
            params, conduit, inner_radius, length
        )

    # ==========================================================================
    # STEP 4: Create guidance channels (default microchannels)
    # ==========================================================================
    if params.enable_guidance_channels and not params.enable_fascicle_chambers:
        # Only add if fascicle chambers not enabled (they serve same purpose)
        channels = []

        # Use smaller guidance channels (guidance_channel_diameter_um) instead of main channel_diameter_um
        actual_channel_radius = guidance_radius_mm
        min_spacing_mm = params.channel_spacing_um / 1000.0  # Convert minimum spacing to mm

        if params.guidance_channel_pattern == "spiral":
            # Use spiral pattern for enhanced cellular activity
            helix_radius = inner_radius * 0.5
            channels = _create_spiral_channels(
                params, length, actual_channel_radius, helix_radius, rng
            )
        else:
            # Linear pattern (default)
            if params.num_channels == 1:
                channel = m3d.Manifold.cylinder(
                    length + 0.4,
                    actual_channel_radius,
                    actual_channel_radius,
                    params.resolution
                ).translate([0, 0, -0.2])
                channels.append(channel)
            else:
                ring_radius = inner_radius * 0.5

                # Store channel positions for spacing validation
                channel_positions = []

                # Central channel
                central = m3d.Manifold.cylinder(
                    length + 0.4,
                    actual_channel_radius,
                    actual_channel_radius,
                    params.resolution
                ).translate([0, 0, -0.2])
                channels.append(central)
                channel_positions.append((0.0, 0.0))

                # First ring
                n_ring = min(params.num_channels - 1, 18)
                for i in range(n_ring):
                    angle = 2 * np.pi * i / n_ring

                    # Apply position noise if enabled
                    noise_x = noise_y = 0.0
                    if params.position_noise_um > 0:
                        noise_x = (rng.random() - 0.5) * 2 * params.position_noise_um / 1000.0
                        noise_y = (rng.random() - 0.5) * 2 * params.position_noise_um / 1000.0

                    x = ring_radius * np.cos(angle) + noise_x
                    y = ring_radius * np.sin(angle) + noise_y

                    # Validate minimum spacing constraint
                    valid_position = True
                    for (px, py) in channel_positions:
                        distance = math.sqrt((x - px)**2 + (y - py)**2)
                        if distance < min_spacing_mm:
                            # Skip this channel if too close to existing channel
                            valid_position = False
                            break

                    if not valid_position:
                        continue

                    # Apply channel diameter variance
                    local_radius = actual_channel_radius
                    if params.channel_variance > 0:
                        variance_factor = 1.0 + (rng.random() - 0.5) * 2 * params.channel_variance
                        local_radius = actual_channel_radius * variance_factor

                    channel = m3d.Manifold.cylinder(
                        length + 0.4,
                        local_radius,
                        local_radius,
                        params.resolution
                    ).translate([x, y, -0.2])
                    channels.append(channel)
                    channel_positions.append((x, y))

                # Second ring if needed
                if params.num_channels > 19:
                    outer_ring_radius = inner_radius * 0.75
                    n_outer = min(params.num_channels - 19, 24)
                    for i in range(n_outer):
                        angle = 2 * np.pi * i / n_outer + np.pi / n_outer

                        # Apply position noise
                        noise_x = noise_y = 0.0
                        if params.position_noise_um > 0:
                            noise_x = (rng.random() - 0.5) * 2 * params.position_noise_um / 1000.0
                            noise_y = (rng.random() - 0.5) * 2 * params.position_noise_um / 1000.0

                        x = outer_ring_radius * np.cos(angle) + noise_x
                        y = outer_ring_radius * np.sin(angle) + noise_y

                        # Validate minimum spacing constraint
                        valid_position = True
                        for (px, py) in channel_positions:
                            distance = math.sqrt((x - px)**2 + (y - py)**2)
                            if distance < min_spacing_mm:
                                valid_position = False
                                break

                        if not valid_position:
                            continue

                        # Apply channel diameter variance
                        local_radius = actual_channel_radius
                        if params.channel_variance > 0:
                            variance_factor = 1.0 + (rng.random() - 0.5) * 2 * params.channel_variance
                            local_radius = actual_channel_radius * variance_factor

                        channel = m3d.Manifold.cylinder(
                            length + 0.4,
                            local_radius,
                            local_radius,
                            params.resolution
                        ).translate([x, y, -0.2])
                        channels.append(channel)
                        channel_positions.append((x, y))

        from ..core import batch_union
        if channels:
            all_channels = batch_union(channels)
            conduit = conduit - all_channels

    # ==========================================================================
    # STEP 5: Add microgrooves for Schwann cell guidance
    # ==========================================================================
    if params.enable_microgrooves:
        conduit = _create_microgrooves(params, conduit, inner_radius, length)

    # ==========================================================================
    # STEP 6: Add growth factor reservoirs
    # ==========================================================================
    if params.enable_growth_factor_reservoirs:
        conduit = _create_growth_factor_reservoirs(
            params, conduit, outer_radius, inner_radius, length
        )

    # ==========================================================================
    # STEP 7: Add wall porosity
    # ==========================================================================
    if params.wall_porosity > 0 and params.pore_size_um > 0:
        conduit = _create_wall_pores(
            params, conduit, outer_radius, inner_radius, length, rng
        )

    # ==========================================================================
    # STEP 8: Add suture holes
    # ==========================================================================
    if params.enable_biodegradable_suture_holes:
        conduit = _create_suture_holes(params, conduit, outer_radius, length)

    # ==========================================================================
    # Calculate statistics
    # ==========================================================================
    mesh = conduit.to_mesh()
    volume = conduit.volume() if hasattr(conduit, 'volume') else 0

    # Count channels
    num_channels_created = params.num_channels if params.enable_guidance_channels else 0
    if params.enable_fascicle_chambers:
        num_channels_created = len(fascicle_positions)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'outer_diameter_mm': effective_outer_radius * 2,
        'inner_diameter_mm': params.inner_diameter_mm,
        'length_mm': length,
        'num_channels': num_channels_created,
        'channel_diameter_um': params.channel_diameter_um,
        'channel_spacing_um': params.channel_spacing_um,
        'wall_thickness_mm': params.wall_thickness_mm,
        'wall_porosity': params.wall_porosity,
        # Fascicle features
        'enable_fascicle_chambers': params.enable_fascicle_chambers,
        'num_fascicles': len(fascicle_positions) if params.enable_fascicle_chambers else 0,
        'fascicle_diameter_um': params.fascicle_diameter_um,
        # Tissue layers
        'enable_perineurium': params.enable_perineurium,
        'perineurium_thickness_um': params.perineurium_thickness_um,
        'enable_epineurium': params.enable_epineurium,
        'epineurium_thickness_um': params.epineurium_thickness_um,
        # Surface features
        'enable_microgrooves': params.enable_microgrooves,
        'groove_width_um': params.groove_width_um,
        'groove_depth_um': params.groove_depth_um,
        'ridge_width_um': params.ridge_width_um,
        'groove_spacing_um': params.groove_spacing_um,
        # Guidance channels
        'enable_guidance_channels': params.enable_guidance_channels,
        'guidance_channel_pattern': params.guidance_channel_pattern,
        'guidance_channel_diameter_um': params.guidance_channel_diameter_um,
        # Wall porosity structure
        'pore_size_um': params.pore_size_um,
        'inner_surface_porosity': params.inner_surface_porosity,
        # Drug delivery
        'enable_growth_factor_reservoirs': params.enable_growth_factor_reservoirs,
        'reservoir_diameter_um': params.reservoir_diameter_um,
        'reservoir_count': params.reservoir_count,
        # Surgical features
        'enable_suture_holes': params.enable_biodegradable_suture_holes,
        'suture_hole_diameter_mm': params.suture_hole_diameter_mm,
        'suture_hole_count': params.suture_hole_count,
        # End geometry
        'enable_flared_ends': params.enable_flared_ends,
        'flare_angle_deg': params.flare_angle_deg,
        'flare_length_mm': params.flare_length_mm,
        'taper_ratio': params.taper_ratio,
        # Randomization parameters
        'position_noise_um': params.position_noise_um,
        'channel_variance': params.channel_variance,
        'random_seed': params.random_seed,
        'scaffold_type': 'nerve_conduit'
    }

    return conduit, stats


def generate_nerve_conduit_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate a nerve conduit scaffold from dictionary parameters.

    Args:
        params: Dictionary of parameters (will be unpacked into NerveConduitParams)

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle parameter name mappings for compatibility
    param_copy = params.copy()

    # Map legacy parameter names
    if 'channel_count' in param_copy and 'num_channels' not in param_copy:
        param_copy['num_channels'] = param_copy.pop('channel_count')

    # Filter to only valid NerveConduitParams fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(NerveConduitParams)}
    filtered_params = {k: v for k, v in param_copy.items() if k in valid_fields}

    return generate_nerve_conduit(NerveConduitParams(**filtered_params))
