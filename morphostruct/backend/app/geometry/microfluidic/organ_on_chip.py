"""
Organ-on-chip microfluidic scaffold generator.

Creates microfluidic channels with tissue chambers for organ-on-chip applications.
Features:
- Configurable main and microchannels leading to tissue chambers
- Rectangular tissue chambers (larger cavities for cell culture)
- Porous membranes between chambers for co-culture
- Multiple inlet/outlet ports for perfusion
- Multi-layer chip architecture
- Biologically realistic dimensions based on published organ-on-chip designs
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from ..core import batch_union


@dataclass
class OrganOnChipParams:
    """
    Parameters for organ-on-chip microfluidic scaffold generation.

    Based on standard organ-on-chip design principles and commercial platforms.
    Dimensions follow typical microfluidic fabrication constraints.
    """
    # === Main Channel Geometry ===
    main_channel_width_um: float = 200.0  # 50-500 um typical
    main_channel_height_um: float = 50.0  # 25-200 um typical
    main_channel_length_mm: float = 15.0  # 5-30 mm typical

    # === Microchannel Network ===
    microchannel_width_um: float = 130.0  # 50-200 um for cell guidance
    microchannel_height_um: float = 30.0  # 20-100 um typical
    enable_microchannels: bool = False
    microchannel_count: int = 8  # Per chamber
    microchannel_spacing_um: float = 200.0  # Center-to-center

    # === Chamber Configuration ===
    chamber_width_mm: float = 12.0  # 1-20 mm typical
    chamber_height_um: float = 50.0  # 25-150 um for thin tissue
    chamber_length_mm: float = 5.0  # 2-15 mm typical
    chamber_volume_nl: float = 500.0  # Informational: ~500 nL typical
    num_chambers: int = 4  # 1-96 for high-throughput
    chamber_spacing_mm: float = 3.0  # Between chambers

    # === Membrane Configuration ===
    enable_membrane: bool = False
    membrane_thickness_um: float = 12.0  # 5-50 um PET/PDMS membranes
    membrane_pore_size_um: float = 0.4  # 0.1-8.0 um for cell separation
    membrane_pore_density_per_cm2: float = 1e5  # Pores per cm^2
    membrane_porosity: float = 0.1  # Calculated void fraction

    # === Inlet/Outlet Ports ===
    num_inlets: int = 2  # 1-8 typical
    num_outlets: int = 2  # 1-8 typical
    port_diameter_mm: float = 1.6  # mm (1/16" Mini Luer standard)
    port_depth_mm: float = 2.0  # Chip thickness at ports

    # === Multi-layer Architecture ===
    num_layers: int = 2  # 1-4 for stacked organ-on-chip
    layer_height_mm: float = 1.0  # Per PDMS layer
    enable_interlayer_vias: bool = False
    via_diameter_mm: float = 0.5

    # === Flow Parameters (informational/design) ===
    design_flow_rate_ul_min: float = 30.0  # 1-100 uL/min typical
    design_shear_stress_pa: float = 0.5  # 0.1-10 Pa physiological

    # === Cell Seeding ===
    cell_seeding_density_per_cm2: float = 1e5  # 1e4-1e6 cells/cm^2
    enable_cell_traps: bool = False
    trap_diameter_um: float = 30.0  # For single-cell capture
    trap_count: int = 100

    # === Chip Dimensions ===
    chip_length_mm: float = 25.0  # Standard 25x75 microscope slide
    chip_width_mm: float = 15.0
    chip_thickness_mm: float = 3.0  # Total chip height

    # === Quality/Resolution ===
    resolution: int = 8
    seed: int = 42


def make_channel(
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    depth: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a rectangular channel between two points.

    Args:
        start: Starting point (x, y, z)
        end: Ending point (x, y, z)
        width: Channel width (cross-section)
        depth: Channel depth (cross-section)
        resolution: Mesh resolution

    Returns:
        Manifold representing the channel volume
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create rectangular channel along X axis
    channel = m3d.Manifold.cube([length, width, depth], center=True)
    channel = channel.translate([length/2, 0, 0])

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)  # horizontal distance

    if h > 0.001 or abs(dz) > 0.001:
        # Rotate to align with direction
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        angle_y = np.arctan2(-dz, h) * 180 / np.pi

        channel = channel.rotate([0, 0, angle_z])
        channel = channel.rotate([0, angle_y, 0])

    return channel.translate([start[0], start[1], start[2]])


def make_chamber(
    center: np.ndarray,
    width: float,
    length: float,
    height: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a rectangular tissue chamber.

    Args:
        center: Center point (x, y, z)
        width: Chamber width (Y dimension)
        length: Chamber length (X dimension)
        height: Chamber height (Z dimension)
        resolution: Mesh resolution

    Returns:
        Manifold representing the chamber volume
    """
    chamber = m3d.Manifold.cube([length, width, height], center=True)
    return chamber.translate([center[0], center[1], center[2]])


def make_port(
    position: np.ndarray,
    diameter: float,
    depth: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a cylindrical inlet/outlet port.

    Args:
        position: Port center position at chip surface
        diameter: Port diameter
        depth: Port depth into chip
        resolution: Mesh resolution

    Returns:
        Manifold representing the port volume
    """
    port = m3d.Manifold.cylinder(depth, diameter/2, diameter/2, resolution)
    # Cylinder is along Z axis, translate so top is at position
    return port.translate([position[0], position[1], position[2] - depth/2])


def make_membrane_with_pores(
    center: np.ndarray,
    width: float,
    length: float,
    thickness: float,
    pore_size: float,
    pore_density_per_cm2: float,
    resolution: int
) -> tuple[m3d.Manifold, int]:
    """
    Create a porous membrane with hexagonally-packed pores.

    Based on literature from Emulate, Nature Scientific Reports, and PMC studies:
    - Thickness: 10-50 um (default 12 um for PET-like, 50 um for PDMS-like)
    - Pore size: 0.4 um (cell separation) to 7-8 um (cell migration)
    - Pore density: 1e6 to 1e8 pores/cm2 (default 1e6 for ALI culture)

    Args:
        center: Center point (x, y, z)
        width: Membrane width (Y dimension) in mm
        length: Membrane length (X dimension) in mm
        thickness: Membrane thickness (Z dimension) in mm
        pore_size: Pore diameter in mm
        pore_density_per_cm2: Pores per cm^2
        resolution: Mesh resolution

    Returns:
        Tuple of (membrane manifold, pore count)
    """
    # Create solid membrane slab
    membrane = m3d.Manifold.cube([length, width, thickness], center=True)

    # Convert pore density from per cm^2 to per mm^2
    pore_density_per_mm2 = pore_density_per_cm2 / 100.0

    # Calculate pore spacing from density (hexagonal packing)
    # For density D pores/mm^2, spacing = sqrt(2/(D*sqrt(3))) mm
    if pore_density_per_mm2 > 0 and pore_size > 0:
        spacing = np.sqrt(2.0 / (pore_density_per_mm2 * np.sqrt(3)))
    else:
        # No pores - return solid membrane
        return membrane.translate([center[0], center[1], center[2]]), 0

    # Ensure minimum spacing to avoid overlapping pores
    spacing = max(spacing, pore_size * 1.5)

    # Generate pore grid with hexagonal packing
    pores = []
    pore_radius = pore_size / 2

    # Calculate grid dimensions with margin from edges
    margin = pore_size * 2
    usable_length = length - 2 * margin
    usable_width = width - 2 * margin

    if usable_length <= 0 or usable_width <= 0:
        # Membrane too small for pores
        return membrane.translate([center[0], center[1], center[2]]), 0

    nx = max(1, int(usable_length / spacing))
    ny = max(1, int(usable_width / (spacing * np.sqrt(3) / 2)))

    for j in range(ny):
        # Hexagonal offset for even rows
        x_offset = (spacing / 2) if j % 2 == 1 else 0
        y = -usable_width / 2 + spacing * np.sqrt(3) / 2 * j

        for i in range(nx):
            x = -usable_length / 2 + spacing / 2 + i * spacing + x_offset

            # Ensure pore is within bounds
            if abs(x) < usable_length / 2 and abs(y) < usable_width / 2:
                # Create cylindrical pore through membrane
                pore = m3d.Manifold.cylinder(
                    thickness * 1.2,  # Slightly longer to ensure clean boolean
                    pore_radius,
                    pore_radius,
                    max(6, resolution // 2)
                )
                pore = pore.translate([x, y, 0])
                pores.append(pore)

    pore_count = len(pores)
    if pores:
        pores_union = batch_union(pores)
        membrane = membrane - pores_union

    return membrane.translate([center[0], center[1], center[2]]), pore_count


def make_cell_trap_array(
    chamber_center: np.ndarray,
    chamber_width: float,
    chamber_length: float,
    chamber_height: float,
    trap_diameter: float,
    trap_count: int,
    resolution: int
) -> tuple[m3d.Manifold, int]:
    """
    Create an array of cell traps in the chamber floor.

    Based on PMC microfluidic cell trapping studies:
    - Trap diameter: 15-30 um (default 20 um for mammalian cells ~10-15 um)
    - Trap depth: 20-25 um (partial depth into chamber floor)
    - Trap geometry: Cylindrical wells
    - Arrangement: Grid pattern in chamber floor

    Args:
        chamber_center: Center of chamber (x, y, z)
        chamber_width: Chamber width (Y dimension) in mm
        chamber_length: Chamber length (X dimension) in mm
        chamber_height: Chamber height (Z dimension) in mm
        trap_diameter: Trap diameter in mm
        trap_count: Number of traps to create
        resolution: Mesh resolution

    Returns:
        Tuple of (traps manifold for subtraction, actual trap count)
    """
    if trap_count <= 0 or trap_diameter <= 0:
        return m3d.Manifold(), 0

    # Trap depth is 20-25 um, but scale with chamber height
    trap_depth = min(0.025, chamber_height * 0.5)
    trap_radius = trap_diameter / 2

    # Arrange traps in grid within 80% of chamber area
    usable_width = chamber_width * 0.8
    usable_length = chamber_length * 0.8

    # Calculate grid dimensions based on trap count and aspect ratio
    aspect = usable_length / usable_width if usable_width > 0 else 1.0
    ny = max(1, int(np.sqrt(trap_count / aspect)))
    nx = max(1, int(trap_count / ny))

    # Recalculate to get closer to requested trap count
    while nx * ny < trap_count and nx < 100 and ny < 100:
        if nx <= ny:
            nx += 1
        else:
            ny += 1

    spacing_x = usable_length / (nx + 1) if nx > 0 else usable_length
    spacing_y = usable_width / (ny + 1) if ny > 0 else usable_width

    # Ensure traps don't overlap
    min_spacing = trap_diameter * 2
    spacing_x = max(spacing_x, min_spacing)
    spacing_y = max(spacing_y, min_spacing)

    traps = []
    actual_count = 0

    for i in range(nx):
        for j in range(ny):
            if actual_count >= trap_count:
                break

            x = chamber_center[0] - usable_length / 2 + spacing_x * (i + 1)
            y = chamber_center[1] - usable_width / 2 + spacing_y * (j + 1)
            # Position trap at bottom of chamber
            z = chamber_center[2] - chamber_height / 2 - trap_depth / 2

            trap = m3d.Manifold.cylinder(
                trap_depth,
                trap_radius,
                trap_radius,
                max(8, resolution)
            )
            trap = trap.translate([x, y, z])
            traps.append(trap)
            actual_count += 1

        if actual_count >= trap_count:
            break

    if traps:
        return batch_union(traps), actual_count
    return m3d.Manifold(), 0


def make_interlayer_vias(
    chamber_centers: list,
    chamber_width: float,
    chamber_length: float,
    via_diameter: float,
    layer_height: float,
    num_layers: int,
    chip_thickness: float,
    resolution: int
) -> tuple[m3d.Manifold, int]:
    """
    Create interlayer vias connecting multiple chip layers.

    Based on multilayer PDMS fabrication literature:
    - Via diameter: 50-100 um for control, 100-500 um for flow (default 0.3-0.5 mm)
    - Placement: At chamber corners and inlet/outlet connections
    - Purpose: Fluidic connection between layers

    Args:
        chamber_centers: List of chamber center positions
        chamber_width: Chamber width (Y dimension) in mm
        chamber_length: Chamber length (X dimension) in mm
        via_diameter: Via diameter in mm
        layer_height: Height between layers in mm
        num_layers: Number of layers
        chip_thickness: Base chip thickness in mm
        resolution: Mesh resolution

    Returns:
        Tuple of (vias manifold for subtraction, via count)
    """
    if num_layers <= 1 or via_diameter <= 0:
        return m3d.Manifold(), 0

    vias = []
    via_radius = via_diameter / 2

    # Total via height spans all layers plus extra for clean boolean
    total_via_height = chip_thickness + (num_layers - 1) * layer_height + 0.2

    for center in chamber_centers:
        # Place vias at 4 corners of each chamber (80% from center to edge)
        corner_offset_x = chamber_length / 2 * 0.8
        corner_offset_y = chamber_width / 2 * 0.8

        corners = [
            (center[0] - corner_offset_x, center[1] - corner_offset_y),
            (center[0] + corner_offset_x, center[1] - corner_offset_y),
            (center[0] - corner_offset_x, center[1] + corner_offset_y),
            (center[0] + corner_offset_x, center[1] + corner_offset_y),
        ]

        for cx, cy in corners:
            via = m3d.Manifold.cylinder(
                total_via_height,
                via_radius,
                via_radius,
                resolution
            )
            # Center via vertically in chip
            via = via.translate([cx, cy, 0])
            vias.append(via)

    via_count = len(vias)
    if vias:
        return batch_union(vias), via_count
    return m3d.Manifold(), 0


def generate_organ_on_chip(params: OrganOnChipParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an organ-on-chip microfluidic scaffold.

    Creates a solid chip with microfluidic channels and tissue chambers
    carved out as negative space. Supports multi-chamber designs with
    membrane interfaces, multi-layer architecture, interlayer vias, and
    cell traps.

    Args:
        params: OrganOnChipParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry (solid chip with channels removed)
        - stats_dict: Dictionary with triangle_count, volume_mm3, channel_count,
                     chamber_count, membrane_count, via_count, trap_count, etc.

    Raises:
        ValueError: If no channels are generated
    """
    rng = np.random.default_rng(params.seed)

    # Convert um to mm
    main_channel_width = params.main_channel_width_um / 1000.0
    main_channel_height = params.main_channel_height_um / 1000.0
    microchannel_width = params.microchannel_width_um / 1000.0
    microchannel_height = params.microchannel_height_um / 1000.0
    chamber_height = params.chamber_height_um / 1000.0
    membrane_thickness = params.membrane_thickness_um / 1000.0
    membrane_pore_size = params.membrane_pore_size_um / 1000.0
    trap_diameter = params.trap_diameter_um / 1000.0

    chip_x = params.chip_length_mm
    chip_y = params.chip_width_mm
    # Adjust chip_z for multi-layer architecture
    chip_z = params.chip_thickness_mm + (params.num_layers - 1) * params.layer_height_mm

    # Create solid chip base
    chip_base = m3d.Manifold.cube([chip_x, chip_y, chip_z], center=True)

    # Calculate chamber positions (evenly spaced along X axis)
    total_chamber_span = params.num_chambers * params.chamber_length_mm + \
                        (params.num_chambers - 1) * params.chamber_spacing_mm
    start_x = -total_chamber_span / 2 + params.chamber_length_mm / 2

    # Base chamber centers (for layer 0)
    base_chamber_centers = []
    for i in range(params.num_chambers):
        x_pos = start_x + i * (params.chamber_length_mm + params.chamber_spacing_mm)
        base_chamber_centers.append(np.array([x_pos, 0.0, 0.0]))

    # Track statistics for new features
    membrane_count = 0
    total_pore_count = 0
    via_count = 0
    actual_trap_count = 0

    # ===== BUILD FEATURES FOR ALL LAYERS =====
    all_chambers = []
    all_inlet_channels = []
    all_inter_channels = []
    all_outlet_channels = []
    all_membranes = []
    all_cell_traps = []

    inlet_spacing = chip_y / (params.num_inlets + 1)
    outlet_spacing = chip_y / (params.num_outlets + 1)

    for layer_idx in range(params.num_layers):
        # Z offset for this layer (centered around 0)
        layer_z_offset = layer_idx * params.layer_height_mm - (params.num_layers - 1) * params.layer_height_mm / 2

        # Chamber centers for this layer
        layer_chamber_centers = [
            np.array([c[0], c[1], layer_z_offset]) for c in base_chamber_centers
        ]

        # Create chambers for this layer
        for center in layer_chamber_centers:
            chamber = make_chamber(
                center,
                params.chamber_width_mm,
                params.chamber_length_mm,
                chamber_height,
                params.resolution
            )
            all_chambers.append(chamber)

        # Create inlet channels from left edge to first chamber
        for i in range(params.num_inlets):
            y_pos = -chip_y / 2 + inlet_spacing * (i + 1)
            start = np.array([-chip_x / 2, y_pos, layer_z_offset])
            end = np.array([layer_chamber_centers[0][0] - params.chamber_length_mm / 2, y_pos, layer_z_offset])

            channel = make_channel(
                start, end,
                main_channel_width,
                main_channel_height,
                params.resolution
            )
            all_inlet_channels.append(channel)

        # Create inter-chamber channels
        for i in range(len(layer_chamber_centers) - 1):
            start = np.array([
                layer_chamber_centers[i][0] + params.chamber_length_mm / 2,
                0.0,
                layer_z_offset
            ])
            end = np.array([
                layer_chamber_centers[i + 1][0] - params.chamber_length_mm / 2,
                0.0,
                layer_z_offset
            ])

            channel = make_channel(
                start, end,
                main_channel_width,
                main_channel_height,
                params.resolution
            )
            all_inter_channels.append(channel)

        # Create outlet channels from last chamber to right edge
        for i in range(params.num_outlets):
            y_pos = -chip_y / 2 + outlet_spacing * (i + 1)
            start = np.array([layer_chamber_centers[-1][0] + params.chamber_length_mm / 2, y_pos, layer_z_offset])
            end = np.array([chip_x / 2, y_pos, layer_z_offset])

            channel = make_channel(
                start, end,
                main_channel_width,
                main_channel_height,
                params.resolution
            )
            all_outlet_channels.append(channel)

        # Create membranes between adjacent chambers if enabled
        if params.enable_membrane and len(layer_chamber_centers) > 1:
            for i in range(len(layer_chamber_centers) - 1):
                # Position membrane at midpoint between chambers
                membrane_x = (layer_chamber_centers[i][0] + layer_chamber_centers[i + 1][0]) / 2
                membrane_center = np.array([membrane_x, 0.0, layer_z_offset])

                # Membrane spans the gap between chambers
                membrane_length = params.chamber_spacing_mm * 0.8  # 80% of gap
                membrane_width = params.chamber_width_mm * 0.9  # 90% of chamber width

                membrane, pore_count = make_membrane_with_pores(
                    membrane_center,
                    membrane_width,
                    membrane_length,
                    membrane_thickness,
                    membrane_pore_size,
                    params.membrane_pore_density_per_cm2,
                    params.resolution
                )
                all_membranes.append(membrane)
                membrane_count += 1
                total_pore_count += pore_count

        # Create cell traps in chamber floor if enabled
        if params.enable_cell_traps and params.trap_count > 0:
            for center in layer_chamber_centers:
                traps, trap_count_in_chamber = make_cell_trap_array(
                    center,
                    params.chamber_width_mm,
                    params.chamber_length_mm,
                    chamber_height,
                    trap_diameter,
                    params.trap_count,
                    params.resolution
                )
                if trap_count_in_chamber > 0:
                    all_cell_traps.append(traps)
                    actual_trap_count += trap_count_in_chamber

    # Create inlet/outlet ports at chip surface (only once, at top)
    ports = []
    port_z = chip_z / 2  # Top of chip

    # Inlet ports
    for i in range(params.num_inlets):
        y_pos = -chip_y / 2 + inlet_spacing * (i + 1)
        port_pos = np.array([-chip_x / 2 + params.port_diameter_mm, y_pos, port_z])
        port = make_port(port_pos, params.port_diameter_mm, params.port_depth_mm, params.resolution)
        ports.append(port)

    # Outlet ports
    for i in range(params.num_outlets):
        y_pos = -chip_y / 2 + outlet_spacing * (i + 1)
        port_pos = np.array([chip_x / 2 - params.port_diameter_mm, y_pos, port_z])
        port = make_port(port_pos, params.port_diameter_mm, params.port_depth_mm, params.resolution)
        ports.append(port)

    # Create microchannels if enabled (for all layers)
    microchannels = []
    if params.enable_microchannels and params.microchannel_count > 0:
        microchannel_spacing = params.microchannel_spacing_um / 1000.0
        for layer_idx in range(params.num_layers):
            layer_z_offset = layer_idx * params.layer_height_mm - (params.num_layers - 1) * params.layer_height_mm / 2
            for base_center in base_chamber_centers:
                center = np.array([base_center[0], base_center[1], layer_z_offset])
                total_width = (params.microchannel_count - 1) * microchannel_spacing
                for j in range(params.microchannel_count):
                    y_offset = -total_width / 2 + j * microchannel_spacing
                    mc_start = np.array([
                        center[0] - params.chamber_length_mm / 2,
                        center[1] + y_offset,
                        center[2]
                    ])
                    mc_end = np.array([
                        center[0] + params.chamber_length_mm / 2,
                        center[1] + y_offset,
                        center[2]
                    ])
                    mc = make_channel(
                        mc_start, mc_end,
                        microchannel_width,
                        microchannel_height,
                        params.resolution
                    )
                    microchannels.append(mc)

    # Create interlayer vias if enabled and multi-layer
    vias_manifold = None
    if params.enable_interlayer_vias and params.num_layers > 1:
        vias_manifold, via_count = make_interlayer_vias(
            base_chamber_centers,
            params.chamber_width_mm,
            params.chamber_length_mm,
            params.via_diameter_mm,
            params.layer_height_mm,
            params.num_layers,
            params.chip_thickness_mm,
            params.resolution
        )

    # Combine all channel/chamber/port features
    all_features = (
        all_chambers +
        all_inlet_channels +
        all_inter_channels +
        all_outlet_channels +
        ports +
        microchannels
    )

    if not all_features:
        raise ValueError("No channels or chambers generated")

    # Union all features
    features_combined = batch_union(all_features)

    # Subtract from chip base (create negative space for channels/chambers)
    result = chip_base - features_combined

    # Add membranes as solid features (they stay in the chip)
    if all_membranes:
        membranes_combined = batch_union(all_membranes)
        result = result + membranes_combined

    # Subtract cell traps (they are wells in the chamber floor)
    if all_cell_traps:
        traps_combined = batch_union(all_cell_traps)
        result = result - traps_combined

    # Subtract interlayer vias
    if vias_manifold is not None and via_count > 0:
        result = result - vias_manifold

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    total_channels = len(all_inlet_channels) + len(all_inter_channels) + len(all_outlet_channels)

    # Calculate chamber surface area for cell seeding (per layer)
    chamber_surface_area_cm2 = (
        params.chamber_width_mm * params.chamber_length_mm / 100.0
    ) * params.num_chambers * params.num_layers

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'channel_count': total_channels,
        'microchannel_count': len(microchannels),
        'chamber_count': params.num_chambers * params.num_layers,
        'port_count': len(ports),
        'chamber_surface_area_cm2': chamber_surface_area_cm2,
        'estimated_cell_count': int(chamber_surface_area_cm2 * params.cell_seeding_density_per_cm2),
        # New stats for implemented features
        'total_layers': params.num_layers,
        'membrane_count': membrane_count,
        'membrane_pore_count': total_pore_count,
        'via_count': via_count,
        'trap_count': actual_trap_count,
        'scaffold_type': 'organ_on_chip'
    }

    return result, stats


def generate_organ_on_chip_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate organ-on-chip from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching OrganOnChipParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Build params object with comprehensive parameter mapping
    p = OrganOnChipParams(
        # Main channel geometry
        main_channel_width_um=params.get('main_channel_width_um', params.get('main_channel_width', 200.0)),
        main_channel_height_um=params.get('main_channel_height_um', params.get('main_channel_height', 50.0)),
        main_channel_length_mm=params.get('main_channel_length_mm', params.get('main_channel_length', 15.0)),

        # Microchannel network
        microchannel_width_um=params.get('microchannel_width_um', params.get('microchannel_width', 130.0)),
        microchannel_height_um=params.get('microchannel_height_um', params.get('microchannel_height', 30.0)),
        enable_microchannels=params.get('enable_microchannels', False),
        microchannel_count=params.get('microchannel_count', 8),
        microchannel_spacing_um=params.get('microchannel_spacing_um', params.get('microchannel_spacing', 200.0)),

        # Chamber configuration
        chamber_width_mm=params.get('chamber_width_mm', params.get('chamber_width', 12.0)),
        chamber_height_um=params.get('chamber_height_um', params.get('chamber_height', 50.0)),
        chamber_length_mm=params.get('chamber_length_mm', params.get('chamber_length', 5.0)),
        chamber_volume_nl=params.get('chamber_volume_nl', params.get('chamber_volume', 500.0)),
        num_chambers=params.get('num_chambers', params.get('chamber_count', 4)),
        chamber_spacing_mm=params.get('chamber_spacing_mm', params.get('chamber_spacing', 3.0)),

        # Membrane configuration
        enable_membrane=params.get('enable_membrane', False),
        membrane_thickness_um=params.get('membrane_thickness_um', params.get('membrane_thickness', 12.0)),
        membrane_pore_size_um=params.get('membrane_pore_size_um', params.get('membrane_pore_size', 0.4)),
        membrane_pore_density_per_cm2=params.get('membrane_pore_density_per_cm2', params.get('membrane_pore_density', 1e5)),
        membrane_porosity=params.get('membrane_porosity', 0.1),

        # Inlet/outlet ports
        num_inlets=params.get('num_inlets', params.get('inlet_count', 2)),
        num_outlets=params.get('num_outlets', params.get('outlet_count', 2)),
        port_diameter_mm=params.get('port_diameter_mm', params.get('port_diameter', 1.6)),
        port_depth_mm=params.get('port_depth_mm', params.get('port_depth', 2.0)),

        # Multi-layer architecture
        num_layers=params.get('num_layers', params.get('layer_count', 2)),
        layer_height_mm=params.get('layer_height_mm', params.get('layer_height', 1.0)),
        enable_interlayer_vias=params.get('enable_interlayer_vias', False),
        via_diameter_mm=params.get('via_diameter_mm', params.get('via_diameter', 0.5)),

        # Flow parameters
        design_flow_rate_ul_min=params.get('design_flow_rate_ul_min', params.get('flow_rate', 30.0)),
        design_shear_stress_pa=params.get('design_shear_stress_pa', params.get('shear_stress', 0.5)),

        # Cell seeding
        cell_seeding_density_per_cm2=params.get('cell_seeding_density_per_cm2', params.get('cell_seeding_density', 1e5)),
        enable_cell_traps=params.get('enable_cell_traps', False),
        trap_diameter_um=params.get('trap_diameter_um', params.get('trap_diameter', 30.0)),
        trap_count=params.get('trap_count', 100),

        # Chip dimensions
        chip_length_mm=params.get('chip_length_mm', params.get('chip_length', 25.0)),
        chip_width_mm=params.get('chip_width_mm', params.get('chip_width', 15.0)),
        chip_thickness_mm=params.get('chip_thickness_mm', params.get('chip_thickness', 3.0)),

        # Quality/Resolution
        resolution=params.get('resolution', 8),
        seed=params.get('seed', params.get('random_seed', 42)),
    )

    # Handle legacy parameter names
    if 'channel_width_mm' in params:
        p.main_channel_width_um = params['channel_width_mm'] * 1000
    if 'channel_depth_mm' in params:
        p.main_channel_height_um = params['channel_depth_mm'] * 1000
    if 'chamber_size_mm' in params:
        chamber_size = params['chamber_size_mm']
        if isinstance(chamber_size, dict):
            p.chamber_length_mm = chamber_size.get('x', 5.0)
            p.chamber_width_mm = chamber_size.get('y', 12.0)
            p.chamber_height_um = chamber_size.get('z', 0.05) * 1000
        elif isinstance(chamber_size, (list, tuple)):
            p.chamber_length_mm = chamber_size[0]
            p.chamber_width_mm = chamber_size[1]
            p.chamber_height_um = chamber_size[2] * 1000
    if 'chip_size_mm' in params:
        chip_size = params['chip_size_mm']
        if isinstance(chip_size, dict):
            p.chip_length_mm = chip_size.get('x', 25.0)
            p.chip_width_mm = chip_size.get('y', 15.0)
            p.chip_thickness_mm = chip_size.get('z', 3.0)
        elif isinstance(chip_size, (list, tuple)):
            p.chip_length_mm = chip_size[0]
            p.chip_width_mm = chip_size[1]
            p.chip_thickness_mm = chip_size[2]

    return generate_organ_on_chip(p)
