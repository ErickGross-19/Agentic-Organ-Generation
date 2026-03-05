"""
Cornea scaffold generator with curved dome geometry.

Provides parametric generation of cornea tissue scaffolds with:
- Prolate ellipsoid shape with asphericity (Q < 0 = flatter at periphery)
- 5 anatomical layers: epithelium, Bowman's layer, stroma, Descemet's membrane, endothelium
- Stromal lamellae architecture (up to 300 layers)
- Limbal zone with palisades of Vogt
- Nerve plexus with radial and whorl patterns
- Keratocyte distribution markers
- Transparency-controlled feature density
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ..core import batch_union


@dataclass
class CorneaParams:
    """
    Parameters for cornea scaffold generation.

    Biologically realistic parameters based on human corneal anatomy:
    - Total thickness: ~520 μm (central), thicker at periphery
    - Five distinct layers: epithelium, Bowman's layer, stroma, Descemet's membrane, endothelium
    - Stroma: 90% of thickness, ~200-300 lamellae of collagen fibrils
    - Collagen fibrils: ~30 nm diameter, ~60 nm spacing
    """
    # === Basic Geometry ===
    diameter_mm: float = 11.5  # 10-12 mm typical (adult)
    radius_of_curvature_mm: float = 7.8  # 7.5-8.0 mm anterior surface

    # === Total Thickness ===
    total_thickness_um: float = 520.0  # 500-560 μm central, thicker peripherally

    # === Layer Thicknesses (from anterior to posterior) ===
    # Epithelium (stratified squamous epithelium)
    epithelium_thickness_um: float = 50.0  # 50-52 μm, 5-6 cell layers

    # Bowman's layer (acellular condensation of anterior stroma)
    bowmans_layer_thickness_um: float = 12.0  # 8-14 μm

    # Stroma (collagen lamellae, keratocytes)
    stroma_thickness_um: float = 500.0  # ~90% of corneal thickness

    # Descemet's membrane (basement membrane of endothelium)
    descemet_thickness_um: float = 8.0  # 3-10 μm, thickens with age

    # Endothelium (single layer of hexagonal cells)
    endothelium_thickness_um: float = 5.0  # 4-6 μm

    # === Stromal Architecture ===
    num_lamellae: int = 300  # 200-500 lamellae in human cornea
    lamella_thickness_um: float = 2.0  # 1.5-2.5 μm each
    lamella_angle_variation_deg: float = 90.0  # Orthogonal arrangement

    # === Collagen Fibril Parameters ===
    collagen_fibril_diameter_nm: float = 30.0  # ~31 nm uniform diameter. REFERENCE ONLY: Nanometer scale not modeled at scaffold resolution
    collagen_fibril_spacing_nm: float = 42.0  # ~62 nm center-to-center. REFERENCE ONLY: Nanometer scale not modeled at scaffold resolution

    # === Keratocyte Distribution ===
    keratocyte_density_per_mm3: float = 20522.0  # ~20,000-25,000/mm³
    enable_keratocyte_markers: bool = False

    # === Nerve Plexus ===
    enable_nerve_plexus: bool = False
    subbasal_nerve_density_per_mm2: float = 5900.0  # 5000-6000/mm²
    stromal_nerve_bundle_count: int = 70  # 60-80 bundles

    # === Optical Properties ===
    transparency_factor: float = 1.0  # 0-1 (affects pore/feature density)
    refractive_index: float = 1.376  # REFERENCE ONLY: Optical property for simulation coupling, no geometric representation

    # === Limbal Transition ===
    enable_limbal_zone: bool = False
    limbal_width_mm: float = 1.5  # 1-2 mm transition zone

    # === Surface Curvature ===
    asphericity_q: float = -0.26  # Prolate ellipsoid (Q = -0.26 typical)
    posterior_radius_mm: float = 6.5  # 6.2-6.8 mm posterior surface

    # === Randomization ===
    seed: int = 42
    thickness_variance: float = 0.05  # 5% thickness variation

    # === Resolution ===
    resolution: int = 24  # High resolution for smooth dome


def conic_sag(r: float, radius_of_curvature: float, asphericity_q: float) -> float:
    """
    Calculate sag (z-height) at radial distance r using conic equation.

    z(r) = (c * r^2) / (1 + sqrt(1 - (1+Q) * c^2 * r^2))

    Where:
    - c = 1/R (curvature = 1/radius_of_curvature)
    - Q = asphericity coefficient (Q < 0 = prolate, Q = 0 = sphere, Q > 0 = oblate)

    Args:
        r: Radial distance from apex
        radius_of_curvature: Radius of curvature at apex
        asphericity_q: Asphericity coefficient (Q)

    Returns:
        Sag (z-height) at radial distance r
    """
    c = 1.0 / radius_of_curvature
    discriminant = 1.0 - (1.0 + asphericity_q) * c * c * r * r

    # Clamp discriminant to avoid sqrt of negative
    if discriminant < 0:
        discriminant = 0.0

    return (c * r * r) / (1.0 + np.sqrt(discriminant))


def make_aspherical_surface_mesh(
    diameter: float,
    radius_of_curvature: float,
    asphericity_q: float,
    n_radial: int,
    n_angular: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mesh vertices and faces for an aspherical (prolate ellipsoid) surface.

    Args:
        diameter: Diameter of the cornea
        radius_of_curvature: Radius of curvature at apex
        asphericity_q: Asphericity coefficient
        n_radial: Number of radial divisions
        n_angular: Number of angular divisions

    Returns:
        Tuple of (vertices, faces) arrays
    """
    radius = diameter / 2.0

    # Generate radial and angular grids
    r_values = np.linspace(0, radius, n_radial)
    theta_values = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)

    vertices = []

    # Add apex point
    vertices.append([0.0, 0.0, 0.0])

    # Generate surface points in concentric rings
    for i, r in enumerate(r_values[1:], 1):
        z = conic_sag(r, radius_of_curvature, asphericity_q)
        for theta in theta_values:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)

    # Generate faces
    faces = []

    # Triangles from apex to first ring
    for j in range(n_angular):
        next_j = (j + 1) % n_angular
        faces.append([0, 1 + j, 1 + next_j])

    # Triangles between rings
    for i in range(n_radial - 2):
        ring_start = 1 + i * n_angular
        next_ring_start = ring_start + n_angular

        for j in range(n_angular):
            next_j = (j + 1) % n_angular

            # Two triangles per quad
            v0 = ring_start + j
            v1 = ring_start + next_j
            v2 = next_ring_start + j
            v3 = next_ring_start + next_j

            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces, dtype=np.uint32)

    return vertices, faces


def make_aspherical_shell(
    diameter: float,
    thickness: float,
    outer_radius: float,
    inner_radius: float,
    asphericity_q: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create an aspherical shell (dome) using prolate ellipsoid surfaces.

    Args:
        diameter: Diameter of the cornea
        thickness: Wall thickness
        outer_radius: Radius of curvature for outer surface
        inner_radius: Radius of curvature for inner surface
        asphericity_q: Asphericity coefficient
        resolution: Angular resolution

    Returns:
        Manifold representing the aspherical shell
    """
    radius = diameter / 2.0
    n_radial = max(resolution // 2, 8)
    n_angular = resolution

    # Create outer aspherical surface
    outer_verts, outer_faces = make_aspherical_surface_mesh(
        diameter, outer_radius, asphericity_q, n_radial, n_angular
    )

    # Create inner aspherical surface (offset inward by thickness)
    inner_verts, inner_faces = make_aspherical_surface_mesh(
        diameter - 2 * thickness, inner_radius, asphericity_q, n_radial, n_angular
    )

    # Calculate the maximum sag for outer surface at edge
    max_sag_outer = conic_sag(radius, outer_radius, asphericity_q)
    max_sag_inner = conic_sag(radius - thickness, inner_radius, asphericity_q)

    # Use manifold cylinder + sphere approach for more robust geometry
    # Create outer sphere for outer surface
    outer_sphere = m3d.Manifold.sphere(outer_radius, resolution)
    outer_sphere = outer_sphere.translate([0, 0, -outer_radius + max_sag_outer])

    # Create inner sphere
    inner_sphere = m3d.Manifold.sphere(inner_radius, resolution)
    inner_height = conic_sag(radius - thickness, inner_radius, asphericity_q)
    inner_sphere = inner_sphere.translate([0, 0, -inner_radius + inner_height])

    # Clipping cylinder
    cap_height = max_sag_outer * 1.5
    clip_cylinder = m3d.Manifold.cylinder(
        cap_height,
        radius,
        radius,
        resolution
    )
    clip_cylinder = clip_cylinder.translate([0, 0, -cap_height * 0.1])

    # Clip to cap shapes (intersection, not XOR)
    outer_cap = outer_sphere & clip_cylinder

    inner_clip = m3d.Manifold.cylinder(
        cap_height,
        radius - thickness,
        radius - thickness,
        resolution
    )
    inner_clip = inner_clip.translate([0, 0, -cap_height * 0.1])
    inner_cap = inner_sphere & inner_clip

    # Create shell
    shell = outer_cap - inner_cap

    return shell


def make_layer_shell(
    diameter: float,
    outer_offset: float,
    inner_offset: float,
    outer_radius: float,
    inner_radius: float,
    asphericity_q: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a single layer shell at specified depth offsets.

    Args:
        diameter: Full cornea diameter
        outer_offset: Offset from anterior surface to outer edge of layer
        inner_offset: Offset from anterior surface to inner edge of layer
        outer_radius: Radius of curvature for outer surface
        inner_radius: Radius of curvature for inner surface
        asphericity_q: Asphericity coefficient
        resolution: Angular resolution

    Returns:
        Manifold representing the layer shell
    """
    radius = diameter / 2.0

    # Adjust radii for layer position
    R_outer = outer_radius - outer_offset
    R_inner = inner_radius - inner_offset

    # Calculate effective diameters for this layer depth
    d_outer = diameter - 2 * outer_offset
    d_inner = diameter - 2 * inner_offset

    if d_outer <= 0 or d_inner <= 0 or R_outer <= 0 or R_inner <= 0:
        return m3d.Manifold()

    # Calculate sag heights
    r_outer = d_outer / 2.0
    r_inner = d_inner / 2.0

    max_sag_outer = conic_sag(r_outer, R_outer, asphericity_q)
    max_sag_inner = conic_sag(r_inner, R_inner, asphericity_q)

    # Create spheres offset appropriately
    outer_sphere = m3d.Manifold.sphere(R_outer, resolution)
    outer_sphere = outer_sphere.translate([0, 0, -R_outer + max_sag_outer])

    inner_sphere = m3d.Manifold.sphere(R_inner, resolution)
    inner_sphere = inner_sphere.translate([0, 0, -R_inner + max_sag_inner])

    # Clip cylinders
    cap_height = max(max_sag_outer, max_sag_inner) * 2 + 1

    outer_clip = m3d.Manifold.cylinder(cap_height, r_outer, r_outer, resolution)
    outer_clip = outer_clip.translate([0, 0, -0.1])

    inner_clip = m3d.Manifold.cylinder(cap_height, r_inner, r_inner, resolution)
    inner_clip = inner_clip.translate([0, 0, -0.1])

    outer_cap = outer_sphere & outer_clip
    inner_cap = inner_sphere & inner_clip

    shell = outer_cap - inner_cap

    return shell


def make_five_layer_cornea(
    params: CorneaParams,
    rng: np.random.Generator = None
) -> Tuple[m3d.Manifold, dict]:
    """
    Create 5-layer cornea structure with anatomically correct layers.

    Layers from anterior (outer) to posterior (inner):
    1. Epithelium (50 μm)
    2. Bowman's layer (12 μm)
    3. Stroma (500 μm) - 90% of thickness
    4. Descemet's membrane (8 μm)
    5. Endothelium (5 μm)

    Args:
        params: CorneaParams specifying layer thicknesses and geometry
        rng: Optional random number generator for thickness variance

    Returns:
        Tuple of (manifold, layer_info_dict)
    """
    diameter = params.diameter_mm
    R_ant = params.radius_of_curvature_mm  # Anterior radius
    R_post = params.posterior_radius_mm    # Posterior radius
    Q = params.asphericity_q
    res = params.resolution

    # Convert layer thicknesses to mm
    epithelium_mm = params.epithelium_thickness_um / 1000.0
    bowmans_mm = params.bowmans_layer_thickness_um / 1000.0
    stroma_mm = params.stroma_thickness_um / 1000.0
    descemet_mm = params.descemet_thickness_um / 1000.0
    endothelium_mm = params.endothelium_thickness_um / 1000.0

    # Apply thickness variance to each layer independently
    if rng is not None and params.thickness_variance > 0:
        variance_factor = params.thickness_variance
        epithelium_mm *= (1.0 + rng.uniform(-variance_factor, variance_factor))
        bowmans_mm *= (1.0 + rng.uniform(-variance_factor, variance_factor))
        stroma_mm *= (1.0 + rng.uniform(-variance_factor, variance_factor))
        descemet_mm *= (1.0 + rng.uniform(-variance_factor, variance_factor))
        endothelium_mm *= (1.0 + rng.uniform(-variance_factor, variance_factor))

    total_thickness_mm = epithelium_mm + bowmans_mm + stroma_mm + descemet_mm + endothelium_mm

    # Calculate cumulative offsets from anterior surface
    offsets = {
        'epithelium': (0, epithelium_mm),
        'bowmans': (epithelium_mm, epithelium_mm + bowmans_mm),
        'stroma': (epithelium_mm + bowmans_mm, epithelium_mm + bowmans_mm + stroma_mm),
        'descemet': (epithelium_mm + bowmans_mm + stroma_mm,
                     epithelium_mm + bowmans_mm + stroma_mm + descemet_mm),
        'endothelium': (epithelium_mm + bowmans_mm + stroma_mm + descemet_mm,
                        total_thickness_mm)
    }

    layers = []
    layer_info = {}

    # Create each layer
    for layer_name, (outer_off, inner_off) in offsets.items():
        # Interpolate radius of curvature through thickness
        t_outer = outer_off / total_thickness_mm
        t_inner = inner_off / total_thickness_mm

        R_layer_outer = R_ant - (R_ant - R_post) * t_outer
        R_layer_inner = R_ant - (R_ant - R_post) * t_inner

        layer = make_layer_shell(
            diameter,
            outer_off,
            inner_off,
            R_layer_outer,
            R_layer_inner,
            Q,
            res
        )

        if layer.num_vert() > 0:
            layers.append(layer)
            layer_info[layer_name] = {
                'outer_offset_mm': outer_off,
                'inner_offset_mm': inner_off,
                'thickness_mm': inner_off - outer_off
            }

    if layers:
        result = batch_union(layers)
    else:
        # Fallback to single shell
        result = make_aspherical_shell(
            diameter, total_thickness_mm, R_ant, R_post, Q, res
        )

    return result, layer_info


def make_limbal_zone(
    diameter: float,
    limbal_width: float,
    thickness: float,
    epithelium_thickness: float,
    radius_of_curvature: float,
    asphericity_q: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create limbal zone with palisades of Vogt.

    The limbal zone is the transition region between cornea and sclera.
    Contains stem cells and has ridge-like palisades of Vogt structures.

    Args:
        diameter: Cornea diameter
        limbal_width: Width of limbal zone
        thickness: Total corneal thickness
        epithelium_thickness: Thickness of epithelium layer
        radius_of_curvature: Radius of curvature
        asphericity_q: Asphericity coefficient
        resolution: Angular resolution

    Returns:
        Manifold representing limbal zone with palisades
    """
    radius = diameter / 2.0
    inner_radius = radius - limbal_width

    if inner_radius <= 0:
        return m3d.Manifold()

    # Calculate sag at outer edge
    max_sag = conic_sag(radius, radius_of_curvature, asphericity_q)

    # Create outer annular ring (thicker epithelium in limbal zone)
    limbal_epithelium_factor = 1.5  # 50% thicker in limbal zone
    limbal_epi_thickness = epithelium_thickness * limbal_epithelium_factor

    # Outer cylinder for limbal ring
    outer_ring = m3d.Manifold.cylinder(
        limbal_epi_thickness,
        radius,
        radius,
        resolution
    )

    # Inner cylinder to hollow out
    inner_ring = m3d.Manifold.cylinder(
        limbal_epi_thickness * 1.1,
        inner_radius,
        inner_radius,
        resolution
    )
    inner_ring = inner_ring.translate([0, 0, -limbal_epi_thickness * 0.05])

    # Create annular base
    annulus = outer_ring - inner_ring

    # Position at edge of cornea dome
    annulus = annulus.translate([0, 0, max_sag - limbal_epi_thickness * 0.5])

    # Add palisades of Vogt (radial ridge structures)
    n_palisades = int(2 * np.pi * radius / 0.3)  # ~0.3mm spacing
    n_palisades = min(n_palisades, 120)  # Cap for performance

    palisades = []
    palisade_height = limbal_epi_thickness * 0.4
    palisade_width = 0.08  # 80 μm wide
    palisade_length = limbal_width * 0.8

    for i in range(n_palisades):
        angle = 2 * np.pi * i / n_palisades

        # Create radial ridge
        ridge = m3d.Manifold.cylinder(
            palisade_length,
            palisade_width / 2,
            palisade_width / 2,
            8
        )

        # Position ridge radially
        mid_radius = (radius + inner_radius) / 2
        x = mid_radius * np.cos(angle)
        y = mid_radius * np.sin(angle)

        # Rotate to point radially
        ridge = ridge.rotate([90, 0, np.degrees(angle)])
        ridge = ridge.translate([x, y, max_sag + palisade_height / 2])

        palisades.append(ridge)

    if palisades:
        palisades_union = batch_union(palisades)
        result = annulus + palisades_union
    else:
        result = annulus

    return result


def make_nerve_plexus(
    diameter: float,
    thickness: float,
    radius_of_curvature: float,
    asphericity_q: float,
    nerve_bundle_count: int,
    nerve_density_per_mm2: float,
    resolution: int,
    seed: int
) -> m3d.Manifold:
    """
    Create nerve plexus with radial channels and subbasal whorl pattern.

    Stromal nerves enter at limbus and radiate toward center.
    Subbasal nerve plexus forms a whorl pattern (vortex) typically inferonasal.

    Args:
        diameter: Cornea diameter
        thickness: Corneal thickness
        radius_of_curvature: Radius of curvature
        asphericity_q: Asphericity coefficient
        nerve_bundle_count: Number of stromal nerve bundles
        nerve_density_per_mm2: Subbasal nerve density for calculation
        resolution: Angular resolution
        seed: Random seed

    Returns:
        Manifold representing nerve channel network
    """
    np.random.seed(seed)

    radius = diameter / 2.0
    channels = []

    # Nerve channel diameter range (100-150 μm)
    channel_diameter_mm = 0.125  # 125 μm average

    # === Stromal nerve bundles ===
    # Enter at limbus and radiate inward
    for i in range(nerve_bundle_count):
        # Entry angle around limbus
        entry_angle = 2 * np.pi * i / nerve_bundle_count
        entry_angle += np.random.uniform(-0.1, 0.1)  # Small variation

        # Entry point at limbus
        entry_radius = radius * 0.95
        entry_x = entry_radius * np.cos(entry_angle)
        entry_y = entry_radius * np.sin(entry_angle)

        # Exit point toward center (varies by nerve)
        exit_radius = radius * np.random.uniform(0.1, 0.4)
        exit_angle = entry_angle + np.random.uniform(-0.3, 0.3)
        exit_x = exit_radius * np.cos(exit_angle)
        exit_y = exit_radius * np.sin(exit_angle)

        # Calculate sag heights
        entry_sag = conic_sag(np.sqrt(entry_x**2 + entry_y**2), radius_of_curvature, asphericity_q)
        exit_sag = conic_sag(np.sqrt(exit_x**2 + exit_y**2), radius_of_curvature, asphericity_q)

        # Create channel as cylinder between entry and exit points
        # Position in mid-stroma
        z_offset = thickness * 0.3  # 30% depth into stroma

        # Length of channel
        dx = exit_x - entry_x
        dy = exit_y - entry_y
        dz = (exit_sag - z_offset) - (entry_sag - z_offset)
        length = np.sqrt(dx**2 + dy**2 + dz**2)

        if length > 0.1:  # Only create if significant length
            channel = m3d.Manifold.cylinder(
                length,
                channel_diameter_mm / 2,
                channel_diameter_mm / 2,
                8
            )

            # Calculate rotation angles
            xy_length = np.sqrt(dx**2 + dy**2)
            angle_z = np.degrees(np.arctan2(dy, dx))
            angle_y = np.degrees(np.arctan2(dz, xy_length))

            channel = channel.rotate([90 - angle_y, 0, angle_z])
            channel = channel.translate([entry_x, entry_y, entry_sag - z_offset])

            channels.append(channel)

    # === Subbasal nerve whorl (vortex pattern) ===
    # Typically located inferonasal (below and toward nose)
    whorl_center_x = -radius * 0.15  # Nasal
    whorl_center_y = -radius * 0.15  # Inferior

    # Create spiral pattern
    n_spiral_arms = 6
    points_per_arm = 12

    for arm in range(n_spiral_arms):
        base_angle = 2 * np.pi * arm / n_spiral_arms

        prev_x, prev_y = None, None
        for p in range(points_per_arm):
            # Spiral equation: r = a + b*theta
            spiral_angle = base_angle + p * 0.5
            spiral_r = 0.1 + p * 0.15

            x = whorl_center_x + spiral_r * np.cos(spiral_angle)
            y = whorl_center_y + spiral_r * np.sin(spiral_angle)

            # Check bounds
            if np.sqrt(x**2 + y**2) > radius * 0.8:
                break

            if prev_x is not None:
                # Create channel segment
                dx = x - prev_x
                dy = y - prev_y
                seg_length = np.sqrt(dx**2 + dy**2)

                if seg_length > 0.01:
                    sag = conic_sag(np.sqrt(x**2 + y**2), radius_of_curvature, asphericity_q)

                    segment = m3d.Manifold.cylinder(
                        seg_length,
                        channel_diameter_mm / 3,  # Thinner subbasal nerves
                        channel_diameter_mm / 3,
                        6
                    )

                    angle_z = np.degrees(np.arctan2(dy, dx))
                    segment = segment.rotate([90, 0, angle_z])
                    segment = segment.translate([prev_x, prev_y, sag - thickness * 0.1])

                    channels.append(segment)

            prev_x, prev_y = x, y

    if channels:
        return batch_union(channels)

    return m3d.Manifold()


def make_keratocyte_markers(
    diameter: float,
    stroma_outer_offset: float,
    stroma_thickness: float,
    radius_of_curvature: float,
    asphericity_q: float,
    density_per_mm3: float,
    resolution: int,
    seed: int
) -> m3d.Manifold:
    """
    Generate keratocyte position markers in stroma.

    Keratocytes are stellate cells between collagen lamellae.
    Density is higher anteriorly (near Bowman's layer).

    Args:
        diameter: Cornea diameter
        stroma_outer_offset: Offset to outer surface of stroma
        stroma_thickness: Thickness of stroma layer
        radius_of_curvature: Radius of curvature
        asphericity_q: Asphericity coefficient
        density_per_mm3: Target keratocyte density
        resolution: Sphere resolution for markers
        seed: Random seed

    Returns:
        Manifold representing keratocyte markers
    """
    np.random.seed(seed)

    radius = diameter / 2.0

    # Calculate stroma volume approximately (hemispherical shell)
    # V ≈ 2/3 * pi * (R_outer^3 - R_inner^3) for hemisphere
    # Simplified: use cylindrical approximation
    area = np.pi * radius * radius
    stroma_volume = area * stroma_thickness * 0.7  # Correction for dome shape

    # Target number of keratocytes
    n_keratocytes = int(density_per_mm3 * stroma_volume)

    # Limit for performance
    max_markers = 500
    if n_keratocytes > max_markers:
        sampling_fraction = max_markers / n_keratocytes
        n_keratocytes = max_markers
    else:
        sampling_fraction = 1.0

    # Marker size (keratocytes are ~40 μm long stellate cells, marker is smaller)
    marker_radius = 0.015  # 15 μm radius sphere

    markers = []

    for _ in range(n_keratocytes):
        # Random position in cylindrical coordinates
        r = radius * np.sqrt(np.random.random()) * 0.9  # Uniform in area
        theta = np.random.random() * 2 * np.pi

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Depth in stroma (biased toward anterior)
        # Use beta distribution for anterior bias
        depth_fraction = np.random.beta(2, 5)  # Skewed toward 0 (anterior)
        z_in_stroma = stroma_outer_offset + stroma_thickness * depth_fraction

        # Calculate z position on dome
        sag = conic_sag(r, radius_of_curvature, asphericity_q)
        z = sag - z_in_stroma

        marker = m3d.Manifold.sphere(marker_radius, max(resolution // 4, 6))
        marker = marker.translate([x, y, z])
        markers.append(marker)

    if markers:
        return batch_union(markers)

    return m3d.Manifold()


def make_stromal_lamellae(
    diameter: float,
    stroma_outer_offset: float,
    stroma_thickness: float,
    num_lamellae: int,
    lamella_thickness: float,
    radius_of_curvature: float,
    asphericity_q: float,
    resolution: int,
    angle_variation_deg: float = 90.0
) -> m3d.Manifold:
    """
    Create stromal lamellae structure.

    The stroma contains 200-500 collagen lamellae, each ~2 μm thick.
    Lamellae are arranged with alternating fiber orientations.

    Args:
        diameter: Cornea diameter
        stroma_outer_offset: Offset to outer stroma surface
        stroma_thickness: Stroma thickness
        num_lamellae: Number of lamellae to create
        lamella_thickness: Thickness of each lamella (in mm)
        radius_of_curvature: Radius of curvature
        asphericity_q: Asphericity coefficient
        resolution: Angular resolution
        angle_variation_deg: Total angular range for lamellae rotation (degrees)

    Returns:
        Manifold representing lamellae structure
    """
    # Creating 300 individual shells is expensive
    # Create representative subset evenly spaced through stroma
    max_visual_lamellae = min(num_lamellae, 50)  # Visual representation

    # Calculate rotation step for angle variation
    angle_step = angle_variation_deg / max(max_visual_lamellae - 1, 1) if max_visual_lamellae > 1 else 0

    lamellae = []

    for i in range(max_visual_lamellae):
        # Position fraction through stroma
        fraction = (i + 0.5) / max_visual_lamellae

        # Offset from anterior surface
        lamella_offset = stroma_outer_offset + stroma_thickness * fraction

        # Calculate radius of curvature at this depth
        # Interpolate between anterior and posterior radii
        R_at_depth = radius_of_curvature - lamella_offset

        if R_at_depth <= 0:
            continue

        # Create thin shell at this depth
        d_at_depth = diameter - 2 * lamella_offset
        if d_at_depth <= 0:
            continue

        r_at_depth = d_at_depth / 2.0
        sag = conic_sag(r_at_depth, R_at_depth, asphericity_q)

        # Create sphere and clip
        sphere = m3d.Manifold.sphere(R_at_depth, resolution)
        sphere = sphere.translate([0, 0, -R_at_depth + sag])

        # Clip cylinder
        clip = m3d.Manifold.cylinder(sag * 2 + 0.1, r_at_depth, r_at_depth, resolution)
        clip = clip.translate([0, 0, -0.05])

        lamella_surface = sphere & clip

        # Make it a thin shell
        inner_sphere = m3d.Manifold.sphere(R_at_depth - lamella_thickness, resolution)
        inner_sag = conic_sag(r_at_depth - lamella_thickness, R_at_depth - lamella_thickness, asphericity_q)
        inner_sphere = inner_sphere.translate([0, 0, -(R_at_depth - lamella_thickness) + inner_sag])

        inner_clip = m3d.Manifold.cylinder(sag * 2 + 0.1, r_at_depth - lamella_thickness, r_at_depth - lamella_thickness, resolution)
        inner_clip = inner_clip.translate([0, 0, -0.05])

        inner_surface = inner_sphere ^ inner_clip

        lamella_shell = lamella_surface - inner_surface

        # Apply rotation based on angular variation
        # Rotation around Z axis to simulate alternating fiber orientations
        rotation_angle = i * angle_step
        lamella_shell = lamella_shell.rotate([0, 0, rotation_angle])

        if lamella_shell.num_vert() > 0:
            lamellae.append(lamella_shell)

    if lamellae:
        return batch_union(lamellae)

    return m3d.Manifold()


def generate_cornea(params: CorneaParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a cornea scaffold with biologically realistic parameters.

    Creates a 5-layer cornea with:
    - Prolate ellipsoid shape (asphericity)
    - 5 anatomical layers
    - Optional limbal zone with palisades of Vogt
    - Optional nerve plexus network
    - Optional keratocyte markers
    - Optional stromal lamellae visualization

    Args:
        params: CorneaParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, curvature_radius,
                     lamella_count, scaffold_type

    Raises:
        ValueError: If parameters are invalid
    """
    # Set random seed for reproducibility
    np.random.seed(params.seed)
    rng = np.random.default_rng(params.seed)

    # Convert total thickness from um to mm
    thickness_mm = params.total_thickness_um / 1000.0

    if params.diameter_mm <= 0 or thickness_mm <= 0:
        raise ValueError("Diameter and thickness must be positive")
    if params.radius_of_curvature_mm <= params.diameter_mm / 2:
        raise ValueError("Radius of curvature must be greater than half the diameter")
    if thickness_mm > params.radius_of_curvature_mm / 2:
        raise ValueError("Thickness too large for given radius of curvature")

    # === Create 5-layer structure ===
    base_structure, layer_info = make_five_layer_cornea(params, rng)

    components = [base_structure]

    # === Add limbal zone if enabled ===
    if params.enable_limbal_zone:
        limbal_zone = make_limbal_zone(
            params.diameter_mm,
            params.limbal_width_mm,
            thickness_mm,
            params.epithelium_thickness_um / 1000.0,
            params.radius_of_curvature_mm,
            params.asphericity_q,
            params.resolution
        )
        if limbal_zone.num_vert() > 0:
            components.append(limbal_zone)

    # === Add stromal lamellae if num_lamellae > 1 ===
    if params.num_lamellae > 1:
        # Calculate stroma boundaries
        epithelium_mm = params.epithelium_thickness_um / 1000.0
        bowmans_mm = params.bowmans_layer_thickness_um / 1000.0
        stroma_outer_offset = epithelium_mm + bowmans_mm
        stroma_thickness_mm = params.stroma_thickness_um / 1000.0

        # Scale number of visual lamellae by transparency
        # Higher transparency = fewer visible internal structures
        visual_lamellae_count = max(1, int(params.num_lamellae * (1 - params.transparency_factor * 0.5)))
        visual_lamellae_count = min(visual_lamellae_count, 50)  # Cap for performance

        if visual_lamellae_count > 1:
            lamellae = make_stromal_lamellae(
                params.diameter_mm,
                stroma_outer_offset,
                stroma_thickness_mm,
                visual_lamellae_count,
                params.lamella_thickness_um / 1000.0,
                params.radius_of_curvature_mm,
                params.asphericity_q,
                params.resolution,
                params.lamella_angle_variation_deg
            )
            if lamellae.num_vert() > 0:
                components.append(lamellae)

    # === Add keratocyte markers if enabled ===
    if params.enable_keratocyte_markers:
        epithelium_mm = params.epithelium_thickness_um / 1000.0
        bowmans_mm = params.bowmans_layer_thickness_um / 1000.0
        stroma_outer_offset = epithelium_mm + bowmans_mm
        stroma_thickness_mm = params.stroma_thickness_um / 1000.0

        # Scale keratocyte visibility by transparency
        effective_density = params.keratocyte_density_per_mm3 * (1 - params.transparency_factor * 0.7)

        keratocytes = make_keratocyte_markers(
            params.diameter_mm,
            stroma_outer_offset,
            stroma_thickness_mm,
            params.radius_of_curvature_mm,
            params.asphericity_q,
            effective_density,
            params.resolution,
            params.seed + 1
        )
        if keratocytes.num_vert() > 0:
            components.append(keratocytes)

    # === Add nerve plexus if enabled ===
    if params.enable_nerve_plexus:
        nerve_plexus = make_nerve_plexus(
            params.diameter_mm,
            thickness_mm,
            params.radius_of_curvature_mm,
            params.asphericity_q,
            params.stromal_nerve_bundle_count,
            params.subbasal_nerve_density_per_mm2,
            params.resolution,
            params.seed + 2
        )
        if nerve_plexus.num_vert() > 0:
            # Nerve channels are subtracted (hollow channels)
            base_idx = 0
            components[base_idx] = components[base_idx] - nerve_plexus

    # Combine all components
    if len(components) > 1:
        result = batch_union(components)
    else:
        result = components[0]

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    # Calculate cap height using aspherical equation
    R = params.radius_of_curvature_mm
    r = params.diameter_mm / 2
    cap_height = conic_sag(r, R, params.asphericity_q)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'diameter_mm': params.diameter_mm,
        'total_thickness_um': params.total_thickness_um,
        'thickness_mm': thickness_mm,
        'radius_of_curvature_mm': params.radius_of_curvature_mm,
        'posterior_radius_mm': params.posterior_radius_mm,
        'cap_height_mm': cap_height,
        'asphericity_q': params.asphericity_q,
        'layer_thicknesses_um': {
            'epithelium': params.epithelium_thickness_um,
            'bowmans_layer': params.bowmans_layer_thickness_um,
            'stroma': params.stroma_thickness_um,
            'descemet': params.descemet_thickness_um,
            'endothelium': params.endothelium_thickness_um
        },
        'layer_structure': layer_info,
        'stromal_architecture': {
            'num_lamellae': params.num_lamellae,
            'lamella_thickness_um': params.lamella_thickness_um,
            'lamella_angle_variation_deg': params.lamella_angle_variation_deg
        },
        'collagen_fibrils': {
            'diameter_nm': params.collagen_fibril_diameter_nm,
            'spacing_nm': params.collagen_fibril_spacing_nm
        },
        'keratocyte_density_per_mm3': params.keratocyte_density_per_mm3,
        'keratocyte_markers_enabled': params.enable_keratocyte_markers,
        'nerve_plexus': {
            'enabled': params.enable_nerve_plexus,
            'subbasal_density_per_mm2': params.subbasal_nerve_density_per_mm2,
            'stromal_bundle_count': params.stromal_nerve_bundle_count
        },
        'limbal_zone': {
            'enabled': params.enable_limbal_zone,
            'width_mm': params.limbal_width_mm
        },
        'optical': {
            'transparency_factor': params.transparency_factor,
            'refractive_index': params.refractive_index
        },
        'scaffold_type': 'cornea'
    }

    return result, stats


def generate_cornea_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate cornea scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching CorneaParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Support legacy parameter names
    total_thickness_um = params.get('total_thickness_um', 520.0)
    if 'thickness_mm' in params:
        total_thickness_um = params['thickness_mm'] * 1000  # Convert mm to um

    lamella_count = params.get('num_lamellae', params.get('lamella_count', 300))

    return generate_cornea(CorneaParams(
        # Basic geometry
        diameter_mm=params.get('diameter_mm', 11.5),
        radius_of_curvature_mm=params.get('radius_of_curvature_mm', 7.8),

        # Total thickness
        total_thickness_um=total_thickness_um,

        # Layer thicknesses
        epithelium_thickness_um=params.get('epithelium_thickness_um', 50.0),
        bowmans_layer_thickness_um=params.get('bowmans_layer_thickness_um', 12.0),
        stroma_thickness_um=params.get('stroma_thickness_um', 500.0),
        descemet_thickness_um=params.get('descemet_thickness_um', 8.0),
        endothelium_thickness_um=params.get('endothelium_thickness_um', 5.0),

        # Stromal architecture
        num_lamellae=lamella_count,
        lamella_thickness_um=params.get('lamella_thickness_um', 2.0),
        lamella_angle_variation_deg=params.get('lamella_angle_variation_deg', 90.0),

        # Collagen fibril parameters
        collagen_fibril_diameter_nm=params.get('collagen_fibril_diameter_nm', 30.0),
        collagen_fibril_spacing_nm=params.get('collagen_fibril_spacing_nm', 42.0),

        # Keratocyte distribution
        keratocyte_density_per_mm3=params.get('keratocyte_density_per_mm3', 20522.0),
        enable_keratocyte_markers=params.get('enable_keratocyte_markers', False),

        # Nerve plexus
        enable_nerve_plexus=params.get('enable_nerve_plexus', False),
        subbasal_nerve_density_per_mm2=params.get('subbasal_nerve_density_per_mm2', 5900.0),
        stromal_nerve_bundle_count=params.get('stromal_nerve_bundle_count', 70),

        # Optical properties
        transparency_factor=params.get('transparency_factor', 1.0),
        refractive_index=params.get('refractive_index', 1.376),

        # Limbal transition
        enable_limbal_zone=params.get('enable_limbal_zone', False),
        limbal_width_mm=params.get('limbal_width_mm', 1.5),

        # Surface curvature
        asphericity_q=params.get('asphericity_q', -0.26),
        posterior_radius_mm=params.get('posterior_radius_mm', 6.5),

        # Randomization
        seed=params.get('seed', 42),
        thickness_variance=params.get('thickness_variance', 0.05),

        # Resolution
        resolution=params.get('resolution', 24)
    ))
