import manifold3d as m3d
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class PrimitiveParams:
    shape: Literal['cylinder', 'sphere', 'box', 'cone'] = 'cylinder'
    dimensions: dict = field(default_factory=lambda: {'radius_mm': 5.0, 'height_mm': 10.0})
    modifications: list[dict] = field(default_factory=list)
    resolution: int = 32

def create_cylinder(dims: dict, resolution: int) -> m3d.Manifold:
    """Create a cylinder primitive."""
    radius = dims.get('radius_mm', 5.0)
    height = dims.get('height_mm', 10.0)
    return m3d.Manifold.cylinder(height, radius, radius, resolution)

def create_sphere(dims: dict, resolution: int) -> m3d.Manifold:
    """Create a sphere primitive."""
    radius = dims.get('radius_mm', 5.0)
    return m3d.Manifold.sphere(radius, resolution)

def create_box(dims: dict, resolution: int) -> m3d.Manifold:
    """Create a box primitive."""
    x = dims.get('x_mm', 10.0)
    y = dims.get('y_mm', 10.0)
    z = dims.get('z_mm', 10.0)
    return m3d.Manifold.cube([x, y, z])

def create_cone(dims: dict, resolution: int) -> m3d.Manifold:
    """Create a cone/frustum primitive."""
    bottom_radius = dims.get('bottom_radius_mm', 5.0)
    top_radius = dims.get('top_radius_mm', 0.0)
    height = dims.get('height_mm', 10.0)
    return m3d.Manifold.cylinder(height, bottom_radius, top_radius, resolution)

SHAPE_CREATORS = {
    'cylinder': create_cylinder,
    'sphere': create_sphere,
    'box': create_box,
    'cone': create_cone
}

def apply_hole(manifold: m3d.Manifold, params: dict, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical hole through the manifold.

    params:
        diameter_mm: hole diameter
        axis: 'x', 'y', or 'z' (default 'z')
        depth_mm: optional, defaults to through-hole
    """
    diameter = params.get('diameter_mm', 2.0)
    axis = params.get('axis', 'z')
    radius = diameter / 2

    # Get bounding box to determine depth
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    min_bounds = verts.min(axis=0)
    max_bounds = verts.max(axis=0)

    # Calculate center and depth based on axis
    center = (min_bounds + max_bounds) / 2

    if axis == 'z':
        depth = (max_bounds[2] - min_bounds[2]) + 0.2
        hole = m3d.Manifold.cylinder(depth, radius, radius, resolution)
        hole = hole.translate([center[0], center[1], min_bounds[2] - 0.1])
    elif axis == 'y':
        depth = (max_bounds[1] - min_bounds[1]) + 0.2
        hole = m3d.Manifold.cylinder(depth, radius, radius, resolution)
        hole = hole.rotate([90, 0, 0]).translate([center[0], min_bounds[1] - 0.1, center[2]])
    else:  # x
        depth = (max_bounds[0] - min_bounds[0]) + 0.2
        hole = m3d.Manifold.cylinder(depth, radius, radius, resolution)
        hole = hole.rotate([0, 90, 0]).translate([min_bounds[0] - 0.1, center[1], center[2]])

    return manifold - hole

def apply_shell(manifold: m3d.Manifold, params: dict, resolution: int) -> m3d.Manifold:
    """
    Hollow out the manifold with given wall thickness.

    params:
        wall_thickness_mm: thickness of shell wall
    """
    thickness = params.get('wall_thickness_mm', 1.0)

    # Scale down to create inner void
    # Get bounding box
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    min_bounds = verts.min(axis=0)
    max_bounds = verts.max(axis=0)
    size = max_bounds - min_bounds
    center = (min_bounds + max_bounds) / 2

    # Calculate scale factor to reduce by wall_thickness on each side
    scale_factors = []
    for i, s in enumerate(size):
        if s > 2 * thickness:
            scale_factors.append((s - 2 * thickness) / s)
        else:
            scale_factors.append(0.5)  # Minimum scale

    # Create scaled inner manifold
    # Move to origin, scale, move back
    inner = manifold.translate([-center[0], -center[1], -center[2]])
    inner = inner.scale([scale_factors[0], scale_factors[1], scale_factors[2]])
    inner = inner.translate([center[0], center[1], center[2]])

    return manifold - inner

MODIFICATIONS = {
    'hole': apply_hole,
    'shell': apply_shell
}

def generate_primitive(params: PrimitiveParams) -> tuple[m3d.Manifold, dict]:
    """Generate a primitive shape with optional modifications."""

    if params.shape not in SHAPE_CREATORS:
        raise ValueError(f"Unknown shape: {params.shape}")

    # Create base shape
    manifold = SHAPE_CREATORS[params.shape](params.dimensions, params.resolution)

    # Apply modifications in order
    for mod in params.modifications:
        operation = mod.get('operation')
        mod_params = mod.get('params', {})

        if operation in MODIFICATIONS:
            manifold = MODIFICATIONS[operation](manifold, mod_params, params.resolution)
        else:
            raise ValueError(f"Unknown modification: {operation}")

    # Statistics
    mesh = manifold.to_mesh()
    volume = manifold.volume() if hasattr(manifold, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'shape': params.shape,
        'modification_count': len(params.modifications),
        'scaffold_type': 'primitive'
    }

    return manifold, stats

def generate_primitive_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    return generate_primitive(PrimitiveParams(
        shape=params.get('shape', 'cylinder'),
        dimensions=params.get('dimensions', {}),
        modifications=params.get('modifications', []),
        resolution=params.get('resolution', 32)
    ))
