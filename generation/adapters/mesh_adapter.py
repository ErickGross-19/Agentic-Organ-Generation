"""
Adapter for converting VascularNetwork to trimesh.Trimesh.

Provides STL mesh export with fast and robust modes, including hollow tube generation.
"""

import numpy as np
import trimesh
from typing import Optional, Literal, Dict, Any
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, ErrorCode


def to_trimesh(
    network: VascularNetwork,
    mode: Literal["fast", "robust"] = "fast",
    radial_resolution: int = 8,
    include_caps: bool = True,
    min_segment_length: float = 1e-6,
) -> OperationResult:
    """
    Convert VascularNetwork to trimesh.Trimesh for STL export.
    
    Two modes:
    - "fast": Concatenate all segment meshes, then repair with voxel remesh
    - "robust": Attempt boolean union (requires backend), fallback to fast
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to convert
    mode : {"fast", "robust"}
        Export mode
    radial_resolution : int
        Number of vertices around each cylinder
    include_caps : bool
        Whether to include hemispherical caps at segment ends
    min_segment_length : float
        Skip segments shorter than this
        
    Returns
    -------
    OperationResult
        Result with mesh in metadata['mesh']
    """
    try:
        meshes = []
        
        for seg_id, segment in network.segments.items():
            start = np.array([
                segment.geometry.start.x,
                segment.geometry.start.y,
                segment.geometry.start.z,
            ])
            end = np.array([
                segment.geometry.end.x,
                segment.geometry.end.y,
                segment.geometry.end.z,
            ])
            
            length = np.linalg.norm(end - start)
            if length < min_segment_length:
                continue
            
            r_start = segment.geometry.radius_start
            r_end = segment.geometry.radius_end
            
            mesh = _create_capsule_mesh(
                start, end, r_start, r_end,
                radial_resolution=radial_resolution,
                include_caps=include_caps,
            )
            meshes.append(mesh)
        
        for node_id, node in network.nodes.items():
            # Start from 0 so we only grow based on attached segments
            max_radius = 0.0
            for seg in network.segments.values():
                if seg.start_node_id == node_id:
                    max_radius = max(max_radius, seg.geometry.radius_start)
                elif seg.end_node_id == node_id:
                    max_radius = max(max_radius, seg.geometry.radius_end)
        
            # If no segments touch this node, skip making a sphere
            if max_radius <= 0.00002:
                continue
        
            # Make node spheres much smaller than vessel radii
            sphere_radius = max_radius * 0.2  # try 0.1–0.3 depending how subtle you want it

            
            center = np.array([node.position.x, node.position.y, node.position.z])
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
            sphere.apply_translation(center)
            meshes.append(sphere)
        
        if not meshes:
            return OperationResult.failure(
                "No meshes created",
                error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
            )
        
        if mode == "robust":
            try:
                combined = meshes[0]
                for mesh in meshes[1:]:
                    combined = trimesh.boolean.union([combined, mesh], engine='blender')
                
                result = OperationResult.success(
                    f"Robust mesh export: {len(meshes)} components unioned",
                    metadata={
                        'mesh': combined,
                        'mode': 'robust',
                        'num_components': len(meshes),
                        'is_watertight': combined.is_watertight,
                    },
                )
                return result
                
            except Exception as e:
                result = OperationResult.partial_success(
                    f"Boolean union failed, falling back to fast mode: {e}",
                )
                result.add_warning(f"Robust mode failed: {e}")
                mode = "fast"
        
        combined = trimesh.util.concatenate(meshes)
        
        result = OperationResult.success(
            f"Fast mesh export: {len(meshes)} components concatenated",
            metadata={
                'mesh': combined,
                'mode': mode,
                'num_components': len(meshes),
                'is_watertight': combined.is_watertight,
                'needs_repair': not combined.is_watertight,
            },
        )
        
        return result
        
    except Exception as e:
        return OperationResult.failure(
            f"Mesh export failed: {e}",
            error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
        )


def _create_capsule_mesh(
    start: np.ndarray,
    end: np.ndarray,
    r_start: float,
    r_end: float,
    radial_resolution: int = 8,
    include_caps: bool = True,
) -> trimesh.Trimesh:
    """
    Create a capsule mesh (frustum/tapered cylinder with optional hemispherical caps).
    
    This function properly handles tapering by creating a frustum (truncated cone)
    when r_start != r_end, rather than ignoring the end radius.
    
    Parameters
    ----------
    start : np.ndarray
        Start point [x, y, z]
    end : np.ndarray
        End point [x, y, z]
    r_start : float
        Radius at start
    r_end : float
        Radius at end
    radial_resolution : int
        Number of vertices around cylinder
    include_caps : bool
        Whether to include hemispherical caps
        
    Returns
    -------
    trimesh.Trimesh
        Capsule mesh with proper tapering
    """
    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length
    
    frustum = _create_frustum(
        length=length,
        radius_start=r_start,
        radius_end=r_end,
        sections=radial_resolution,
    )
    
    z_axis = np.array([0, 0, 1])
    if not np.allclose(direction, z_axis) and not np.allclose(direction, -z_axis):
        rotation_axis = np.cross(z_axis, direction)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, rotation_axis
            )
            frustum.apply_transform(rotation_matrix)
    elif np.allclose(direction, -z_axis):
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        frustum.apply_transform(rotation_matrix)
    
    center = (start + end) / 2
    frustum.apply_translation(center)
    
    if include_caps:
        cap_start = trimesh.creation.icosphere(subdivisions=1, radius=r_start)
        cap_start.apply_translation(start)
        
        cap_end = trimesh.creation.icosphere(subdivisions=1, radius=r_end)
        cap_end.apply_translation(end)
        
        return trimesh.util.concatenate([frustum, cap_start, cap_end])
    
    return frustum


def _create_frustum(
    length: float,
    radius_start: float,
    radius_end: float,
    sections: int = 8,
) -> trimesh.Trimesh:
    """
    Create a frustum (truncated cone) mesh.
    
    This is used for tapered vessel segments where the radius changes
    from start to end.
    
    Parameters
    ----------
    length : float
        Length of the frustum along z-axis
    radius_start : float
        Radius at z = -length/2 (start)
    radius_end : float
        Radius at z = +length/2 (end)
    sections : int
        Number of vertices around the circumference
        
    Returns
    -------
    trimesh.Trimesh
        Frustum mesh centered at origin, aligned with z-axis
    """
    if abs(radius_start - radius_end) < 1e-10:
        return trimesh.creation.cylinder(
            radius=radius_start,
            height=length,
            sections=sections,
        )
    
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    
    z_bottom = -length / 2
    z_top = length / 2
    
    bottom_vertices = np.column_stack([
        radius_start * np.cos(angles),
        radius_start * np.sin(angles),
        np.full(sections, z_bottom),
    ])
    
    top_vertices = np.column_stack([
        radius_end * np.cos(angles),
        radius_end * np.sin(angles),
        np.full(sections, z_top),
    ])
    
    vertices = np.vstack([bottom_vertices, top_vertices])
    
    faces = []
    for i in range(sections):
        next_i = (i + 1) % sections
        
        faces.append([i, next_i, sections + i])
        faces.append([next_i, sections + next_i, sections + i])
    
    bottom_center_idx = len(vertices)
    vertices = np.vstack([vertices, [[0, 0, z_bottom]]])
    for i in range(sections):
        next_i = (i + 1) % sections
        faces.append([bottom_center_idx, next_i, i])
    
    top_center_idx = len(vertices)
    vertices = np.vstack([vertices, [[0, 0, z_top]]])
    for i in range(sections):
        next_i = (i + 1) % sections
        faces.append([top_center_idx, sections + i, sections + next_i])
    
    faces = np.array(faces)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def export_stl(
    network: VascularNetwork,
    output_path: str,
    mode: Literal["fast", "robust"] = "fast",
    repair: bool = True,
    output_units: str = "mm",
    write_metadata: bool = True,
    **kwargs,
) -> OperationResult:
    """
    Export VascularNetwork to STL file with unit scaling.
    
    The library uses DIMENSIONLESS internal units where 1 internal unit = 1 output unit.
    When output_units="mm", 1 internal unit = 1 mm in the output STL file.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to export
    output_path : str
        Path to output STL file
    mode : {"fast", "robust"}
        Export mode
    repair : bool
        Whether to repair mesh before export
    output_units : str
        Units for the output STL file. Default: "mm"
        Supported: "m", "mm", "cm", "um"
        Since internal units are dimensionless (1 = 1mm equivalent),
        this controls the scale of coordinates in the output file.
    write_metadata : bool
        Whether to write a sidecar JSON file with unit metadata.
        The sidecar file will be named {output_path}.units.json
    **kwargs
        Additional arguments passed to to_trimesh
        
    Returns
    -------
    OperationResult
        Result of export operation
        
    Notes
    -----
    STL files do not encode units. The output_units parameter controls
    the numeric scale of vertex coordinates. A sidecar JSON file is
    written to document the units used.
    """
    from generation.utils.units import UnitContext
    import json
    from pathlib import Path
    
    result = to_trimesh(network, mode=mode, **kwargs)
    
    if not result.is_success():
        return result
    
    mesh = result.metadata['mesh']
    
    if repair and not mesh.is_watertight:
        try:
            from validity.mesh.repair import meshfix_repair
            mesh = meshfix_repair(mesh, keep_largest_component=True)
            result.add_warning("Mesh repaired with meshfix")
            result.metadata['was_repaired'] = True
            result.metadata['is_watertight'] = mesh.is_watertight
        except Exception as e:
            result.add_warning(f"Mesh repair failed: {e}")
            result.metadata['repair_failed'] = True
    
    unit_ctx = UnitContext(output_units=output_units)
    scaled_mesh = unit_ctx.scale_mesh(mesh)
    
    result.metadata['output_units'] = output_units
    result.metadata['scale_factor'] = unit_ctx.scale_factor
    
    try:
        scaled_mesh.export(output_path)
        result.message = f"Exported to {output_path} (units: {output_units})"
        result.metadata['output_path'] = output_path
        
        if write_metadata:
            metadata_path = str(output_path) + ".units.json"
            metadata = unit_ctx.get_metadata()
            metadata['stl_file'] = str(output_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            result.metadata['metadata_path'] = metadata_path
        
        return result
    except Exception as e:
        return OperationResult.failure(
            f"Export failed: {e}",
            error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
        )


def to_hollow_tube_mesh(
    network: VascularNetwork,
    wall_thickness: float,
    radial_resolution: int = 16,
    min_segment_length: float = 1e-6,
    min_inner_radius: float = 0.0001,
) -> OperationResult:
    """
    Convert VascularNetwork to a hollow tube mesh for fluid flow.
    
    Creates a continuous hollow channel from inlet to terminal nodes,
    with walls of specified thickness. The inner channel allows water
    or other fluids to flow through the entire vascular network.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to convert
    wall_thickness : float
        Thickness of the tube walls (in same units as network, typically mm).
        The inner radius will be: outer_radius - wall_thickness
    radial_resolution : int
        Number of vertices around each cylinder (higher = smoother)
    min_segment_length : float
        Skip segments shorter than this
    min_inner_radius : float
        Minimum inner radius to prevent degenerate geometry.
        If inner_radius would be smaller, a warning is issued.
        
    Returns
    -------
    OperationResult
        Result with metadata containing:
        - 'mesh': trimesh.Trimesh of the hollow tube
        - 'outer_mesh': trimesh.Trimesh of outer surface only
        - 'inner_mesh': trimesh.Trimesh of inner surface only
        - 'num_segments': number of segments processed
        - 'warnings': list of any warnings (e.g., segments with thin walls)
    
    Example
    -------
    >>> from vascular_lib import create_network, add_inlet, grow_branch
    >>> from generation.adapters.mesh_adapter import to_hollow_tube_mesh
    >>> 
    >>> # Create a simple network
    >>> network = create_network(domain, seed=42)
    >>> add_inlet(network, position=(0, 0, 0), direction=(1, 0, 0), radius=5.0)
    >>> grow_branch(network, from_node_id=0, length=20.0, direction=(1, 0, 0))
    >>> 
    >>> # Convert to hollow tube with 1mm wall thickness
    >>> result = to_hollow_tube_mesh(network, wall_thickness=1.0)
    >>> if result.is_success():
    ...     hollow_mesh = result.metadata['mesh']
    ...     hollow_mesh.export('hollow_network.stl')
    """
    try:
        outer_meshes = []
        inner_meshes = []
        warnings = []
        segments_processed = 0
        
        for seg_id, segment in network.segments.items():
            start = np.array([
                segment.geometry.start.x,
                segment.geometry.start.y,
                segment.geometry.start.z,
            ])
            end = np.array([
                segment.geometry.end.x,
                segment.geometry.end.y,
                segment.geometry.end.z,
            ])
            
            length = np.linalg.norm(end - start)
            if length < min_segment_length:
                continue
            
            r_start_outer = segment.geometry.radius_start
            r_end_outer = segment.geometry.radius_end
            
            r_start_inner = r_start_outer - wall_thickness
            r_end_inner = r_end_outer - wall_thickness
            
            if r_start_inner < min_inner_radius:
                warnings.append(
                    f"Segment {seg_id}: start inner radius {r_start_inner:.4f} "
                    f"below minimum {min_inner_radius:.4f}, clamping"
                )
                r_start_inner = min_inner_radius
            
            if r_end_inner < min_inner_radius:
                warnings.append(
                    f"Segment {seg_id}: end inner radius {r_end_inner:.4f} "
                    f"below minimum {min_inner_radius:.4f}, clamping"
                )
                r_end_inner = min_inner_radius
            
            outer_tube = _create_hollow_tube_segment(
                start, end, r_start_outer, r_end_outer,
                radial_resolution=radial_resolution,
                is_outer=True,
            )
            outer_meshes.append(outer_tube)
            
            inner_tube = _create_hollow_tube_segment(
                start, end, r_start_inner, r_end_inner,
                radial_resolution=radial_resolution,
                is_outer=False,
            )
            inner_meshes.append(inner_tube)
            
            segments_processed += 1
        
        for node_id, node in network.nodes.items():
            max_outer_radius = 0.0
            max_inner_radius = 0.0
            
            for seg in network.segments.values():
                if seg.start_node_id == node_id:
                    outer_r = seg.geometry.radius_start
                    inner_r = max(outer_r - wall_thickness, min_inner_radius)
                    max_outer_radius = max(max_outer_radius, outer_r)
                    max_inner_radius = max(max_inner_radius, inner_r)
                elif seg.end_node_id == node_id:
                    outer_r = seg.geometry.radius_end
                    inner_r = max(outer_r - wall_thickness, min_inner_radius)
                    max_outer_radius = max(max_outer_radius, outer_r)
                    max_inner_radius = max(max_inner_radius, inner_r)
            
            if max_outer_radius <= min_inner_radius:
                continue
            
            center = np.array([node.position.x, node.position.y, node.position.z])
            
            outer_sphere = trimesh.creation.icosphere(
                subdivisions=2, radius=max_outer_radius
            )
            outer_sphere.apply_translation(center)
            outer_meshes.append(outer_sphere)
            
            inner_sphere = trimesh.creation.icosphere(
                subdivisions=2, radius=max_inner_radius
            )
            inner_sphere.apply_translation(center)
            inner_meshes.append(inner_sphere)
        
        if not outer_meshes:
            return OperationResult.failure(
                "No meshes created - network may be empty",
                error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
            )
        
        outer_combined = trimesh.util.concatenate(outer_meshes)
        inner_combined = trimesh.util.concatenate(inner_meshes)
        
        inner_combined.invert()
        
        hollow_mesh = trimesh.util.concatenate([outer_combined, inner_combined])
        
        end_cap_meshes = _create_end_caps(
            network, wall_thickness, radial_resolution, min_inner_radius
        )
        if end_cap_meshes:
            hollow_mesh = trimesh.util.concatenate([hollow_mesh] + end_cap_meshes)
        
        result = OperationResult.success(
            f"Created hollow tube mesh from {segments_processed} segments",
            metadata={
                'mesh': hollow_mesh,
                'outer_mesh': outer_combined,
                'inner_mesh': inner_combined,
                'num_segments': segments_processed,
                'wall_thickness': wall_thickness,
                'is_watertight': hollow_mesh.is_watertight,
            },
        )
        
        for warning in warnings:
            result.add_warning(warning)
        
        return result
        
    except Exception as e:
        return OperationResult.failure(
            f"Hollow tube mesh creation failed: {e}",
            error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
        )


def _create_hollow_tube_segment(
    start: np.ndarray,
    end: np.ndarray,
    r_start: float,
    r_end: float,
    radial_resolution: int = 16,
    is_outer: bool = True,
) -> trimesh.Trimesh:
    """
    Create a single tube segment (cylinder without caps).
    
    Parameters
    ----------
    start : np.ndarray
        Start point [x, y, z]
    end : np.ndarray
        End point [x, y, z]
    r_start : float
        Radius at start
    r_end : float
        Radius at end
    radial_resolution : int
        Number of vertices around cylinder
    is_outer : bool
        If True, normals point outward; if False, normals point inward
        
    Returns
    -------
    trimesh.Trimesh
        Tube segment mesh (open-ended cylinder)
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return trimesh.Trimesh()
    
    direction = direction / length
    
    mean_radius = (r_start + r_end) / 2.0
    
    cylinder = trimesh.creation.cylinder(
        radius=mean_radius,
        height=length,
        sections=radial_resolution,
        cap=False,
    )
    
    z_axis = np.array([0, 0, 1])
    if not np.allclose(direction, z_axis) and not np.allclose(direction, -z_axis):
        rotation_axis = np.cross(z_axis, direction)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, rotation_axis
            )
            cylinder.apply_transform(rotation_matrix)
    elif np.allclose(direction, -z_axis):
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0]
        )
        cylinder.apply_transform(rotation_matrix)
    
    center = (start + end) / 2
    cylinder.apply_translation(center)
    
    return cylinder


def _create_end_caps(
    network: VascularNetwork,
    wall_thickness: float,
    radial_resolution: int,
    min_inner_radius: float,
) -> list:
    """
    Create annular end caps at inlet and outlet nodes.
    
    These caps seal the hollow tube at the entry/exit points while
    leaving the inner channel open for fluid flow.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    wall_thickness : float
        Wall thickness for computing inner radius
    radial_resolution : int
        Number of vertices around each ring
    min_inner_radius : float
        Minimum inner radius
        
    Returns
    -------
    list
        List of trimesh.Trimesh objects for end caps
    """
    caps = []
    
    for node_id, node in network.nodes.items():
        if node.node_type not in ("inlet", "outlet", "terminal"):
            continue
        
        connected_segments = []
        for seg in network.segments.values():
            if seg.start_node_id == node_id:
                connected_segments.append((seg, "start"))
            elif seg.end_node_id == node_id:
                connected_segments.append((seg, "end"))
        
        if not connected_segments:
            continue
        
        seg, end_type = connected_segments[0]
        
        if end_type == "start":
            outer_radius = seg.geometry.radius_start
            seg_start = np.array([
                seg.geometry.start.x,
                seg.geometry.start.y,
                seg.geometry.start.z,
            ])
            seg_end = np.array([
                seg.geometry.end.x,
                seg.geometry.end.y,
                seg.geometry.end.z,
            ])
            direction = seg_end - seg_start
            cap_center = seg_start
        else:
            outer_radius = seg.geometry.radius_end
            seg_start = np.array([
                seg.geometry.start.x,
                seg.geometry.start.y,
                seg.geometry.start.z,
            ])
            seg_end = np.array([
                seg.geometry.end.x,
                seg.geometry.end.y,
                seg.geometry.end.z,
            ])
            direction = seg_start - seg_end
            cap_center = seg_end
        
        inner_radius = max(outer_radius - wall_thickness, min_inner_radius)
        
        direction_len = np.linalg.norm(direction)
        if direction_len < 1e-10:
            continue
        direction = direction / direction_len
        
        cap = _create_annular_cap(
            center=cap_center,
            normal=direction,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            radial_resolution=radial_resolution,
        )
        caps.append(cap)
    
    return caps


def _create_annular_cap(
    center: np.ndarray,
    normal: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    radial_resolution: int = 16,
) -> trimesh.Trimesh:
    """
    Create an annular (ring-shaped) cap.
    
    Parameters
    ----------
    center : np.ndarray
        Center point of the cap
    normal : np.ndarray
        Normal direction of the cap (pointing outward from tube)
    inner_radius : float
        Inner radius of the ring (hole for fluid flow)
    outer_radius : float
        Outer radius of the ring
    radial_resolution : int
        Number of vertices around each ring
        
    Returns
    -------
    trimesh.Trimesh
        Annular cap mesh
    """
    angles = np.linspace(0, 2 * np.pi, radial_resolution, endpoint=False)
    
    inner_circle = np.column_stack([
        inner_radius * np.cos(angles),
        inner_radius * np.sin(angles),
        np.zeros(radial_resolution),
    ])
    
    outer_circle = np.column_stack([
        outer_radius * np.cos(angles),
        outer_radius * np.sin(angles),
        np.zeros(radial_resolution),
    ])
    
    vertices = np.vstack([inner_circle, outer_circle])
    
    faces = []
    n = radial_resolution
    for i in range(n):
        i_next = (i + 1) % n
        faces.append([i, i_next, n + i])
        faces.append([i_next, n + i_next, n + i])
    
    faces = np.array(faces)
    
    cap = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    
    if not np.allclose(normal, z_axis) and not np.allclose(normal, -z_axis):
        rotation_axis = np.cross(z_axis, normal)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, normal), -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, rotation_axis
            )
            cap.apply_transform(rotation_matrix)
    elif np.allclose(normal, -z_axis):
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0]
        )
        cap.apply_transform(rotation_matrix)
    
    cap.apply_translation(center)
    
    return cap


def export_hollow_tube_stl(
    network: VascularNetwork,
    output_path: str,
    wall_thickness: float,
    repair: bool = True,
    output_units: str = "mm",
    write_metadata: bool = True,
    **kwargs,
) -> OperationResult:
    """
    Export VascularNetwork as a hollow tube STL file with unit scaling.
    
    Creates a hollow tube mesh where water can flow through the entire
    network from inlet to terminal nodes, then exports to STL.
    
    The library uses DIMENSIONLESS internal units where 1 internal unit = 1 output unit.
    When output_units="mm", 1 internal unit = 1 mm in the output STL file.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to export
    output_path : str
        Path to output STL file
    wall_thickness : float
        Thickness of tube walls (in internal dimensionless units)
    repair : bool
        Whether to attempt mesh repair before export
    output_units : str
        Units for the output STL file. Default: "mm"
        Supported: "m", "mm", "cm", "um"
    write_metadata : bool
        Whether to write a sidecar JSON file with unit metadata.
    **kwargs
        Additional arguments passed to to_hollow_tube_mesh
        
    Returns
    -------
    OperationResult
        Result of export operation
        
    Example
    -------
    >>> from generation.adapters.mesh_adapter import export_hollow_tube_stl
    >>> result = export_hollow_tube_stl(
    ...     network,
    ...     output_path='hollow_network.stl',
    ...     wall_thickness=1.0,  # 1 internal unit (= 1mm if output_units="mm")
    ...     output_units="mm",
    ... )
    """
    from generation.utils.units import UnitContext
    import json
    
    result = to_hollow_tube_mesh(network, wall_thickness=wall_thickness, **kwargs)
    
    if not result.is_success():
        return result
    
    mesh = result.metadata['mesh']
    
    if repair and not mesh.is_watertight:
        try:
            from validity.mesh.repair import meshfix_repair
            mesh = meshfix_repair(mesh, keep_largest_component=False)
            result.add_warning("Mesh repaired with meshfix")
            result.metadata['was_repaired'] = True
            result.metadata['is_watertight'] = mesh.is_watertight
        except Exception as e:
            result.add_warning(f"Mesh repair failed: {e}")
            result.metadata['repair_failed'] = True
    
    unit_ctx = UnitContext(output_units=output_units)
    scaled_mesh = unit_ctx.scale_mesh(mesh)
    
    result.metadata['output_units'] = output_units
    result.metadata['scale_factor'] = unit_ctx.scale_factor
    
    try:
        scaled_mesh.export(output_path)
        result.message = f"Exported hollow tube to {output_path} (units: {output_units})"
        result.metadata['output_path'] = output_path
        
        if write_metadata:
            metadata_path = str(output_path) + ".units.json"
            metadata = unit_ctx.get_metadata()
            metadata['stl_file'] = str(output_path)
            metadata['wall_thickness_internal'] = wall_thickness
            metadata['wall_thickness_output'] = unit_ctx.to_output(wall_thickness)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            result.metadata['metadata_path'] = metadata_path
        
        return result
    except Exception as e:
        return OperationResult.failure(
            f"Export failed: {e}",
            error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
        )
