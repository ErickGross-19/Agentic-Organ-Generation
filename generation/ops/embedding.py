"""
Negative space embedding operations for vascular networks.

This module provides functionality to embed vascular tree STL meshes into
domain volumes (box or ellipsoid) as negative space (voids).

Units: All spatial parameters are in millimeters by default.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from scipy import ndimage
from skimage.measure import marching_cubes

from ..core.domain import DomainSpec, BoxDomain, EllipsoidDomain
from ..utils.units import detect_unit, warn_if_legacy_units, CANONICAL_UNIT


def _voxelized_with_retry(
    mesh: trimesh.Trimesh,
    pitch: float,
    max_attempts: int = 4,
    factor: float = 1.5,
    log_prefix: str = "",
):
    """
    Voxelize a mesh with automatic retry on memory errors.
    
    If voxelization fails due to memory constraints, the pitch is increased
    by the specified factor and retried up to max_attempts times.
    
    Returns tuple of (voxel_grid, final_pitch) so caller knows if pitch changed.
    """
    cur = float(pitch)
    last_exc = None
    
    for attempt in range(max_attempts):
        try:
            vox = mesh.voxelized(cur)
            return vox, cur
        except MemoryError as e:
            last_exc = e
            print(
                f"{log_prefix}voxelized(pitch={cur:.4g}) raised MemoryError, "
                f"increasing pitch (attempt {attempt + 1}/{max_attempts})..."
            )
            cur *= factor
        except ValueError as e:
            last_exc = e
            print(
                f"{log_prefix}voxelized(pitch={cur:.4g}) failed ({e}), "
                f"increasing pitch (attempt {attempt + 1}/{max_attempts})..."
            )
            cur *= factor
    
    raise RuntimeError(
        f"{log_prefix}voxelization failed after {max_attempts} attempts; "
        f"final pitch={cur:.4g}, original pitch={pitch:.4g}"
    ) from last_exc


def _compute_ellipsoid_mask_slicewise(
    grid_shape: np.ndarray,
    domain_min_padded: np.ndarray,
    voxel_pitch: float,
    center: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """
    Compute ellipsoid mask slice-by-slice to avoid allocating 3 full float64 meshgrid arrays.
    
    This reduces peak memory from 3*N*8 bytes (three float64 grids) to N/grid_shape[0]*8 bytes
    (one 2D float64 slice at a time), where N is the total number of voxels.
    
    Parameters
    ----------
    grid_shape : np.ndarray
        Shape of the output mask (nx, ny, nz)
    domain_min_padded : np.ndarray
        Lower corner of the padded domain
    voxel_pitch : float
        Voxel size
    center : np.ndarray
        Center of ellipsoid (x, y, z)
    radii : np.ndarray
        Semi-axes of ellipsoid (a, b, c)
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates inside ellipsoid
    """
    mask = np.zeros(grid_shape, dtype=bool)
    
    y = np.linspace(
        domain_min_padded[1],
        domain_min_padded[1] + grid_shape[1] * voxel_pitch,
        grid_shape[1]
    )
    z = np.linspace(
        domain_min_padded[2],
        domain_min_padded[2] + grid_shape[2] * voxel_pitch,
        grid_shape[2]
    )
    
    y_term = ((y - center[1]) / radii[1]) ** 2
    z_term = ((z - center[2]) / radii[2]) ** 2
    yz_sum = y_term[:, np.newaxis] + z_term[np.newaxis, :]
    
    for i in range(grid_shape[0]):
        x_val = domain_min_padded[0] + i * voxel_pitch
        x_term = ((x_val - center[0]) / radii[0]) ** 2
        mask[i, :, :] = (x_term + yz_sum) <= 1.0
    
    return mask


def embed_tree_as_negative_space(
    tree_stl_path: Union[str, Path],
    domain: DomainSpec,
    voxel_pitch: float = 1.0,
    margin: float = 0.0,
    dilation_voxels: int = 0,
    smoothing_iters: int = 5,
    output_void: bool = True,
    output_shell: bool = False,
    shell_thickness: float = 2.0,
    stl_units: str = "auto",
    geometry_units: str = CANONICAL_UNIT,
    use_morphological_shell: bool = True,
) -> Dict[str, Optional[trimesh.Trimesh]]:
    """
    Embed a vascular tree STL mesh into a domain as negative space (void).
    
    This creates a solid domain mesh with the tree carved out as a void.
    Useful for creating molds, scaffolds, or perfusion chambers.
    
    **Units**: All spatial parameters are in millimeters by default.
    
    Parameters
    ----------
    tree_stl_path : str or Path
        Path to the vascular tree STL file
    domain : DomainSpec
        Domain specification (BoxDomain or EllipsoidDomain) in millimeters
    voxel_pitch : float
        Voxel size in millimeters (default: 1.0mm)
        Smaller values give higher resolution but slower computation
    margin : float
        Additional margin around tree bounds in millimeters (default: 0)
    dilation_voxels : int
        Number of voxels to dilate the tree by (useful if tree is thin centerlines)
        Default: 0 (no dilation)
    smoothing_iters : int
        Number of smoothing iterations to reduce voxel artifacts (default: 5)
    output_void : bool
        Whether to output the void mesh (tree volume) (default: True)
    output_shell : bool
        Whether to output a shell mesh around the void (default: False)
    shell_thickness : float
        Thickness of shell around void in millimeters (default: 2.0mm)
    stl_units : str
        Units of the input STL file ('mm', 'm', 'auto'). Default: 'auto'
        'auto' will attempt to detect based on bounding box size
    geometry_units : str
        Units for domain and output geometry ('mm', 'm', 'cm'). Default: 'mm'
    use_morphological_shell : bool
        If True (default), use morphological dilation for shell generation which
        is much more memory-efficient. If False, use distance_transform_edt which
        gives continuous thickness in world units but allocates a full float64 grid.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'domain_with_void': trimesh.Trimesh of solid domain with void carved out
        - 'void': trimesh.Trimesh of the void volume (if output_void=True)
        - 'shell': trimesh.Trimesh of shell around void (if output_shell=True)
        - 'metadata': dict with voxel grid info and statistics
    
    Examples
    --------
    >>> from generation.core.domain import BoxDomain
    >>> from generation.core.types import Point3D
    >>> from generation.ops.embedding import embed_tree_as_negative_space
    >>> 
    >>> # Create a box domain (in millimeters)
    >>> domain = BoxDomain.from_center_and_size(
    ...     center=Point3D(0, 0, 0),
    ...     width=100, height=100, depth=100  # 100mm cube
    ... )
    >>> 
    >>> # Embed tree as negative space
    >>> result = embed_tree_as_negative_space(
    ...     tree_stl_path='tree.stl',
    ...     domain=domain,
    ...     voxel_pitch=0.5,  # 0.5mm voxels
    ...     output_void=True,
    ...     output_shell=True,
    ...     stl_units='auto',  # Auto-detect STL units
    ... )
    >>> 
    >>> # Export results
    >>> result['domain_with_void'].export('domain_with_void.stl')
    >>> result['void'].export('void.stl')
    >>> result['shell'].export('shell.stl')
    """
    tree_stl_path = Path(tree_stl_path)
    
    tree_mesh = trimesh.load(tree_stl_path)
    if not isinstance(tree_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh object, got {type(tree_mesh)}")
    
    tree_bounds = tree_mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    tree_min = tree_bounds[0]
    tree_max = tree_bounds[1]
    tree_center = (tree_min + tree_max) / 2
    tree_size = tree_max - tree_min
    
    if stl_units == "auto":
        max_dimension = np.max(tree_size)
        if max_dimension < 1.0:
            stl_units = "m"
            print(f"Auto-detected STL units: meters (max dimension: {max_dimension:.4f}m)")
        elif max_dimension < 1000.0:
            stl_units = "mm"
            print(f"Auto-detected STL units: millimeters (max dimension: {max_dimension:.1f}mm)")
        else:
            stl_units = "mm"
            print(f"Warning: STL max dimension {max_dimension:.1f} is unusually large. Assuming millimeters.")
    
    if stl_units != geometry_units:
        from ..utils.units import convert_length
        scale_factor = convert_length(1.0, stl_units, geometry_units)
        tree_mesh.apply_scale(scale_factor)
        tree_bounds = tree_mesh.bounds
        tree_min = tree_bounds[0]
        tree_max = tree_bounds[1]
        tree_center = (tree_min + tree_max) / 2
        tree_size = tree_max - tree_min
        print(f"Scaled STL from {stl_units} to {geometry_units} (scale factor: {scale_factor})")
    
    if isinstance(domain, BoxDomain):
        domain_min = np.array([domain.x_min, domain.y_min, domain.z_min])
        domain_max = np.array([domain.x_max, domain.y_max, domain.z_max])
    elif isinstance(domain, EllipsoidDomain):
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
        radii = np.array([domain.semi_axis_a, domain.semi_axis_b, domain.semi_axis_c])
        domain_min = center - radii
        domain_max = center + radii
    else:
        raise ValueError(f"Unsupported domain type: {type(domain)}")
    
    if margin > 0:
        domain_min -= margin
        domain_max += margin
    
    grid_shape = np.ceil((domain_max - domain_min) / voxel_pitch).astype(int)
    
    grid_shape = np.maximum(grid_shape, 10)
    
    padding = 2
    grid_shape_padded = grid_shape + 2 * padding
    domain_min_padded = domain_min - padding * voxel_pitch
    
    print(f"Creating voxel grid: {grid_shape_padded} voxels (with {padding}-voxel padding)")
    print(f"Domain bounds: {domain_min} to {domain_max}")
    print(f"Padded bounds: {domain_min_padded} to {domain_min_padded + grid_shape_padded * voxel_pitch}")
    print(f"Voxel pitch: {voxel_pitch}")
    
    domain_mask = np.zeros(grid_shape_padded, dtype=bool)
    
    if isinstance(domain, BoxDomain):
        domain_mask[padding:-padding, padding:-padding, padding:-padding] = True
    elif isinstance(domain, EllipsoidDomain):
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
        radii = np.array([domain.semi_axis_a, domain.semi_axis_b, domain.semi_axis_c])
        
        domain_mask = _compute_ellipsoid_mask_slicewise(
            grid_shape_padded, domain_min_padded, voxel_pitch, center, radii
        )
    
    print(f"Domain mask: {domain_mask.sum()} voxels ({100 * domain_mask.sum() / domain_mask.size:.1f}%)")
    
    tree_voxels, actual_pitch = _voxelized_with_retry(
        tree_mesh,
        pitch=voxel_pitch,
        max_attempts=4,
        factor=1.5,
        log_prefix="[embed_tree_as_negative_space] ",
    )
    if actual_pitch != voxel_pitch:
        print(f"Note: voxel_pitch was adjusted from {voxel_pitch:.4g} to {actual_pitch:.4g} due to memory constraints")
        voxel_pitch = actual_pitch
    tree_mask = tree_voxels.matrix
    
    tree_origin = tree_voxels.transform[:3, 3]
    
    offset_voxels = np.round((tree_origin - domain_min_padded) / voxel_pitch).astype(int)
    
    aligned_tree_mask = np.zeros(grid_shape_padded, dtype=bool)
    
    tree_shape = np.array(tree_mask.shape)
    
    src_start = np.maximum(-offset_voxels, 0)
    dst_start = np.maximum(offset_voxels, 0)
    
    copy_size = np.minimum(
        tree_shape - src_start,  # Remaining tree voxels from src_start
        grid_shape_padded - dst_start   # Remaining domain voxels from dst_start
    )
    
    copy_size = np.maximum(copy_size, 0)
    
    print(f"Alignment diagnostics:")
    print(f"  tree_origin: {tree_origin}")
    print(f"  offset_voxels: {offset_voxels}")
    print(f"  src_start: {src_start}, dst_start: {dst_start}")
    print(f"  copy_size: {copy_size}")
    
    if np.all(copy_size > 0):
        aligned_tree_mask[
            dst_start[0]:dst_start[0] + copy_size[0],
            dst_start[1]:dst_start[1] + copy_size[1],
            dst_start[2]:dst_start[2] + copy_size[2]
        ] = tree_mask[
            src_start[0]:src_start[0] + copy_size[0],
            src_start[1]:src_start[1] + copy_size[1],
            src_start[2]:src_start[2] + copy_size[2]
        ]
    else:
        print(f"WARNING: No overlap between tree and domain!")
        print(f"  Tree bounds: {tree_min} to {tree_max}")
        print(f"  Domain bounds: {domain_min} to {domain_max}")
        print(f"  Possible causes:")
        print(f"    - STL units misdetected (detected: {stl_units})")
        print(f"    - Tree outside domain boundaries")
        print(f"    - Domain too small for tree")
        print(f"  Suggestions:")
        print(f"    - Increase margin parameter")
        print(f"    - Check STL units (set stl_units='mm' or 'm' explicitly)")
        print(f"    - Verify domain dimensions match tree size")
    
    print(f"Tree mask: {aligned_tree_mask.sum()} voxels ({100 * aligned_tree_mask.sum() / aligned_tree_mask.size:.1f}%)")
    
    del tree_voxels, tree_mask
    
    if aligned_tree_mask.sum() == 0 and np.all(copy_size > 0):
        print(f"WARNING: Tree disappeared during voxelization!")
        print(f"  Voxel pitch ({voxel_pitch} mm) may be too coarse for thin vessels")
        print(f"  Suggestions:")
        print(f"    - Reduce voxel_pitch to 0.25-0.5 mm")
        print(f"    - Set dilation_voxels=1-2 to thicken thin features")
        print(f"    - Check that vessel diameters are > 2× voxel_pitch")
    
    if dilation_voxels > 0:
        print(f"Dilating tree by {dilation_voxels} voxels...")
        aligned_tree_mask = ndimage.binary_dilation(
            aligned_tree_mask,
            iterations=dilation_voxels
        )
        print(f"After dilation: {aligned_tree_mask.sum()} voxels")
    
    domain_voxel_count = int(domain_mask.sum())
    
    aligned_tree_mask &= domain_mask
    tree_voxel_count = int(aligned_tree_mask.sum())
    void_mask = aligned_tree_mask
    del aligned_tree_mask
    
    solid_mask = domain_mask & (~void_mask)
    
    print(f"Void mask: {void_mask.sum()} voxels")
    print(f"Solid mask: {solid_mask.sum()} voxels")
    
    if smoothing_iters > 0:
        print(f"Smoothing with {smoothing_iters} iterations...")
        solid_mask = ndimage.binary_closing(solid_mask, iterations=smoothing_iters // 2)
        solid_mask = ndimage.binary_opening(solid_mask, iterations=smoothing_iters // 2)
        solid_mask &= ~void_mask
        print(f"Solid mask after smoothing (with void re-enforced): {solid_mask.sum()} voxels")
    
    result = {}
    
    if solid_mask.any():
        print("Generating domain mesh with marching cubes...")
        verts, faces, _, _ = marching_cubes(
            volume=solid_mask.astype(np.uint8),
            level=0.5,
            spacing=(voxel_pitch, voxel_pitch, voxel_pitch),
            allow_degenerate=False,
        )
        
        verts = verts[:, [2, 1, 0]]
        
        verts += domain_min_padded
        
        domain_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces.astype(np.int64),
            process=False,
        )
        
        domain_mesh.remove_unreferenced_vertices()
        
        if domain_mesh.volume < 0:
            domain_mesh.invert()
        
        trimesh.repair.fix_normals(domain_mesh)
        
        if not domain_mesh.is_watertight:
            trimesh.repair.fill_holes(domain_mesh)
        
        result['domain_with_void'] = domain_mesh
        print(f"Domain mesh: {len(domain_mesh.vertices)} vertices, {len(domain_mesh.faces)} faces")
        print(f"  Watertight: {domain_mesh.is_watertight}, Volume: {domain_mesh.volume:.9f}")
    else:
        result['domain_with_void'] = None
        print("Warning: Solid mask is empty, no domain mesh generated")
    
    if output_void and void_mask.any():
        print("Generating void mesh with marching cubes...")
        verts, faces, _, _ = marching_cubes(
            volume=void_mask.astype(np.uint8),
            level=0.5,
            spacing=(voxel_pitch, voxel_pitch, voxel_pitch),
            allow_degenerate=False,
        )
        
        verts = verts[:, [2, 1, 0]]
        
        verts += domain_min_padded
        
        void_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces.astype(np.int64),
            process=False,
        )
        
        void_mesh.remove_unreferenced_vertices()
        
        if void_mesh.volume < 0:
            void_mesh.invert()
        
        trimesh.repair.fix_normals(void_mesh)
        
        if not void_mesh.is_watertight:
            trimesh.repair.fill_holes(void_mesh)
        
        result['void'] = void_mesh
        print(f"Void mesh: {len(void_mesh.vertices)} vertices, {len(void_mesh.faces)} faces")
        print(f"  Watertight: {void_mesh.is_watertight}, Volume: {void_mesh.volume:.9f}")
    else:
        result['void'] = None
    
    if output_shell and void_mask.any():
        print(f"Generating shell mesh (thickness={shell_thickness} mm)...")
        
        effective_thickness = shell_thickness
        if shell_thickness < voxel_pitch:
            print(f"WARNING: shell_thickness ({shell_thickness} mm) < voxel_pitch ({voxel_pitch} mm)")
            print(f"  Using effective_thickness = {voxel_pitch} mm (1 voxel layer)")
            effective_thickness = voxel_pitch
        
        th_vox = max(1, int(np.ceil(effective_thickness / voxel_pitch)))
        print(f"  Shell thickness: {th_vox} voxels ({effective_thickness} mm)")
        
        if use_morphological_shell:
            print(f"  Using morphological dilation (memory-efficient mode)...")
            shell_mask = ndimage.binary_dilation(void_mask, iterations=th_vox) & domain_mask & (~void_mask)
        else:
            print(f"  Using EDT-based shell (continuous thickness mode)...")
            dist_from_void = ndimage.distance_transform_edt(
                ~void_mask,
                sampling=(voxel_pitch, voxel_pitch, voxel_pitch)
            )
            shell_mask = (dist_from_void <= effective_thickness) & domain_mask & (~void_mask)
            del dist_from_void
            
            if not shell_mask.any() and th_vox >= 1:
                print(f"  EDT-based shell is empty, using morphological fallback...")
                shell_mask = ndimage.binary_dilation(void_mask, iterations=th_vox) & domain_mask & (~void_mask)
        
        if shell_mask.any():
            print(f"  Shell mask: {shell_mask.sum()} voxels")
            verts, faces, _, _ = marching_cubes(
                volume=shell_mask.astype(np.uint8),
                level=0.5,
                spacing=(voxel_pitch, voxel_pitch, voxel_pitch),
                allow_degenerate=False,
            )
            
            verts = verts[:, [2, 1, 0]]
            
            verts += domain_min_padded
            
            shell_mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces.astype(np.int64),
                process=False,
            )
            
            shell_mesh.remove_unreferenced_vertices()
            
            if shell_mesh.volume < 0:
                shell_mesh.invert()
            
            trimesh.repair.fix_normals(shell_mesh)
            
            if not shell_mesh.is_watertight:
                trimesh.repair.fill_holes(shell_mesh)
            
            result['shell'] = shell_mesh
            print(f"Shell mesh: {len(shell_mesh.vertices)} vertices, {len(shell_mesh.faces)} faces")
            print(f"  Watertight: {shell_mesh.is_watertight}, Volume: {shell_mesh.volume:.9f}")
        else:
            result['shell'] = None
            print("WARNING: Shell mask is empty after both EDT and morphological methods")
            print(f"  Possible causes:")
            print(f"    - Void mask is empty (no tree voxels in domain)")
            print(f"    - shell_thickness ({shell_thickness} mm) too small")
            print(f"    - voxel_pitch ({voxel_pitch} mm) too coarse")
            print(f"  Suggestions:")
            print(f"    - Increase shell_thickness to at least {2 * voxel_pitch} mm")
            print(f"    - Reduce voxel_pitch to 0.25-0.5 mm")
            print(f"    - Set dilation_voxels=1-2 to thicken tree")
    else:
        result['shell'] = None
        if output_shell and not void_mask.any():
            print("WARNING: Cannot generate shell - void mask is empty")
            print("  The tree was not successfully voxelized in the domain")
            print("  See warnings above for tree mask alignment issues")
    
    del domain_mask
    
    result['metadata'] = {
        'voxel_pitch': voxel_pitch,
        'grid_shape': grid_shape_padded.tolist(),
        'domain_bounds': {
            'min': domain_min.tolist(),
            'max': domain_max.tolist(),
        },
        'padded_bounds': {
            'min': domain_min_padded.tolist(),
            'max': (domain_min_padded + grid_shape_padded * voxel_pitch).tolist(),
        },
        'tree_bounds': {
            'min': tree_min.tolist(),
            'max': tree_max.tolist(),
        },
        'voxel_counts': {
            'domain': domain_voxel_count,
            'tree': tree_voxel_count,
            'void': int(void_mask.sum()),
            'solid': int(solid_mask.sum()),
        },
        'dilation_voxels': dilation_voxels,
        'smoothing_iters': smoothing_iters,
    }
    
    return result
