import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh.smoothing as tmsmooth
from tqdm import tqdm

from .voxel_utils import voxelized_with_retry

# pymeshfix is an optional dependency - only required for meshfix_repair()
_PYMESHFIX_AVAILABLE = False
_PYMESHFIX_IMPORT_ERROR = None
try:
    from pymeshfix import MeshFix
    _PYMESHFIX_AVAILABLE = True
except ImportError as e:
    _PYMESHFIX_IMPORT_ERROR = e


def _require_pymeshfix():
    """Raise ImportError with helpful message if pymeshfix is not available."""
    if not _PYMESHFIX_AVAILABLE:
        raise ImportError(
            "pymeshfix is required for mesh repair operations but is not installed. "
            "Install it with: pip install pymeshfix\n"
            "Note: pymeshfix requires a C++ compiler. On Ubuntu: apt-get install build-essential\n"
            f"Original error: {_PYMESHFIX_IMPORT_ERROR}"
        )


def auto_adjust_voxel_pitch(
    mesh: trimesh.Trimesh,
    requested_pitch: float,
    max_voxels: float = 3e7,
    min_pitch: float = 1e-5,
) -> float:
    """
    Given a mesh and requested voxel pitch, compute a safe pitch
    such that (dx*dy*dz) / pitch^3 <= max_voxels.

    Returns the adjusted pitch (>= requested_pitch, >= min_pitch).
    """
    extents = mesh.bounding_box.extents.astype(float)
    dx, dy, dz = extents
    volume_box = dx * dy * dz

    if max_voxels <= 0 or volume_box <= 0:
        return max(requested_pitch, min_pitch)

    pitch_min = (volume_box / max_voxels) ** (1.0 / 3.0)

    adjusted = max(requested_pitch, pitch_min, min_pitch)
    return float(adjusted)


def voxel_remesh_and_smooth(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
    smooth_iters: int = 40,
    use_taubin: bool = True,
    dilation_iters: int = 1,
    closing_iters: int = 1,
    opening_iters: int = 1,
    max_voxels: float = 3e7,
) -> trimesh.Trimesh:
    """
    Voxel remesh + morphology + smoothing.

    Design goals:
      - produce a SINGLE, well-connected lumen
      - aggressively smooth away jaggies & tiny artifacts
      - do NOT try to preserve exact volume

    Steps:
      1) Auto-adjust pitch to keep voxel grid size sane.
      2) Voxelize to binary mask, relax pitch if trimesh complains.
      3) Keep only largest voxel connected component.
      4) Closing (fills small gaps) and opening (removes spikes).
      5) Dilation (thicken thin branches a bit).
      6) Marching cubes back to surface.
      7) Taubin or Laplacian smoothing.
    """
    mesh = mesh.copy()
    
    extents = mesh.bounding_box.extents.astype(float)
    Lmax = float(np.max(extents))
    
    target_grid_resolution = 256
    recommended_pitch = Lmax / target_grid_resolution
    
    if pitch > recommended_pitch * 2:
        print(
            f"[voxel_remesh_and_smooth] WARNING: voxel_pitch={pitch:.4g} is too coarse "
            f"for mesh size (Lmax={Lmax:.4g}). Recommended pitch: {recommended_pitch:.4g}. "
            f"Automatically reducing pitch to prevent geometry loss."
        )
        pitch = recommended_pitch

    pitch_eff = auto_adjust_voxel_pitch(
        mesh, requested_pitch=pitch, max_voxels=max_voxels
    )

    vox = voxelized_with_retry(
        mesh,
        pitch=pitch_eff,
        max_attempts=4,
        factor=1.5,
        log_prefix="[voxel_remesh_and_smooth] ",
    )

    vox_filled = vox.fill()
    fluid_mask = vox_filled.matrix.astype(bool)
    del vox, vox_filled

    if not fluid_mask.any():
        raise RuntimeError(
            "Voxelization produced an empty fluid mask. Check pitch/scale."
        )
    
    total_voxels = fluid_mask.size
    initial_filled = np.count_nonzero(fluid_mask)
    filled_fraction = initial_filled / total_voxels
    
    print(f"[voxel_remesh_and_smooth] Initial voxelization: {initial_filled} voxels ({filled_fraction:.2%})")

    structure = ndimage.generate_binary_structure(3, 1)
    labels = np.zeros(fluid_mask.shape, dtype=np.int32)
    num_labels = ndimage.label(fluid_mask, structure=structure, output=labels)[1]
    if num_labels > 1:
        sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
        largest_label = 1 + int(np.argmax(sizes))
        fluid_mask = labels == largest_label
        after_component = np.count_nonzero(fluid_mask)
        print(f"[voxel_remesh_and_smooth] After keeping largest component: {after_component} voxels")
    del labels

    if closing_iters > 0:
        print(f"[voxel_remesh_and_smooth] Applying binary closing ({closing_iters} iterations)...")
        for _ in tqdm(range(closing_iters), desc="Closing", unit="iter"):
            fluid_mask = ndimage.binary_closing(fluid_mask, iterations=1)
        after_closing = np.count_nonzero(fluid_mask)
        print(f"[voxel_remesh_and_smooth] After closing: {after_closing} voxels")

    if opening_iters > 0:
        print(f"[voxel_remesh_and_smooth] Applying binary opening ({opening_iters} iterations)...")
        for _ in tqdm(range(opening_iters), desc="Opening", unit="iter"):
            fluid_mask = ndimage.binary_opening(fluid_mask, iterations=1)
        after_opening = np.count_nonzero(fluid_mask)
        print(f"[voxel_remesh_and_smooth] After opening: {after_opening} voxels")
        
        if after_opening == 0:
            print(
                f"[voxel_remesh_and_smooth] WARNING: Opening removed all voxels. "
                f"Skipping opening and reducing dilation."
            )
            fluid_mask = ndimage.binary_closing(fluid_mask, iterations=closing_iters)
            dilation_iters = max(0, dilation_iters - 1)

    if dilation_iters > 0:
        print(f"[voxel_remesh_and_smooth] Applying binary dilation ({dilation_iters} iterations)...")
        for _ in tqdm(range(dilation_iters), desc="Dilation", unit="iter"):
            fluid_mask = ndimage.binary_dilation(fluid_mask, iterations=1)
        after_dilation = np.count_nonzero(fluid_mask)
        print(f"[voxel_remesh_and_smooth] After dilation: {after_dilation} voxels")

    volume_uint8 = fluid_mask.astype(np.uint8)
    if volume_uint8.max() == 0:
        raise RuntimeError(
            f"All voxels were removed after morphology; nothing left to remesh.\n"
            f"Mesh bounding box: {extents}\n"
            f"Voxel pitch used: {pitch_eff:.6g}\n"
            f"Grid resolution: {Lmax/pitch_eff:.1f} voxels along longest axis\n"
            f"Initial filled voxels: {initial_filled}\n"
            f"Suggestion: Reduce voxel_pitch to {recommended_pitch:.6g} or smaller, "
            f"and/or reduce opening_iters and dilation_iters."
        )

    verts, faces, _, _ = marching_cubes(
        volume=volume_uint8,
        level=0.5,
        spacing=(pitch_eff, pitch_eff, pitch_eff),
    )

    mesh_voxel = trimesh.Trimesh(
        vertices=verts,
        faces=faces.astype(np.int64),
        process=False,
    )
    mesh_voxel.remove_unreferenced_vertices()

    if smooth_iters > 0:
        try:
            print(f"[voxel_remesh_and_smooth] Applying {'Taubin' if use_taubin else 'Laplacian'} smoothing ({smooth_iters} iterations)...")
            with tqdm(total=smooth_iters, desc="Smoothing", unit="iter") as pbar:
                if use_taubin:
                    for _ in range(smooth_iters):
                        tmsmooth.filter_taubin(
                            mesh_voxel,
                            lamb=0.5,
                            nu=-0.53,
                            iterations=1,
                        )
                        pbar.update(1)
                else:
                    for _ in range(smooth_iters):
                        tmsmooth.filter_laplacian(
                            mesh_voxel,
                            lamb=0.5,
                            iterations=1,
                        )
                        pbar.update(1)
        except Exception as e:
            print(f"[voxel_remesh_and_smooth] smoothing failed: {e}")

    mesh_voxel.remove_unreferenced_vertices()
    return mesh_voxel


def match_volume(
    mesh: trimesh.Trimesh,
    target_volume: float,
    max_scale_factor: float = 3.0,
) -> trimesh.Trimesh:
    """
    Uniformly scale mesh so its volume approx equals target_volume.

    - Computes scale = (target / current)^(1/3).
    - Clamps scale to [1/max_scale_factor, max_scale_factor] to avoid
      insane jumps (e.g., if current volume is tiny or huge).
    """
    mesh = mesh.copy()

    try:
        vol_now = float(mesh.volume)
    except Exception:
        vol_now = 0.0

    if vol_now <= 0 or target_volume <= 0:
        return mesh

    raw_scale = (target_volume / vol_now) ** (1.0 / 3.0)

    min_s = 1.0 / max_scale_factor
    max_s = max_scale_factor
    scale = float(np.clip(raw_scale, min_s, max_s))

    center = mesh.center_mass
    mesh.vertices -= center
    mesh.vertices *= scale
    mesh.vertices += center

    mesh.remove_unreferenced_vertices()
    return mesh


def meshfix_repair(
    mesh: trimesh.Trimesh,
    keep_largest_component: bool = True,
) -> trimesh.Trimesh:
    """
    Run MeshFix to enforce watertightness and fix topological issues,
    then optionally keep only the largest connected component.

    This version favors a single, clean lumen over preserving
    extra tiny islands or exact original volume.
    
    Requires pymeshfix to be installed. Install with: pip install pymeshfix
    """
    _require_pymeshfix()
    
    mesh = mesh.copy()

    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)

    fixer = MeshFix(v, f)
    fixer.repair(
        verbose=False,
        joincomp=True,
        remove_smallest_components=False,
    )

    if hasattr(fixer, "points"):
        v_repaired = np.asarray(fixer.points, dtype=float)
    else:
        v_repaired = np.asarray(fixer.v, dtype=float)

    if hasattr(fixer, "faces"):
        f_repaired = np.asarray(fixer.faces, dtype=np.int64)
    else:
        f_repaired = np.asarray(fixer.f, dtype=np.int64)

    repaired = trimesh.Trimesh(
        vertices=v_repaired,
        faces=f_repaired,
        process=False,
    )
    repaired.remove_unreferenced_vertices()

    if keep_largest_component:
        comps = repaired.split(only_watertight=False)
        if len(comps) > 1:
            largest = max(comps, key=lambda c: c.faces.shape[0])
            repaired = largest.copy()
            repaired.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(repaired, multibody=True)
        trimesh.repair.fix_winding(repaired)
    except Exception:
        pass

    repaired.remove_unreferenced_vertices()
    return repaired


def validate_and_repair_geometry(
    input_path: str,
    report_path: str = None,
    voxel_pitch: float = None,
    opening_iters: int = 2,
    dilation_iters: int = 3,
    smoothing_method: str = "taubin",
    smoothing_iters: int = 10,
):
    """
    Validate and repair geometry from STL file with auto voxel pitch selection.
    
    This is a convenience wrapper around voxel_remesh_and_smooth that:
    1. Loads the mesh from file
    2. Auto-selects voxel pitch if not provided (D2: Auto-pitch selection)
    3. Applies voxel remeshing and smoothing
    4. Optionally saves a report
    
    Parameters
    ----------
    input_path : str
        Path to input STL file
    report_path : str, optional
        Path to save report (not implemented yet)
    voxel_pitch : float, optional
        Voxel pitch in meters. If None, auto-selects based on mesh size (default: None)
    opening_iters : int
        Binary opening iterations (default: 2)
    dilation_iters : int
        Binary dilation iterations (default: 3)
    smoothing_method : str
        "taubin" or "laplacian" (default: "taubin")
    smoothing_iters : int
        Number of smoothing iterations (default: 10)
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Repaired mesh
    report : dict or None
        Report dictionary (not implemented yet, returns None)
    """
    import trimesh
    import numpy as np
    
    # Load mesh
    mesh_orig = trimesh.load(input_path, force="mesh")
    
    # D2: Auto-select voxel pitch if not provided
    if voxel_pitch is None:
        extents = mesh_orig.bounding_box.extents.astype(float)
        Lmax = float(np.max(extents))
        target_resolution = 256
        voxel_pitch = Lmax / target_resolution
        print(f"Auto-selected voxel_pitch = {voxel_pitch:.6f}m for {target_resolution} voxel resolution")
    
    # Apply voxel remeshing
    use_taubin = (smoothing_method.lower() == "taubin")
    
    mesh_repaired = voxel_remesh_and_smooth(
        mesh_orig,
        pitch=voxel_pitch,
        smooth_iters=smoothing_iters,
        use_taubin=use_taubin,
        dilation_iters=dilation_iters,
        opening_iters=opening_iters,
    )
    
    # Apply MeshFix for watertightness
    mesh_final = meshfix_repair(mesh_repaired, keep_largest_component=True)
    
    # TODO: Generate report if report_path is provided
    report = None
    
    return mesh_final, report
