from pathlib import Path
from typing import Optional, Sequence, Any, List, Tuple, Dict
import numpy as np
import networkx as nx
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
import trimesh

from .models import ValidationReport, ValidationFlags
from .io.loaders import load_stl_mesh, cad_python_to_stl
from .io.exporters import save_validation_report_json
from .mesh.cleaning import basic_clean
from .mesh.repair import voxel_remesh_and_smooth, meshfix_repair
from .mesh.diagnostics import compute_diagnostics, compute_surface_quality
from .analysis.connectivity import analyze_connectivity_voxel
from .analysis.centerline import extract_centerline_graph
from .analysis.cfd import compute_poiseuille_network


def compute_drift_metrics(
    original_spec: Optional[Dict[str, Any]],
    centerline_graph: nx.Graph,
    connectivity_info: Dict[str, Any],
    min_channel_diameter_threshold: float = 0.0005,
) -> Dict[str, Any]:
    """
    Compute drift-aware validation metrics after embedding/repair.
    
    Priority E: Automatically compute min channel diameter, connectivity,
    and drift vs original network spec. Promotes failures back into
    automation loop with actionable messages.
    
    Parameters
    ----------
    original_spec : dict, optional
        Original network specification with expected values:
        - expected_min_radius: float (in meters)
        - expected_num_inlets: int
        - expected_num_outlets: int
        - expected_connectivity: bool (inlet-outlet connected)
    centerline_graph : nx.Graph
        Extracted centerline graph with radius attributes
    connectivity_info : dict
        Connectivity analysis results
    min_channel_diameter_threshold : float
        Minimum acceptable channel diameter in meters (default: 0.5mm = 0.0005m)
    
    Returns
    -------
    drift_metrics : dict
        Dictionary containing:
        - min_channel_diameter: float (actual minimum)
        - min_channel_diameter_ok: bool
        - connectivity_preserved: bool
        - inlet_outlet_connected: bool
        - drift_warnings: list of str (actionable messages)
        - drift_errors: list of str (critical failures)
        - suggested_fixes: list of str (parameter adjustments)
    """
    drift_warnings = []
    drift_errors = []
    suggested_fixes = []
    
    radii = [
        float(data.get("radius", 0.0))
        for _, data in centerline_graph.nodes(data=True)
        if data.get("radius") is not None and data.get("radius") > 0
    ]
    
    if radii:
        min_radius = min(radii)
        max_radius = max(radii)
        mean_radius = np.mean(radii)
        min_channel_diameter = 2 * min_radius
    else:
        min_radius = 0.0
        max_radius = 0.0
        mean_radius = 0.0
        min_channel_diameter = 0.0
        drift_warnings.append("No valid radii found in centerline graph")
    
    min_channel_diameter_ok = min_channel_diameter >= min_channel_diameter_threshold
    if not min_channel_diameter_ok:
        drift_errors.append(
            f"Min channel diameter {min_channel_diameter*1000:.3f}mm below threshold "
            f"{min_channel_diameter_threshold*1000:.3f}mm"
        )
        suggested_fixes.append(
            f"Increase voxel_pitch or reduce smoothing_iters to preserve small channels"
        )
    
    num_components = connectivity_info.get("num_components", 1)
    connectivity_preserved = num_components == 1
    if not connectivity_preserved:
        drift_errors.append(f"Network split into {num_components} disconnected components")
        suggested_fixes.append(
            "Reduce voxel_pitch or increase dilation_iters to maintain connectivity"
        )
    
    inlet_outlet_connected = connectivity_info.get("inlet_outlet_connected", True)
    if not inlet_outlet_connected:
        drift_errors.append("Inlet and outlet are not connected after repair")
        suggested_fixes.append(
            "Check that inlet/outlet ports are not sealed during embedding"
        )
    
    if original_spec:
        expected_min_radius = original_spec.get("expected_min_radius")
        if expected_min_radius and min_radius > 0:
            radius_drift = (expected_min_radius - min_radius) / expected_min_radius
            if radius_drift > 0.2:
                drift_warnings.append(
                    f"Min radius drifted by {radius_drift*100:.1f}% "
                    f"(expected: {expected_min_radius*1000:.3f}mm, actual: {min_radius*1000:.3f}mm)"
                )
                suggested_fixes.append(
                    f"Reduce smoothing_iters or use constraint-preserving smoothing"
                )
        
        expected_num_inlets = original_spec.get("expected_num_inlets")
        expected_num_outlets = original_spec.get("expected_num_outlets")
        if expected_num_inlets or expected_num_outlets:
            actual_inlets = connectivity_info.get("num_inlets", 0)
            actual_outlets = connectivity_info.get("num_outlets", 0)
            if expected_num_inlets and actual_inlets != expected_num_inlets:
                drift_warnings.append(
                    f"Inlet count changed: expected {expected_num_inlets}, got {actual_inlets}"
                )
            if expected_num_outlets and actual_outlets != expected_num_outlets:
                drift_warnings.append(
                    f"Outlet count changed: expected {expected_num_outlets}, got {actual_outlets}"
                )
    
    return {
        "min_channel_diameter": min_channel_diameter,
        "min_channel_diameter_mm": min_channel_diameter * 1000,
        "min_channel_diameter_ok": min_channel_diameter_ok,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "mean_radius": mean_radius,
        "connectivity_preserved": connectivity_preserved,
        "num_components": num_components,
        "inlet_outlet_connected": inlet_outlet_connected,
        "drift_warnings": drift_warnings,
        "drift_errors": drift_errors,
        "suggested_fixes": suggested_fixes,
        "validation_passed": len(drift_errors) == 0,
    }


def make_scaffold_shell_from_fluid(
    fluid_mask: np.ndarray,
    pitch: float,
    wall_thickness: float = 0.4,
    use_morphological: bool = True,
):
    """
    Build a scaffold as a narrow shell around the fluid domain.

    Parameters
    ----------
    fluid_mask : (nx, ny, nz) bool
        True for fluid voxels (inside channels), False elsewhere.
    pitch : float
        Voxel size in world units (mm, etc.).
    wall_thickness : float
        Desired wall thickness of the scaffold around channels.
    use_morphological : bool
        If True (default), use morphological dilation which is much more
        memory-efficient. If False, use distance_transform_edt which gives
        continuous thickness but allocates a full float64 grid.

    Returns
    -------
    scaffold_mesh : trimesh.Trimesh
        Mesh of the scaffold shell (no big filled-in regions).
    """
    th_vox = max(1, int(np.ceil(wall_thickness / pitch)))
    
    if use_morphological:
        dilated = ndimage.binary_dilation(fluid_mask, iterations=th_vox)
        shell_mask = dilated & (~fluid_mask)
        del dilated
    else:
        dist_vox = distance_transform_edt(~fluid_mask)
        dist = dist_vox * pitch
        del dist_vox
        shell_mask = (~fluid_mask) & (dist <= wall_thickness)
        del dist

    if not shell_mask.any():
        raise RuntimeError("Shell mask is empty; try increasing wall_thickness.")

    vol_uint8 = shell_mask.astype(np.uint8)
    del shell_mask
    
    verts, faces, _, _ = marching_cubes(
        volume=vol_uint8,
        level=0.5,
        spacing=(pitch, pitch, pitch),
    )
    del vol_uint8

    scaffold_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces.astype(np.int64),
        process=False,
    )
    scaffold_mesh.remove_unreferenced_vertices()
    return scaffold_mesh


def validate_and_repair_geometry(
    input_path: str | Path,
    cleaned_stl_path: Optional[str | Path] = None,
    scaffold_stl_path: Optional[str | Path] = None,
    wall_thickness: float = 0.4,
    report_path: Optional[str | Path] = None,
    cad_tessellation_tolerance: float = 0.1,
    voxel_pitch: float = 0.1,
    smooth_iters: int = 40,
    dilation_iters: int = 2,
    inlet_nodes: Optional[Sequence[Any]] = None,
    outlet_nodes: Optional[Sequence[Any]] = None,
) -> Tuple[ValidationReport, nx.Graph]:
    """
    Full pipeline:

    - Load STL or CadQuery .py
    - Basic clean
    - Voxel remesh + smooth
    - MeshFix repair
    - Diagnostics + surface quality (before/after)
    - Connectivity
    - Centerlines
    - Poiseuille network (CFD)
    - Save cleaned STL and JSON report (optional)

    Parameters
    ----------
    input_path : str or Path
        Path to input STL or Python CAD file
    cleaned_stl_path : str or Path, optional
        Path to save cleaned STL
    scaffold_stl_path : str or Path, optional
        Path to save scaffold shell STL
    wall_thickness : float
        Wall thickness for scaffold shell
    report_path : str or Path, optional
        Path to save JSON report
    cad_tessellation_tolerance : float
        Tessellation tolerance for CAD files
    voxel_pitch : float
        Voxel pitch for remeshing
    smooth_iters : int
        Number of smoothing iterations
    dilation_iters : int
        Number of dilation iterations
    inlet_nodes : sequence, optional
        Inlet nodes for CFD analysis
    outlet_nodes : sequence, optional
        Outlet nodes for CFD analysis

    Returns
    -------
    report : ValidationReport
        Validation report with all metrics
    G_centerline : nx.Graph
        Centerline graph with CFD results
    """
    input_path = Path(input_path)

    intermediate_stl_path: Optional[Path]
    if input_path.suffix.lower() == ".stl":
        intermediate_stl_path = input_path
    elif input_path.suffix.lower() == ".py":
        intermediate_stl_path = input_path.with_suffix("_tess.stl")
        cad_python_to_stl(input_path, intermediate_stl_path, tolerance=cad_tessellation_tolerance)
    else:
        raise ValueError(f"Unsupported input type: {input_path.suffix}")

    if cleaned_stl_path is None:
        cleaned_stl_path = intermediate_stl_path.with_name(
            intermediate_stl_path.stem + "_cleaned.stl"
        )
    cleaned_stl_path = Path(cleaned_stl_path)

    mesh_original = load_stl_mesh(intermediate_stl_path)
    diag_before = compute_diagnostics(mesh_original)
    surf_before = compute_surface_quality(mesh_original)

    mesh_clean = basic_clean(mesh_original)
    diag_after_basic = compute_diagnostics(mesh_clean)

    mesh_voxel = voxel_remesh_and_smooth(
        mesh_clean,
        pitch=voxel_pitch,
        smooth_iters=smooth_iters,
        dilation_iters=dilation_iters,
        closing_iters=1,
        opening_iters=1,
    )

    diag_after_voxel = compute_diagnostics(mesh_voxel)

    mesh_repaired = meshfix_repair(mesh_voxel)

    diag_after_repair = compute_diagnostics(mesh_repaired)
    surf_after = compute_surface_quality(mesh_repaired)

    flags_list: List[str] = []
    neg_flags: List[str] = []

    if not diag_before.watertight and diag_after_repair.watertight:
        flags_list.append("fixed_watertightness")
    if diag_before.num_components != diag_after_repair.num_components:
        flags_list.append("changed_component_count")

    if not diag_after_repair.watertight:
        neg_flags.append("not_watertight")
    if diag_after_repair.num_components > 1:
        neg_flags.append("multiple_components")
    if diag_after_repair.non_manifold_edges > 0:
        neg_flags.append("non_manifold_edges")
    if diag_after_repair.degenerate_faces > 0:
        neg_flags.append("degenerate_faces")

    if any(f in ("not_watertight", "multiple_components") for f in neg_flags):
        status = "fail"
    elif neg_flags:
        status = "warnings"
    else:
        status = "ok"

    flags = ValidationFlags(
        status=status,
        flags=flags_list + neg_flags,
    )

    connectivity_info, fluid_mask, bbox_min, spacing = analyze_connectivity_voxel(
        mesh_repaired,
        pitch=voxel_pitch,
    )

    centerline_warnings = []
    try:
        G_centerline, centerline_meta = extract_centerline_graph(
            fluid_mask,
            bbox_min=bbox_min,
            spacing=spacing,
        )
    except RuntimeError as e:
        if "Skeletonization produced zero voxels" in str(e):
            warning_msg = f"Centerline extraction failed at voxel_pitch={voxel_pitch}: {e}. Proceeding without centerline."
            centerline_warnings.append(warning_msg)
            print(f"[WARNING] {warning_msg}")
            G_centerline = nx.Graph()
            centerline_meta = {
                "bbox_min": bbox_min.tolist(),
                "spacing": spacing.tolist(),
                "num_nodes": 0,
                "num_edges": 0,
                "grid_shape": list(fluid_mask.shape),
                "warning": warning_msg,
            }
        else:
            raise

    pitch_mean = float(np.mean(spacing))
    scaffold_mesh = make_scaffold_shell_from_fluid(fluid_mask, pitch_mean, wall_thickness)

    if scaffold_stl_path is None:
        intermediate_stl_path = Path(intermediate_stl_path)
        scaffold_stl_path = intermediate_stl_path.with_name(
            intermediate_stl_path.stem + "_scaffold_shell.stl"
        )
    else:
        scaffold_stl_path = Path(scaffold_stl_path)

    scaffold_mesh.export(scaffold_stl_path)
    print("Scaffold STL saved at:", scaffold_stl_path)

    radii = [float(data.get("radius", 0.0)) for _, data in G_centerline.nodes(data=True) if data.get("radius") is not None]
    centerline_summary = {
        "meta": centerline_meta,
        "radius_min": float(min(radii)) if radii else 0.0,
        "radius_max": float(max(radii)) if radii else 0.0,
        "radius_mean": float(np.mean(radii)) if radii else 0.0,
    }

    drift_metrics = compute_drift_metrics(
        original_spec=None,
        centerline_graph=G_centerline,
        connectivity_info=connectivity_info,
        min_channel_diameter_threshold=0.0005,
    )
    
    if drift_metrics["drift_errors"]:
        print(f"[DRIFT VALIDATION] Errors detected:")
        for err in drift_metrics["drift_errors"]:
            print(f"  - {err}")
        print(f"[DRIFT VALIDATION] Suggested fixes:")
        for fix in drift_metrics["suggested_fixes"]:
            print(f"  - {fix}")
    elif drift_metrics["drift_warnings"]:
        print(f"[DRIFT VALIDATION] Warnings:")
        for warn in drift_metrics["drift_warnings"]:
            print(f"  - {warn}")

    network_results = compute_poiseuille_network(
        G_centerline,
        mu=1.0,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        pin=1.0,
        pout=0.0,
    )

    poiseuille_summary = network_results["summary"]

    mesh_repaired.export(cleaned_stl_path)

    report = ValidationReport(
        input_file=str(input_path),
        intermediate_stl=str(intermediate_stl_path),
        cleaned_stl=str(cleaned_stl_path),
        scafold_stl=str(scaffold_stl_path),
        before=diag_before,
        after_basic_clean=diag_after_basic,
        after_voxel=diag_after_voxel,
        after_repair=diag_after_repair,
        flags=flags,
        surface_before=surf_before,
        surface_after=surf_after,
        connectivity=connectivity_info,
        centerline_summary=centerline_summary,
        poiseuille_summary=poiseuille_summary,
        drift_metrics=drift_metrics,
    )

    if report_path is not None:
        save_validation_report_json(
            report,
            report_path,
            centerline_graph=G_centerline,
            node_pressures=network_results["node_pressures"],
            edge_flows=network_results["edge_flows"],
        )

    return report, G_centerline
