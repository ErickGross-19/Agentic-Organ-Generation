#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Cell 1: imports, install, and core library imports

import pathlib
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np

REPO_ROOT = pathlib.Path(".").resolve()
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def ensure_vascular_packages_installed() -> None:
    """Ensure vascular_lib and vascular_network are importable (editable install)."""
    try:
        import vascular_lib  # type: ignore  # noqa: F401
        import vascular_network  # type: ignore  # noqa: F401
        print("✅ vascular_lib and vascular_network already importable")
        return
    except ImportError:
        print("⚙️ Installing dependencies and package (editable mode)...")

    requirements = REPO_ROOT / "requirements.txt"
    if requirements.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)]
    )

    import vascular_lib  # noqa: F401, E402
    import vascular_network  # noqa: F401, E402
    print("✅ Installation complete")


ensure_vascular_packages_installed()

# --- Core imports from vascular_lib / vascular_network ---

from vascular_lib.core import EllipsoidDomain
from vascular_lib.core.domain import BoxDomain
from vascular_lib.core.types import Direction3D
from vascular_lib.ops.build import create_network, add_inlet, add_outlet
from vascular_lib.ops.space_colonization import SpaceColonizationParams, space_colonization_step
from vascular_lib.ops.growth import grow_branch
from vascular_lib.rules.constraints import BranchingConstraints
from vascular_lib.adapters.mesh_adapter import export_stl
from vascular_lib.ops.embedding import embed_tree_as_negative_space

from vascular_lib.analysis.coverage import compute_coverage
from vascular_lib.analysis.flow import estimate_flows, check_hemodynamic_plausibility

from vascular_network.pipeline import validate_and_repair_geometry

import sys
import subprocess
import pathlib
try:
        import trimesh  # type: ignore
except ImportError:
        print("Installing trimesh[all]...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh[all]"])
        import trimesh  # type: ignore


# In[5]:


# -------------------------------
# DESIGN: dual tree in ~25 mm box
# -------------------------------

# Domain: approximate 25 x 25 x 25 mm as an ellipsoid with semi-axes 12.5 mm
L = 0.025  # 25 mm in meters
domain = EllipsoidDomain(
    semi_axis_a=L / 2.0,
    semi_axis_b=L / 2.0,
    semi_axis_c=L / 2.0,
)

network = create_network(
    domain,
    metadata={"name": "demo_dual_tree_box", "units": "m"},
    seed=42,
)

# Inlet/outlet: 2 mm diameter => 1 mm radius in meters
root_radius_m = 0.005  # 1 mm

inlet_result = add_inlet(
    network,
    position=(-0.9 * L / 2.0, 0.0, 0.0),
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),
    radius=root_radius_m,
    vessel_type="arterial",
)
inlet_id = inlet_result.new_ids["node"]

outlet_result = add_outlet(
    network,
    position=(+0.9 * L / 2.0, 0.0, 0.0),
    direction=Direction3D(dx=-1.0, dy=0.0, dz=0.0),
    radius=root_radius_m,
    vessel_type="venous",
)
outlet_id = outlet_result.new_ids["node"]

print("=== Initial network ===")
print(f"Inlet node id : {inlet_id}")
print(f"Outlet node id: {outlet_id}")
print(f"Nodes         : {len(network.nodes)}")
print(f"Segments      : {len(network.segments)}")

# -------------------------------
# TISSUE POINTS: denser near center
# -------------------------------

rng = np.random.default_rng(123)

# Uniform samples inside domain
tissue_all = domain.sample_points(n_points=10000, seed=123)

# Bias probabilities toward x ≈ 0 -> denser in the middle
sigma = (L / 2.0) / 5.0  # width of central region
weights = np.exp(-(tissue_all[:, 0] ** 2) / (2.0 * sigma**2))
weights = weights / weights.max()
mask = rng.random(len(tissue_all)) < weights
tissue_all = tissue_all[mask]

# Left/right split so trees grow towards each other
tissue_left = tissue_all[tissue_all[:, 0] <= 0.0]
tissue_right = tissue_all[tissue_all[:, 0] > 0.0]

print("\n=== Tissue points ===")
print(f"Total points : {len(tissue_all)}")
print(f"Left  points : {len(tissue_left)}")
print(f"Right points : {len(tissue_right)}")

# Positions of inlet/outlet as numpy arrays
inlet_pos = np.array([-0.9 * L / 2.0, 0.0, 0.0])
outlet_pos = np.array([+0.9 * L / 2.0, 0.0, 0.0])

def min_dist_to(points, p):
    if len(points) == 0:
        return None
    pts = np.asarray(points)
    return np.linalg.norm(pts - p[None, :], axis=1).min()

print("\n=== Distance diagnostics ===")
print("Left points :", len(tissue_left))
print("Right points:", len(tissue_right))
print("Min dist(left → inlet)  :", min_dist_to(tissue_left, inlet_pos))
print("Min dist(right → outlet):", min_dist_to(tissue_right, outlet_pos))


# -------------------------------
# GROW: arterial tree (left)
# -------------------------------

params_art = SpaceColonizationParams(
    # spatial scale for L = 0.1 (10 cm)
    influence_radius = 0.012,   # 12 mm: still sees a decent neighborhood, but less global
    kill_radius      = 0.004,   # 4 mm: points get removed more aggressively near branches
    step_size        = 0.0020,  # 2 mm steps: more curvature, more segments
    min_radius       = 0.00010,
    taper_factor     = 0.97,
    vessel_type      = "arterial",
    max_steps        = 100,

    # directional constraints: weaker cone, more freedom
    preferred_direction = (1.0, 0.0, 0.0),
    directional_bias    = 0.25,   # was 0.4 → now 25% weight on preferred dir
    max_deviation_deg   = 110.0,  # allow much wider spread
    smoothing_weight    = 0.2,    # a bit less “stiffness”

    # bifurcation controls: fewer kids, deeper, more selective
    encourage_bifurcation=True,
    max_children_per_node=2,           # at most bifurcations, not 4-way stars
    min_attractions_for_bifurcation=6, # need more distinct pull to branch
    bifurcation_angle_threshold_deg=30.0,  # only branch when directions truly diverge
    bifurcation_probability=0.6,      # not every eligible node branches
)

art_result = space_colonization_step(
    network,
    tissue_points=tissue_left,
    params=params_art,
    seed=67,
)

print("\n=== Arterial colonization ===")
print(art_result)
print("Status      :", art_result.status)
print("Message     :", art_result.message)
print("Nodes grown :", art_result.metadata.get("nodes_grown"))
print("Steps taken :", art_result.metadata.get("steps_taken"))

# -------------------------------
# GROW: venous tree (right)
# -------------------------------

params_ven = SpaceColonizationParams(
    # spatial scale for L = 0.1 (10 cm)
    influence_radius = 0.012,   # 12 mm: still sees a decent neighborhood, but less global
    kill_radius      = 0.004,   # 4 mm: points get removed more aggressively near branches
    step_size        = 0.0020,  # 2 mm steps: more curvature, more segments
    min_radius       = 0.00010,
    taper_factor     = 0.97,
    vessel_type      = "venous",
    max_steps        = 100,

    # directional constraints: weaker cone, more freedom
    preferred_direction = (1.0, 0.0, 0.0),
    directional_bias    = 0.25,   # was 0.4 → now 25% weight on preferred dir
    max_deviation_deg   = 110.0,  # allow much wider spread
    smoothing_weight    = 0.2,    # a bit less “stiffness”

    # bifurcation controls: fewer kids, deeper, more selective
    encourage_bifurcation=True,
    max_children_per_node=2,           # at most bifurcations, not 4-way stars
    min_attractions_for_bifurcation=6, # need more distinct pull to branch
    bifurcation_angle_threshold_deg=30.0,  # only branch when directions truly diverge
    bifurcation_probability=0.6,      # not every eligible node branches
)

ven_result = space_colonization_step(
    network,
    tissue_points=tissue_right,
    params=params_ven,
    seed=456,
)

print("\n=== Venous colonization ===")
print("Status      :", ven_result.status)
print("Message     :", ven_result.message)
print("Nodes grown :", ven_result.metadata.get("nodes_grown"))
print("Steps taken :", ven_result.metadata.get("steps_taken"))

print("\n=== Final network ===")
print(f"Nodes    : {len(network.nodes)}")
print(f"Segments : {len(network.segments)}")

# -------------------------------
# ANALYSIS: coverage & flow
# -------------------------------
from collections import defaultdict, deque

# -------------------------------
# EXPORT: STL mesh for printing
# -------------------------------

output_dir = REPO_ROOT / "output"
output_dir.mkdir(exist_ok=True)

stl_path = output_dir / "demo_dual_tree_box.stl"
mesh_result = export_stl(
    network,
    output_path=str(stl_path),
    mode="robust",   # "robust" uses boolean unions; slower but cleaner
    repair=True,   # try meshfix repair if not watertight
)

print("\n=== Mesh export ===")
print("Status :", mesh_result.status)
print("Message:", mesh_result.message)
print("STL    :", mesh_result.metadata.get("output_path"))

# -------------------------------
# VALIDATION: geometric pipeline
# -------------------------------

output_dir = REPO_ROOT / "output"
output_dir.mkdir(exist_ok=True)
report_path = output_dir / "demo_dual_tree_box_report.json"

# IMPORTANT: explicit, fine voxel size (in meters)
VOXEL_PITCH = 1.0e-4      # 0.1 mm
SMOOTH_ITERS = 20
DILATION_ITERS = 1

print("\n=== Validation + repair (voxel remesh) ===")
print(f"Using voxel_pitch = {VOXEL_PITCH} m")

try:
    report, G_centerline = validate_and_repair_geometry(
        input_path=str(stl_path),
        report_path=str(report_path),
        voxel_pitch=VOXEL_PITCH,
        smooth_iters=SMOOTH_ITERS,
        dilation_iters=DILATION_ITERS,
    )

    print("\n=== Validation summary (after repair) ===")
    print(f"Watertight          : {report.after_repair.watertight}")
    print(f"Volume [mm^3]       : {report.after_repair.volume}")
    print(f"Components          : {report.after_repair.num_components}")
    print(f"Flags/status        : {report.flags.status} | {report.flags.flags}")
    print(f"Centerline segments : {report.centerline_summary.get('num_segments')}")
    print(f"CFD inlet flow [m^3/s]  : {report.poiseuille_summary.get('total_inlet_flow')}")
    print(f"CFD outlet flow [m^3/s] : {report.poiseuille_summary.get('total_outlet_flow')}")
    print(f"CFD flow balance error  : {report.poiseuille_summary.get('flow_balance_error_fraction')}")

except RuntimeError as e:
    # This catches the "All voxels were removed after morphology" case and lets you continue.
    print("⚠️ Voxel remesh failed with RuntimeError:")
    print(e)
    print("Skipping voxel-based remesh/repair for now; you can still use the raw STL.")

print("\n=== Outputs ===")
print(f"STL mesh      : {stl_path}")
print(f"JSON report   : {report_path}")

mesh = trimesh.load_mesh(stl_path, process=False)
print(mesh)

# Show in notebook (or fall back if needed)
scene = trimesh.Scene(mesh)
try:
        scene.show(jupyter=True)
except Exception as e:
        print("⚠️ Inline viewer failed, trying generic viewer...")
        print(f"Error: {e}")
        scene.show()


# In[ ]:


# ----------------------------------------------------------------------------
# DESIGN: dual trees in an ellipsoidal domain
# ----------------------------------------------------------------------------

# Ellipsoid size (in meters).
# Here: ~40 x 25 x 25 mm (0.04 x 0.025 x 0.025 m) ellipsoid.
a = 4   # 40 mm along x
b = 25  # 25 mm along y
c = 25  # 25 mm along z

domain = EllipsoidDomain(
    semi_axis_a=a,
    semi_axis_b=b,
    semi_axis_c=c,
)

network = create_network(
    domain,
    metadata={"name": "dual_tree_ellipsoid", "units": "m"},
    seed=42,
)

# Inlet/outlet: 2 mm diameter => 1 mm radius in meters
root_radius_m = 1  # 1 mm

# Place them near the ±x poles of the ellipsoid, slightly inside the surface
inlet_pos  = (-0.9 * a, 0.0, 0.0)  # arterial side
outlet_pos = (+0.9 * a, 0.0, 0.0)  # venous side

inlet_result = add_inlet(
    network,
    position=inlet_pos,
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),   # grow inward (+x)
    radius=root_radius_m,
    vessel_type="arterial",
)
inlet_id = inlet_result.new_ids["node"]

outlet_result = add_outlet(
    network,
    position=outlet_pos,
    direction=Direction3D(dx=-1.0, dy=0.0, dz=0.0),  # grow inward (-x)
    radius=root_radius_m,
    vessel_type="venous",
)
outlet_id = outlet_result.new_ids["node"]

print("=== Initial network (ellipsoid) ===")
print(f"Inlet node id : {inlet_id}")
print(f"Outlet node id: {outlet_id}")
print(f"Nodes         : {len(network.nodes)}")
print(f"Segments      : {len(network.segments)}")

# ----------------------------------------------------------------------------
# PHASE 1: trunk + bifurcating branches (before SC)
# ---# ----------------------------------------------------------------------------
# PHASE 1: trunk + bifurcating branches (before SC)
# ----------------------------------------------------------------------------

# Constraints used by grow_branch for the initial trunks + bifurcations
trunk_constraints = BranchingConstraints(
    min_radius          = 0.10,  # 0.10 m
    max_radius          = 0.70,  # 0.70 m
    max_branch_order    = 6,
    min_segment_length  = 2.0,   # 2 m
    max_segment_length  = 15.0,  # 15 m
    max_branch_angle_deg= 80.0,
    curvature_limit_deg = 30.0,
)

def grow_trunk_and_bifurcations(root_node_id: int, base_dir: np.ndarray, label: str):
    """
    From the inlet/outlet:
      - grow a short straight trunk (3 segments) along base_dir
      - then create 2 bifurcating branches and extend each for a couple of segments
    """
    trunk_len = 4.0   # 4 m per trunk segment
    trunk_segments = 3

    last_id = root_node_id

    # Straight trunk
    for i in range(trunk_segments):
        result = grow_branch(
            network,
            from_node_id=last_id,
            length=trunk_len,
            direction=Direction3D(
                dx=float(base_dir[0]),
                dy=float(base_dir[1]),
                dz=float(base_dir[2]),
            ),
            target_radius=None,        # use current radius stored at node
            constraints=trunk_constraints,
            check_collisions=True,
        )
        if not result.is_success():
            print(f"[{label}] trunk segment {i} failed: {result.message}")
            break
        last_id = result.new_ids["node"]

    trunk_tip_id = last_id

    # Make two diverging branch directions in the y-z plane
    base_dir_norm = base_dir / np.linalg.norm(base_dir)
    bend_vec1 = np.array([0.0,  0.4,  0.3])
    bend_vec2 = np.array([0.0, -0.4, -0.3])

    dir1 = base_dir_norm + bend_vec1
    dir2 = base_dir_norm + bend_vec2
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)

    branch_len = 3.0  # 3 m per segment
    branch_steps = 3

    for k, dvec in enumerate((dir1, dir2)):
        parent_id = trunk_tip_id
        for j in range(branch_steps):
            result = grow_branch(
                network,
                from_node_id=parent_id,
                length=branch_len,
                direction=Direction3D(
                    dx=float(dvec[0]),
                    dy=float(dvec[1]),
                    dz=float(dvec[2]),
                ),
                target_radius=None,
                constraints=trunk_constraints,
                check_collisions=True,
            )
            if not result.is_success():
                print(f"[{label}] branch {k}, segment {j} failed: {result.message}")
                break
            parent_id = result.new_ids["node"]

    print(
        f"[{label}] Trunk + branches complete. "
        f"Current nodes={len(network.nodes)}, segments={len(network.segments)}"
    )


# Arterial: trunk + bifurcations from left, pointing +x
grow_trunk_and_bifurcations(
    root_node_id=inlet_id,
    base_dir=np.array([1.0, 0.0, 0.0]),
    label="arterial",
)

# Venous: trunk + bifurcations from right, pointing -x
grow_trunk_and_bifurcations(
    root_node_id=outlet_id,
    base_dir=np.array([-1.0, 0.0, 0.0]),
    label="venous",
)

print("\n=== After deterministic trunks + branches ===")
print(f"Nodes    : {len(network.nodes)}")
print(f"Segments : {len(network.segments)}")

# ----------------------------------------------------------------------------
# PHASE 2: space colonization over full ellipsoid for BOTH trees
#      (meter-scale constraints so SC can actually grow)
# ----------------------------------------------------------------------------

rng = np.random.default_rng(123)

# Uniform samples inside the ellipsoid (full volume)
tissue_all = domain.sample_points(n_points=10000, seed=123)

print("\n=== Tissue points for SC ===")
print(f"Total points (full ellipsoid): {len(tissue_all)}")

# Each tree gets its own copy of the same tissue cloud
tissue_for_arterial = tissue_all.copy()
tissue_for_venous   = tissue_all.copy()

# Constraints for SC in *meters* (step_size ≈ 0.2 m)
sc_constraints = BranchingConstraints(
    min_radius          = 0.10,   # don't taper below 0.10 m
    max_radius          = 1.00,   # root is 1 m
    max_branch_order    = 12,
    min_segment_length  = 0.05,   # 5 cm
    max_segment_length  = 0.50,   # 50 cm  (>= step_size)
    max_branch_angle_deg= 120.0,
    curvature_limit_deg = 45.0,
)

# GROW: arterial tree (from inlet, across whole ellipsoid)
params_art = SpaceColonizationParams(
    influence_radius = 1.2,   # 1.2 m
    kill_radius      = 0.4,   # 0.4 m
    step_size        = 0.20,  # 0.20 m
    min_radius       = 0.10,  # must be >= sc_constraints.min_radius
    taper_factor     = 0.95,
    vessel_type      = "arterial",
    max_steps        = 1000,

    preferred_direction = (1.0, 0.0, 0.0),
    directional_bias    = 0.25,
    max_deviation_deg   = 110.0,
    smoothing_weight    = 0.2,

    encourage_bifurcation=True,
    max_children_per_node=3,
    min_attractions_for_bifurcation=1,
    bifurcation_angle_threshold_deg=30.0,
    bifurcation_probability=1.0,
    grow_from_terminals_only=True,
)

art_result = space_colonization_step(
    network,
    tissue_points=tissue_for_arterial,
    params=params_art,
    constraints=sc_constraints,   # 🔴 critical: use meter-scale constraints
    seed=67,
)
print(art_result)

print("\n=== Arterial colonization ===")
print("Status      :", art_result.status)
print("Message     :", art_result.message)
print("Nodes grown :", art_result.metadata.get("nodes_grown"))
print("Steps taken :", art_result.metadata.get("steps_taken"))

print(f"Network after arterial SC: nodes={len(network.nodes)}, segments={len(network.segments)}")

# GROW: venous tree (from outlet, across whole ellipsoid)
params_ven = SpaceColonizationParams(
    influence_radius = 1.2,
    kill_radius      = 0.4,
    step_size        = 0.20,
    min_radius       = 0.10,
    taper_factor     = 0.95,
    vessel_type      = "venous",
    max_steps        = 1000,

    preferred_direction = (-1.0, 0.0, 0.0),
    directional_bias    = 0.25,
    max_deviation_deg   = 110.0,
    smoothing_weight    = 0.2,

    encourage_bifurcation=True,
    max_children_per_node=3,
    min_attractions_for_bifurcation=1,
    bifurcation_angle_threshold_deg=30.0,
    bifurcation_probability=1.0,
    grow_from_terminals_only=True,
)

ven_result = space_colonization_step(
    network,
    tissue_points=tissue_for_venous,
    params=params_ven,
    constraints=sc_constraints,   # 🔴 same meter-scale constraints
    seed=456,
)

print("\n=== Venous colonization ===")
print("Status      :", ven_result.status)
print("Message     :", ven_result.message)
print("Nodes grown :", ven_result.metadata.get("nodes_grown"))
print("Steps taken :", ven_result.metadata.get("steps_taken"))

print("\n=== Final dual-tree network ===")
print(f"Nodes    : {len(network.nodes)}")
print(f"Segments : {len(network.segments)}")


# ----------------------------------------------------------------------------
# EXPORT: STL mesh for the lumen
# ----------------------------------------------------------------------------

output_dir = REPO_ROOT / "output"
output_dir.mkdir(exist_ok=True)

stl_path = output_dir / "dual_tree_ellipsoid_trunk_SC.stl"
mesh_result = export_stl(
    network,
    output_path=str(stl_path),
    mode="robust",   # "robust" uses boolean unions; slower but cleaner
    repair=True,     # try meshfix repair if not watertight
)

print("\n=== Mesh export ===")
print("Status :", mesh_result.status)
print("Message:", mesh_result.message)
print("STL    :", mesh_result.metadata.get("output_path"))

# ----------------------------------------------------------------------------
# VALIDATION: geometric pipeline on the STL
# ----------------------------------------------------------------------------

report_path = output_dir / "dual_tree_ellipsoid_trunk_SC_report.json"

VOXEL_PITCH = 1.0e-4      # 0.1 mm
SMOOTH_ITERS = 20
DILATION_ITERS = 1

print("\n=== Validation + repair (voxel remesh) ===")
print(f"Using voxel_pitch = {VOXEL_PITCH} m")

try:
    report, G_centerline = validate_and_repair_geometry(
        input_path=str(stl_path),
        report_path=str(report_path),
        voxel_pitch=VOXEL_PITCH,
        smooth_iters=SMOOTH_ITERS,
        dilation_iters=DILATION_ITERS,
    )

    print("\n=== Validation summary (after repair) ===")
    print(f"Watertight          : {report.after_repair.watertight}")
    print(f"Volume [mm^3]       : {report.after_repair.volume}")
    print(f"Components          : {report.after_repair.num_components}")
    print(f"Flags/status        : {report.flags.status} | {report.flags.flags}")
    print(f"Centerline segments : {report.centerline_summary.get('num_segments')}")
    print(f"CFD inlet flow [m^3/s]  : {report.poiseuille_summary.get('total_inlet_flow')}")
    print(f"CFD outlet flow [m^3/s] : {report.poiseuille_summary.get('total_outlet_flow')}")
    print(f"CFD flow balance error  : {report.poiseuille_summary.get('flow_balance_error_fraction')}")

except RuntimeError as e:
    print("⚠️ Voxel remesh failed with RuntimeError:")
    print(e)
    print("Skipping voxel-based remesh/repair for now; you can still use the raw STL.")

print("\n=== Outputs ===")
print(f"STL mesh      : {stl_path}")
print(f"JSON report   : {report_path}")
mesh = trimesh.load_mesh(stl_path, process=False)
print(mesh)

# Show in notebook (or fall back if needed)
scene = trimesh.Scene(mesh)
try:
        scene.show(jupyter=True)
except Exception as e:
        print("⚠️ Inline viewer failed, trying generic viewer...")
        print(f"Error: {e}")
        scene.show()


# In[7]:


# -------------------------------------------------------------
# DOMAIN: rectangular box (20 x 60 x 30 mm)
# -------------------------------------------------------------
Lx = 0.020  # 20 mm (x)
Ly = 0.060  # 60 mm (y)
Lz = 0.030  # 30 mm (z)

margin = 0.001  # keep geometry slightly inside the box walls

domain = BoxDomain(
    x_min=0.0,
    x_max=Lx,
    y_min=0.0,
    y_max=Ly,
    z_min=0.0,
    z_max=Lz,
)

network = create_network(
    domain,
    metadata={"name": "box_inlet_outlet_3leg_backbone_parallel", "units": "m"},
    seed=42,
)

# -------------------------------------------------------------
# INLET & OUTLET: same face (x = margin, same z)
# -------------------------------------------------------------

root_radius_m = 0.001  # 2 mm diameter => 1 mm radius

# Choose inlet location on the x = margin face
y_inlet = 0.20 * Ly
z_inlet = 0.30 * Lz
inlet_pos  = (margin, y_inlet, z_inlet)

# Choose outlet on SAME face, SAME z, different y
y_outlet = 0.80 * Ly
outlet_pos = (margin, y_outlet, z_inlet)  # 👈 same x, same z as inlet

inlet_result = add_inlet(
    network,
    position=inlet_pos,
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),  # initial direction: into the box (+x)
    radius=root_radius_m,
    vessel_type="arterial",
)
inlet_id = inlet_result.new_ids["node"]

print("=== Initial network (box) ===")
print(f"Inlet node id : {inlet_id}")
print(f"Nodes         : {len(network.nodes)}")
print(f"Segments      : {len(network.segments)}")

# -------------------------------------------------------------
# Constraints for growth
# -------------------------------------------------------------

constraints = BranchingConstraints(
    min_radius=0.0003,
    max_radius=0.0030,
    max_branch_order=6,
    min_segment_length=0.001,
    max_segment_length=0.05,
    max_branch_angle_deg=80.0,
    curvature_limit_deg=15.0,
)

# -------------------------------------------------------------
# Helper: grow a straight segment from a node to an exact target point
# -------------------------------------------------------------

def grow_to_point(network, from_node_id: int, target_xyz: np.ndarray, label: str) -> int:
    """Grow a single straight segment from a node to a specific 3D point."""
    node = network.nodes[from_node_id]
    p0 = node.position.to_array()
    v = target_xyz - p0
    length = float(np.linalg.norm(v))
    if length <= 0.0:
        print(f"[{label}] Target coincides with node {from_node_id}; skipping.")
        return from_node_id

    direction = v / length

    result = grow_branch(
        network,
        from_node_id=from_node_id,
        length=length,
        direction=Direction3D(
            dx=float(direction[0]),
            dy=float(direction[1]),
            dz=float(direction[2]),
        ),
        target_radius=None,
        constraints=constraints,
        check_collisions=True,
    )
    if not result.is_success():
        print(f"[{label}] grow_to_point failed from node {from_node_id}: {result.message}")
        return from_node_id

    new_node_id = result.new_ids["node"]
    return new_node_id

# -------------------------------------------------------------
# BACKBONE: build 3-leg path inlet → outlet
# -------------------------------------------------------------
#
# Let:
#   p_inlet  = (xi, yi, zi)
#   p_outlet = (xi, yo, zi)   # same x, same z
#
# Leg 1: p_inlet → p1:  +x, no change in y,z
#   p1 = (xi + d, yi, zi)
#
# Leg 2: p1 → p2:  ±y toward yo, no change in x,z
#   p2 = (xi + d, yo, zi)
#
# Leg 3: p2 → p_outlet: pure -x, SAME |Δx| = d
#   p_outlet = (xi, yo, zi)
#   so leg3 direction is (-1, 0, 0), parallel to leg 1.

p_inlet = np.array(inlet_pos, dtype=float)
p_outlet = np.array(outlet_pos, dtype=float)

xi, yi, zi = p_inlet
xo, yo, zo = p_outlet

# Sanity checks
assert abs(xi - xo) < 1e-9, "Inlet and outlet must share same x."
assert abs(zi - zo) < 1e-9, "Inlet and outlet must share same z."

# Choose leg1 depth d, staying inside box
max_depth = (Lx - 2 * margin) * 0.8  # 80% of interior in x
d = max_depth
x1 = xi + d
if x1 > (Lx - margin):
    x1 = Lx - margin
    d = x1 - xi

# Leg 1 end: same y,z, deeper x
p1 = np.array([x1, yi, zi], dtype=float)

# Leg 2 end: same x,z, y = yo
p2 = np.array([x1, yo, zi], dtype=float)

# Leg 3 end: exactly outlet (back in x to xi, y already yo, z already zi)
p3 = p_outlet.copy()

node_id = inlet_id
node_id = grow_to_point(network, node_id, p1, label="leg1_inward_x")
node_id = grow_to_point(network, node_id, p2, label="leg2_along_y")
node_id = grow_to_point(network, node_id, p3, label="leg3_back_x")

outlet_node_id = node_id

# Mark final node as an outlet
outlet_node = network.nodes[outlet_node_id]
outlet_node.node_type = "outlet"
outlet_node.vessel_type = "arterial"

print("\n=== After 3-leg backbone growth ===")
print(f"Outlet node id: {outlet_node_id}")
print(f"Nodes          : {len(network.nodes)}")
print(f"Segments       : {len(network.segments)}")
print(f"Inlet pos      : {network.nodes[inlet_id].position.to_array()}")
print(f"Outlet pos     : {network.nodes[outlet_node_id].position.to_array()}")

# Sanity: leg1 and leg3 are parallel and same |Δx|
print("\n=== Geometry checks ===")
print("Leg1 Δx:", p1[0] - p_inlet[0])
print("Leg3 Δx:", p3[0] - p2[0])

# -------------------------------------------------------------
# EXPORT: STL of this backbone network
# -------------------------------------------------------------

output_dir = REPO_ROOT / "output"
output_dir.mkdir(exist_ok=True)

stl_path = output_dir / "box_inlet_outlet_3leg_backbone_parallel.stl"
mesh_result = export_stl(
    network,
    output_path=str(stl_path),
    mode="robust",
    repair=True,
)

print("\n=== Mesh export ===")
print("Status :", mesh_result.status)
print("Message:", mesh_result.message)
print("STL    :", mesh_result.metadata.get("output_path"))

# -------------------------------------------------------------
# VALIDATION: geometric pipeline on the STL
# -------------------------------------------------------------

report_path = output_dir / "box_inlet_outlet_3leg_backbone_parallel_report.json"

VOXEL_PITCH = 1.0e-4      # 0.1 mm
SMOOTH_ITERS = 20
DILATION_ITERS = 1

print("\n=== Validation + repair (voxel remesh) ===")
print(f"Using voxel_pitch = {VOXEL_PITCH} m")

try:
    report, G_centerline = validate_and_repair_geometry(
        input_path=str(stl_path),
        report_path=str(report_path),
        voxel_pitch=VOXEL_PITCH,
        smooth_iters=SMOOTH_ITERS,
        dilation_iters=DILATION_ITERS,
    )

    print("\n=== Validation summary (after repair) ===")
    print(f"Watertight          : {report.after_repair.watertight}")
    print(f"Volume [mm^3]       : {report.after_repair.volume}")
    print(f"Components          : {report.after_repair.num_components}")
    print(f"Flags/status        : {report.flags.status} | {report.flags.flags}")
    print(f"Centerline segments : {report.centerline_summary.get('num_segments')}")
    print(f"CFD inlet flow [m^3/s]  : {report.poiseuille_summary.get('total_inlet_flow')}")
    print(f"CFD outlet flow [m^3/s] : {report.poiseuille_summary.get('total_outlet_flow')}")
    print(f"CFD flow balance error  : {report.poiseuille_summary.get('flow_balance_error_fraction')}")

except RuntimeError as e:
    print("⚠️ Voxel remesh failed with RuntimeError:")
    print(e)
    print("Skipping voxel-based remesh/repair for now; you can still use the raw STL.")

print("\n=== Validation outputs ===")
print(f"STL mesh    : {stl_path}")
print(f"JSON report : {report_path}")

mesh = trimesh.load_mesh(stl_path, process=False)
print(mesh)

# Show in notebook (or fall back if needed)
scene = trimesh.Scene(mesh)
try:
        scene.show(jupyter=True)
except Exception as e:
        print("⚠️ Inline viewer failed, trying generic viewer...")
        print(f"Error: {e}")
        scene.show()
# -------------------------------------------------------------
# EMBEDDING: embed tree as negative space in THIS box
# -------------------------------------------------------------

print("\n=== Embedding tree as negative space in box domain ===")

embed_result = embed_tree_as_negative_space(
    tree_stl_path=stl_path,
    domain=domain,          # BoxDomain: 0.02 x 0.06 x 0.03
    voxel_pitch=3.0e-4,      # 1 mm voxels; lower if you want more detail
    margin=-0.001,
    dilation_voxels=0,
    smoothing_iters=5,
    output_void=True,
    output_shell=False,
    stl_units = 'm',
    geometry_units = 'm',
)

print(embed_result)
domain_with_void = embed_result.get("domain_with_void")
void_mesh = embed_result.get("void")

embedded_domain_path = output_dir / "box_inlet_outlet_3leg_backbone_parallel_embedded_domain.stl"
void_path = output_dir / "box_inlet_outlet_3leg_backbone_parallel_void.stl"

if domain_with_void is not None:
    domain_with_void.export(str(embedded_domain_path))
    print(f"Domain-with-void STL: {embedded_domain_path}")

if void_mesh is not None:
    void_mesh.export(str(void_path))
    print(f"Void STL            : {void_path}")

# After running your embedding code:
print(f"Domain volume: {domain_with_void.volume}")
print(f"Void volume: {void_mesh.volume}")
# Domain volume should be noticeably less than a solid box
# and the difference should roughly equal the void volume

mesh = trimesh.load_mesh(embedded_domain_path, process=False)
print(mesh)

# Show in notebook (or fall back if needed)
scene = trimesh.Scene(mesh)
try:
        scene.show(jupyter=True)
except Exception as e:
        print("⚠️ Inline viewer failed, trying generic viewer...")
        print(f"Error: {e}")
        scene.show()
print("\n=== Done ===")


# In[8]:


import sys
import subprocess
import pathlib
import numpy as np

# --- deps: trimesh + matplotlib + ipywidgets (for slider) ---
try:
    import trimesh
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh[all]"])
    import trimesh

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    from ipywidgets import interact, IntSlider
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("ipywidgets not installed – viewer will show a single slice only.")


def slice_viewer_embedded_domain(
    stl_path,
    voxel_pitch=1.0e-3,   # 1 mm voxels (in meters)
    axis="z",             # 'x', 'y', or 'z'
):
    """
    Voxelize an STL and view it slice-by-slice in Jupyter.

    Parameters
    ----------
    stl_path : str or Path
        Path to the embedded-domain STL (box with void).
    voxel_pitch : float
        Voxel size in meters. Smaller = more detail, more memory.
    axis : {'x','y','z'}
        Axis along which to slice.
    """
    stl_path = pathlib.Path(stl_path).resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    print(f"Loading STL: {stl_path}")
    mesh = trimesh.load_mesh(str(stl_path), process=False)
    print(mesh)
    print(f"Voxelizing with pitch = {voxel_pitch} m ...")

    # voxelize (coarse to avoid huge memory)
    vg = mesh.voxelized(pitch=voxel_pitch)
    vol = vg.matrix  # boolean 3D array

    print("Voxel grid shape (z, y, x):", vol.shape)

    # Reorder axes so that we always slice along the last dimension
    if axis == "z":
        data = vol  # (nz, ny, nx)
        axis_label = "z"
    elif axis == "y":
        data = np.swapaxes(vol, 0, 1)  # (ny, nz, nx)
        axis_label = "y"
    elif axis == "x":
        data = np.swapaxes(vol, 0, 2)  # (nx, ny, nz)
        axis_label = "x"
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    n_slices = data.shape[0]
    print(f"Number of slices along {axis_label}: {n_slices}")

    def show_slice(k=0):
        k = int(np.clip(k, 0, n_slices - 1))
        slice_2d = data[k, :, :]  # (ny, nx)
        plt.figure(figsize=(4, 4))
        plt.title(f"Slice {k} along {axis_label}")
        plt.imshow(slice_2d, origin="lower", cmap="gray")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    if HAS_WIDGETS:
        interact(
            show_slice,
            k=IntSlider(min=0, max=n_slices - 1, step=1, value=n_slices // 2),
        )
    else:
        # Fallback: just show middle slice
        show_slice(n_slices // 2)


# --- example call ---
# Adjust the path if needed:
embedded_domain_path = pathlib.Path("output") / "box_inlet_outlet_3leg_backbone_parallel_embedded_domain.stl"

slice_viewer_embedded_domain(
    embedded_domain_path,
    voxel_pitch=3.0e-4,  # try 1e-3 first; you can go down to 3e-4 if memory allows
    axis="z",            # try 'z', then 'y'/'x'
)


# In[11]:


# -------------------------------------------------------------
# DOMAIN: rectangular box (20 x 60 x 30 mm)
# -------------------------------------------------------------
Lx = 0.020  # 20 mm
Ly = 0.060  # 60 mm
Lz = 0.030  # 30 mm

margin = 0.001  # small inset from walls

domain = BoxDomain(
    x_min=0.0,
    x_max=Lx,
    y_min=0.0,
    y_max=Ly,
    z_min=0.0,
    z_max=Lz,
)

network = create_network(
    domain,
    metadata={"name": "box_bifurcation_merge_x_constant_branches", "units": "m"},
    seed=42,
)

# -------------------------------------------------------------
# INLET & OUTLET: same x, same z, different y
# -------------------------------------------------------------
root_radius_m = 0.001  # 2 mm diameter

y_inlet = 0.20 * Ly
y_outlet = 0.80 * Ly
z_const = 0.50 * Lz

inlet_pos  = (margin, y_inlet,  z_const)
outlet_pos = (margin, y_outlet, z_const)

inlet_result = add_inlet(
    network,
    position=inlet_pos,
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),  # inward along +x
    radius=root_radius_m,
    vessel_type="arterial",
)
inlet_id = inlet_result.new_ids["node"]

print("=== Initial network ===")
print(f"Inlet node id : {inlet_id}")
print(f"Nodes         : {len(network.nodes)}")
print(f"Segments      : {len(network.segments)}")

# -------------------------------------------------------------
# Constraints for deterministic growth
# -------------------------------------------------------------
constraints = BranchingConstraints(
    min_radius=0.0003,
    max_radius=0.0030,
    max_branch_order=6,
    min_segment_length=0.001,
    max_segment_length=0.05,
    max_branch_angle_deg=80.0,
    curvature_limit_deg=15.0,
)

# -------------------------------------------------------------
# Helper: grow straight to a target point
# -------------------------------------------------------------
def grow_to_point(network, from_node_id: int, target_xyz: np.ndarray, label: str) -> int:
    """Grow a single straight segment from a node to a specific 3D point."""
    node = network.nodes[from_node_id]
    p0 = node.position.to_array()
    v = target_xyz - p0
    length = float(np.linalg.norm(v))
    if length <= 0.0:
        print(f"[{label}] Target coincides with node {from_node_id}; skipping.")
        return from_node_id

    direction = v / length
    result = grow_branch(
        network,
        from_node_id=from_node_id,
        length=length,
        direction=Direction3D(
            dx=float(direction[0]),
            dy=float(direction[1]),
            dz=float(direction[2]),
        ),
        target_radius=None,
        constraints=constraints,
        check_collisions=True,
    )
    if not result.is_success():
        print(f"[{label}] grow_to_point failed from node {from_node_id}: {result.message}")
        return from_node_id

    new_node_id = result.new_ids["node"]
    return new_node_id

# -------------------------------------------------------------
# GEOMETRY for the path
# -------------------------------------------------------------
p_inlet  = np.array(inlet_pos,  dtype=float)
p_outlet = np.array(outlet_pos, dtype=float)

xi, yi, zi = p_inlet
xo, yo, zo = p_outlet

# Same x, same z condition
assert abs(xi - xo) < 1e-9, "Inlet and outlet must share same x."
assert abs(zi - zo) < 1e-9, "Inlet and outlet must share same z."

delta_y = yo - yi

# Depth for Leg 1
max_depth = (Lx - 2 * margin) * 0.4
d = max_depth
x_mid = xi + d
if x_mid > (Lx - margin):
    x_mid = Lx - margin
    d = x_mid - xi

# y positions for bifurcation and merge
y_bif   = yi + 0.25 * delta_y
y_merge = yi + 0.75 * delta_y

# z offsets for the two lateral branches (keep within box)
dz_branch = min(0.25 * (Lz - 2 * margin), 0.003)  # ~3 mm or less

# Key points
p_leg1_end = np.array([x_mid, yi,      zi], dtype=float)  # end of Leg 1
p_bif_node = np.array([x_mid, y_bif,   zi], dtype=float)  # bifurcation point

# Merge "center" (same x_mid, same z_const)
p_merge_c  = np.array([x_mid, y_merge, zi], dtype=float)

# Branch 1: B → mid (offset in z, moving in y) → merge center
p_b1_mid = np.array([x_mid, 0.5 * (y_bif + y_merge), zi + dz_branch], dtype=float)
p_b1_end = p_merge_c.copy()

# Branch 2: B → mid (offset opposite in z) → merge center
p_b2_mid = np.array([x_mid, 0.5 * (y_bif + y_merge), zi - dz_branch], dtype=float)
p_b2_end = p_merge_c.copy()

# NEW: after merge, continue in y/z plane until y,z match outlet,
#      keeping x = x_mid. Since zo == zi, this is just y → yo.
p_post_merge = np.array([x_mid, yo, zo], dtype=float)

# Final leg: from (x_mid, yo, zo) back to outlet (xi, yo, zo) along -x
# (pure x direction, parallel to Leg 1)
# p_outlet already defined above.

# -------------------------------------------------------------
# BUILD THE STRUCTURE
# -------------------------------------------------------------
node_id = inlet_id

# Leg 1: inlet → inward (+x)
node_id = grow_to_point(network, node_id, p_leg1_end, label="leg1_inward_x")

# Leg 2: turn 90° → along +y up to bifurcation
node_id = grow_to_point(network, node_id, p_bif_node, label="leg2_to_bif_y")

bif_node_id = node_id

# Bifurcation: two branches from bif_node_id, SAME x, different z
b1_node = grow_to_point(network, bif_node_id, p_b1_mid, label="branch1_mid")
b1_node = grow_to_point(network, b1_node,    p_b1_end, label="branch1_to_merge")

b2_node = grow_to_point(network, bif_node_id, p_b2_mid, label="branch2_mid")
b2_node = grow_to_point(network, b2_node,    p_b2_end, label="branch2_to_merge")

# Use branch1 endpoint as the merge-centerline node (at x_mid, y_merge, z_const)
merge_node_id = b1_node

# From merge center: continue in y/z plane until y,z match outlet
node_id = merge_node_id
node_id = grow_to_point(network, node_id, p_post_merge, label="post_merge_yz_to_outlet_yz")

# Finally: one vertical leg in -x back to the outlet
node_id = grow_to_point(network, node_id, p_outlet, label="to_outlet_x")

outlet_node_id = node_id
outlet_node = network.nodes[outlet_node_id]
outlet_node.node_type = "outlet"
outlet_node.vessel_type = "arterial"

print("\n=== Final structure ===")
print(f"Outlet node id: {outlet_node_id}")
print(f"Nodes          : {len(network.nodes)}")
print(f"Segments       : {len(network.segments)}")
print(f"Inlet pos      : {network.nodes[inlet_id].position.to_array()}")
print(f"Outlet pos     : {network.nodes[outlet_node_id].position.to_array()}")

# -------------------------------------------------------------
# EXPORT: STL of this branch structure
# -------------------------------------------------------------
output_dir = REPO_ROOT / "output"
output_dir.mkdir(exist_ok=True)

stl_path = output_dir / "box_bifurcation_merge_x_constant_branches.stl"
mesh_result = export_stl(
    network,
    output_path=str(stl_path),
    mode="robust",
    repair=True,
)

print("\n=== Mesh export ===")
print("Status :", mesh_result.status)
print("Message:", mesh_result.message)
print("STL    :", mesh_result.metadata.get("output_path"))

# -------------------------------------------------------------
# VALIDATION (optional)
# -------------------------------------------------------------
report_path = output_dir / "box_bifurcation_merge_x_constant_branches_report.json"

VOXEL_PITCH = 1.0e-4
SMOOTH_ITERS = 20
DILATION_ITERS = 1

print("\n=== Validation + repair (voxel remesh) ===")
print(f"Using voxel_pitch = {VOXEL_PITCH} m")

try:
    report, G_centerline = validate_and_repair_geometry(
        input_path=str(stl_path),
        report_path=str(report_path),
        voxel_pitch=VOXEL_PITCH,
        smooth_iters=SMOOTH_ITERS,
        dilation_iters=DILATION_ITERS,
    )

    print("\n=== Validation summary (after repair) ===")
    print(f"Watertight          : {report.after_repair.watertight}")
    print(f"Volume [mm^3]       : {report.after_repair.volume}")
    print(f"Components          : {report.after_repair.num_components}")
    print(f"Flags/status        : {report.flags.status} | {report.flags.flags}")
    print(f"Centerline segments : {report.centerline_summary.get('num_segments')}")
    print(f"CFD inlet flow [m^3/s]  : {report.poiseuille_summary.get('total_inlet_flow')}")
    print(f"CFD outlet flow [m^3/s] : {report.poiseuille_summary.get('total_outlet_flow')}")
    print(f"CFD flow balance error  : {report.poiseuille_summary.get('flow_balance_error_fraction')}")

except RuntimeError as e:
    print("⚠️ Voxel remesh failed with RuntimeError:")
    print(e)
    print("Skipping voxel-based remesh/repair; raw STL is still usable.")

print("\n=== Outputs ===")
print(f"STL mesh    : {stl_path}")
print(f"JSON report : {report_path}")
mesh = trimesh.load_mesh(stl_path, process=False)
print(mesh)

# Show in notebook (or fall back if needed)
scene = trimesh.Scene(mesh)
try:
        scene.show(jupyter=True)
except Exception as e:
        print("⚠️ Inline viewer failed, trying generic viewer...")
        print(f"Error: {e}")
        scene.show()


# -------------------------------------------------------------
# EMBEDDING: embed tree as negative space in THIS box
# -------------------------------------------------------------

print("\n=== Embedding tree as negative space in box domain ===")

embed_result = embed_tree_as_negative_space(
    tree_stl_path=stl_path,
    domain=domain,          # BoxDomain: 0.02 x 0.06 x 0.03
    voxel_pitch=3.0e-4,      # 1 mm voxels; lower if you want more detail
    margin=-0.001,
    dilation_voxels=0,
    smoothing_iters=5,
    output_void=True,
    output_shell=False,
    stl_units = 'm',
    geometry_units = 'm',
)

domain_with_void = embed_result.get("domain_with_void")
void_mesh = embed_result.get("void")

embedded_domain_path = output_dir / "box_bifurcation_merge_x_constant_branches_embedded_domain.stl"
void_path = output_dir / "box_bifurcation_merge_x_constant_branches_void.stl"

if domain_with_void is not None:
    domain_with_void.export(str(embedded_domain_path))
    print(f"Domain-with-void STL: {embedded_domain_path}")

if void_mesh is not None:
    void_mesh.export(str(void_path))
    print(f"Void STL            : {void_path}")

mesh = trimesh.load_mesh(embedded_domain_path, process=False)
print(mesh)


# In[12]:


import sys
import subprocess
import pathlib
import numpy as np

# --- deps: trimesh + matplotlib + ipywidgets (for slider) ---
try:
    import trimesh
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh[all]"])
    import trimesh

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    from ipywidgets import interact, IntSlider
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("ipywidgets not installed – viewer will show a single slice only.")


def slice_viewer_embedded_domain(
    stl_path,
    voxel_pitch=1.0e-3,   # 1 mm voxels (in meters)
    axis="z",             # 'x', 'y', or 'z'
):
    """
    Voxelize an STL and view it slice-by-slice in Jupyter.

    Parameters
    ----------
    stl_path : str or Path
        Path to the embedded-domain STL (box with void).
    voxel_pitch : float
        Voxel size in meters. Smaller = more detail, more memory.
    axis : {'x','y','z'}
        Axis along which to slice.
    """
    stl_path = pathlib.Path(stl_path).resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    print(f"Loading STL: {stl_path}")
    mesh = trimesh.load_mesh(str(stl_path), process=False)
    print(mesh)
    print(f"Voxelizing with pitch = {voxel_pitch} m ...")

    # voxelize (coarse to avoid huge memory)
    vg = mesh.voxelized(pitch=voxel_pitch)
    vol = vg.matrix  # boolean 3D array

    print("Voxel grid shape (z, y, x):", vol.shape)

    # Reorder axes so that we always slice along the last dimension
    if axis == "z":
        data = vol  # (nz, ny, nx)
        axis_label = "z"
    elif axis == "y":
        data = np.swapaxes(vol, 0, 1)  # (ny, nz, nx)
        axis_label = "y"
    elif axis == "x":
        data = np.swapaxes(vol, 0, 2)  # (nx, ny, nz)
        axis_label = "x"
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    n_slices = data.shape[0]
    print(f"Number of slices along {axis_label}: {n_slices}")

    def show_slice(k=0):
        k = int(np.clip(k, 0, n_slices - 1))
        slice_2d = data[k, :, :]  # (ny, nx)
        plt.figure(figsize=(4, 4))
        plt.title(f"Slice {k} along {axis_label}")
        plt.imshow(slice_2d, origin="lower", cmap="gray")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    if HAS_WIDGETS:
        interact(
            show_slice,
            k=IntSlider(min=0, max=n_slices - 1, step=1, value=n_slices // 2),
        )
    else:
        # Fallback: just show middle slice
        show_slice(n_slices // 2)


# --- example call ---
# Adjust the path if needed:
embedded_domain_path = pathlib.Path("output") / "box_bifurcation_merge_x_constant_branches_embedded_domain.stl"

slice_viewer_embedded_domain(
    embedded_domain_path,
    voxel_pitch=3.0e-4,  # try 1e-3 first; you can go down to 3e-4 if memory allows
    axis="z",            # try 'z', then 'y'/'x'
)


# In[26]:


import numpy as np
import inspect

from vascular_lib.core.types import Direction3D  # (not strictly needed here)
from vascular_lib.ops.growth import grow_to_point as _lib_grow_to_point



def _call_lib_grow_to_point(
    network,
    from_node_id: int,
    target_xyz,   # can be np array, tuple, list
    constraints,
    check_collisions: bool = True,
    target_radius=None,
    seed=None,
):
    """
    Call vascular_lib.ops.growth.grow_to_point, converting target to a tuple so
    BoxDomain.contains works (expects Point3D-like with .x/.y/.z, and grow_to_point
    converts tuples to Point3D internally).
    """
    arr = np.asarray(target_xyz, dtype=float).reshape(3,)
    target_tuple = (float(arr[0]), float(arr[1]), float(arr[2]))

    fn = _lib_grow_to_point
    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs = {}

    if "network" in params:
        kwargs["network"] = network

    for name in ("from_node_id", "start_node_id", "node_id"):
        if name in params:
            kwargs[name] = from_node_id
            break

    # In YOUR repo it's called target_point (per traceback)
    if "target_point" in params:
        kwargs["target_point"] = target_tuple
    else:
        # fallback names
        for name in ("target_xyz", "target", "point", "position"):
            if name in params:
                kwargs[name] = target_tuple
                break

    if "constraints" in params:
        kwargs["constraints"] = constraints

    if "check_collisions" in params:
        kwargs["check_collisions"] = check_collisions

    if "target_radius" in params:
        kwargs["target_radius"] = target_radius

    if seed is not None and "seed" in params:
        kwargs["seed"] = seed

    result = fn(**kwargs)

    if not result.is_success():
        return result, from_node_id

    new_node_id = result.new_ids.get("node", None)
    if new_node_id is None:
        new_nodes = result.new_ids.get("nodes", [])
        new_node_id = new_nodes[-1] if new_nodes else from_node_id

    return result, new_node_id





def _grow_to_point_segmented(
    network,
    from_node_id: int,
    target_xyz: np.ndarray,
    constraints,
    check_collisions: bool = True,
    target_radius=None,
    max_step_fraction: float = 0.90,
):
    """
    Grow from current node to target, but if the straight-line distance exceeds
    constraints.max_segment_length, break it into multiple grow_to_point calls.
    """
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3,)

    p0 = network.nodes[from_node_id].position.to_array()
    v = target_xyz - p0
    dist = float(np.linalg.norm(v))

    if dist <= 1e-12:
        return from_node_id

    max_len = float(getattr(constraints, "max_segment_length", dist))
    step_len = max_len * max_step_fraction

    # If one hop is OK, do it directly.
    if dist <= max_len + 1e-12:
        _res, nid = _call_lib_grow_to_point(
            network, from_node_id, target_xyz, constraints,
            check_collisions=check_collisions,
            target_radius=target_radius,
        )
        return nid

    # Otherwise subdivide into K steps
    direction = v / dist
    K = int(np.ceil(dist / step_len))
    nid = from_node_id
    for k in range(1, K + 1):
        pk = p0 + direction * (dist * (k / K))
        _res, nid_new = _call_lib_grow_to_point(
            network, nid, pk, constraints,
            check_collisions=check_collisions,
            target_radius=target_radius,
        )
        nid = nid_new
    return nid



def build_planar_braid(
    network,
    start_node_id: int,
    x_mid: float,
    y_start: float,
    y_end: float,
    z0: float,
    dz_branch: float,
    levels: int,
    constraints,
    dz_decay: float = 0.5,
    check_collisions: bool = True,
    # --- taper controls ---
    root_radius: float = 0.001,     # 1 mm default
    taper_factor: float = 0.9,      # ~0.707 is Murray-ish
    min_radius: float = 0.000015,    # 150 µm floor
):
    """
    Build a planar (y–z) braid at fixed x=x_mid, progressing from y_start→y_end.

    levels meaning:
      - 0: 1 lane straight (1)
      - 1: 1 → 2 → 1
      - 2: 1 → 2 → 4 → 2 → 1
      - etc.

    Tapering:
      - split stage s uses radius_at_level(s)
      - merge stage m uses radius_at_level(m-1) (so it never thickens on merge)

    Returns: node_id at the final single-lane merge point (x_mid, y_end, z0).
    """
    if levels < 0:
        raise ValueError("levels must be >= 0")

    def radius_at_level(level: int) -> float:
        r = float(root_radius) * (float(taper_factor) ** int(level))
        return max(r, float(min_radius))

    # Always start on the centerline at y_start
    active = [(start_node_id, 0.0)]  # list of (node_id, z_offset)
    y0 = float(y_start)
    y1 = float(y_end)

    # If no braid, straight shot to the end
    if levels == 0:
        target = np.array([x_mid, y1, z0], dtype=float)
        nid = _grow_to_point_segmented(
            network, start_node_id, target, constraints,
            check_collisions=check_collisions,
            target_radius=radius_at_level(0),
        )
        return nid

    # We need 2*levels stages: levels splits + levels merges
    stations = np.linspace(y0, y1, 2 * levels + 1)

    # -------------------------
    # SPLIT ladder: 1 → 2^levels
    # -------------------------
    for s in range(1, levels + 1):
        y_s = float(stations[s])
        dz_s = float(dz_branch * (dz_decay ** (s - 1)))
        r_s = radius_at_level(s)

        new_active = []
        for (nid, zoff) in active:
            for sign in (+1.0, -1.0):
                z_child = zoff + sign * dz_s
                target = np.array([x_mid, y_s, z0 + z_child], dtype=float)
                nid_child = _grow_to_point_segmented(
                    network, nid, target, constraints,
                    check_collisions=check_collisions,
                    target_radius=r_s,
                )
                new_active.append((nid_child, z_child))

        new_active.sort(key=lambda t: t[1])
        active = new_active

    # -------------------------
    # MERGE ladder: 2^levels → 1
    # -------------------------
    stage_index = levels + 1
    for m in range(levels, 0, -1):
        y_m = float(stations[stage_index])
        stage_index += 1

        # after this merge, you're effectively at "level m-1"
        r_m = radius_at_level(m - 1)

        active.sort(key=lambda t: t[1])
        if len(active) % 2 != 0:
            raise RuntimeError("Active lane count is not even during merge; bug in logic.")

        merged = []
        for i in range(0, len(active), 2):
            (nid1, z1) = active[i]
            (nid2, z2) = active[i + 1]
            z_merge = 0.5 * (z1 + z2)

            target = np.array([x_mid, y_m, z0 + z_merge], dtype=float)

            # IMPORTANT: keep your original behavior: grow BOTH into merge target
            nid1m = _grow_to_point_segmented(
                network, nid1, target, constraints,
                check_collisions=check_collisions,
                target_radius=r_m,
            )
            nid2m = _grow_to_point_segmented(
                network, nid2, target, constraints,
                check_collisions=check_collisions,
                target_radius=r_m,
            )

            # keep one representative (structure is still complete because nid2m segment exists)
            merged.append((nid1m, z_merge))

        merged.sort(key=lambda t: t[1])
        active = merged

    if len(active) != 1:
        raise RuntimeError(f"Expected 1 active lane at end, got {len(active)}")

    nid_final, _ = active[0]
    target_final = np.array([x_mid, y1, z0], dtype=float)

    # final snap to exact end point, keep level-0 radius
    nid_final = _grow_to_point_segmented(
        network, nid_final, target_final, constraints,
        check_collisions=check_collisions,
        target_radius=radius_at_level(0),
    )
    return nid_final



# In[29]:


# -------------------------------------------------------------
# DOMAIN: rectangular box (20 x 60 x 30 mm)
# -------------------------------------------------------------
Lx = 0.020  # 20 mm
Ly = 0.060  # 60 mm
Lz = 0.030  # 30 mm

margin = 0.001  # small inset from walls

domain = BoxDomain(
    x_min=0.0,
    x_max=Lx,
    y_min=0.0,
    y_max=Ly,
    z_min=0.0,
    z_max=Lz,
)

network = create_network(
    domain,
    metadata={"name": "box_bifurcation_merge_4_constant_branches", "units": "m"},
    seed=42,
)

# -------------------------------------------------------------
# INLET & OUTLET: same x, same z, different y
# -------------------------------------------------------------
root_radius_m = 0.001  # 2 mm diameter

y_inlet = 0.20 * Ly
y_outlet = 0.80 * Ly
z_const = 0.50 * Lz

inlet_pos  = (margin, y_inlet,  z_const)
outlet_pos = (margin, y_outlet, z_const)

inlet_result = add_inlet(
    network,
    position=inlet_pos,
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),  # inward along +x
    radius=root_radius_m,
    vessel_type="arterial",
)
inlet_id = inlet_result.new_ids["node"]

print("=== Initial network ===")
print(f"Inlet node id : {inlet_id}")
print(f"Nodes         : {len(network.nodes)}")
print(f"Segments      : {len(network.segments)}")

# -------------------------------------------------------------
# Constraints for deterministic growth
# -------------------------------------------------------------
constraints = BranchingConstraints(
    min_radius=0.0003,
    max_radius=0.0030,
    max_branch_order=6,
    min_segment_length=0.001,
    max_segment_length=0.05,
    max_branch_angle_deg=80.0,
    curvature_limit_deg=15.0,
)

# --- TAPER SETTINGS (meters) ---
ROOT_RADIUS = root_radius_m     # start from your inlet radius (1 mm)
TAPER_FACTOR = 0.7              # per bifurcation level (≈0.707 is Murray-ish)
MIN_RADIUS = 0.00015            # 150 µm floor

def radius_at_level(level: int) -> float:
    r = ROOT_RADIUS * (TAPER_FACTOR ** level)
    return max(r, MIN_RADIUS)

# -------------------------------------------------------------
# Helper: grow straight to a target point
# -------------------------------------------------------------
def grow_to_point(network, from_node_id: int, target_xyz: np.ndarray, label: str) -> int:
    """Grow a single straight segment from a node to a specific 3D point."""
    node = network.nodes[from_node_id]
    p0 = node.position.to_array()
    v = target_xyz - p0
    length = float(np.linalg.norm(v))
    if length <= 0.0:
        print(f"[{label}] Target coincides with node {from_node_id}; skipping.")
        return from_node_id

    direction = v / length
    result = grow_branch(
        network,
        from_node_id=from_node_id,
        length=length,
        direction=Direction3D(
            dx=float(direction[0]),
            dy=float(direction[1]),
            dz=float(direction[2]),
        ),
        target_radius=None,
        constraints=constraints,
        check_collisions=True,
    )
    if not result.is_success():
        print(f"[{label}] grow_to_point failed from node {from_node_id}: {result.message}")
        return from_node_id

    new_node_id = result.new_ids["node"]
    return new_node_id

# -------------------------------------------------------------
# GEOMETRY for the path
# -------------------------------------------------------------
p_inlet  = np.array(inlet_pos,  dtype=float)
p_outlet = np.array(outlet_pos, dtype=float)

xi, yi, zi = p_inlet
xo, yo, zo = p_outlet

# Same x, same z condition
assert abs(xi - xo) < 1e-9, "Inlet and outlet must share same x."
assert abs(zi - zo) < 1e-9, "Inlet and outlet must share same z."

delta_y = yo - yi

# Depth for Leg 1
max_depth = (Lx - 2 * margin) * 0.4
d = max_depth
x_mid = xi + d
if x_mid > (Lx - margin):
    x_mid = Lx - margin
    d = x_mid - xi

# y positions for bifurcation and merge
y_bif   = yi + 0.25 * delta_y
y_merge = yi + 0.75 * delta_y

# z offsets for the two lateral branches (keep within box)
dz_branch = min(0.25 * (Lz - 2 * margin), 0.003)  # ~3 mm or less

# Key points
p_leg1_end = np.array([x_mid, yi,      zi], dtype=float)  # end of Leg 1
p_bif_node = np.array([x_mid, y_bif,   zi], dtype=float)  # bifurcation point

# Merge "center" (same x_mid, same z_const)
p_merge_c  = np.array([x_mid, y_merge, zi], dtype=float)

# Branch 1: B → mid (offset in z, moving in y) → merge center
p_b1_mid = np.array([x_mid, 0.5 * (y_bif + y_merge), zi + dz_branch], dtype=float)
p_b1_end = p_merge_c.copy()

# Branch 2: B → mid (offset opposite in z) → merge center
p_b2_mid = np.array([x_mid, 0.5 * (y_bif + y_merge), zi - dz_branch], dtype=float)
p_b2_end = p_merge_c.copy()

# NEW: after merge, continue in y/z plane until y,z match outlet,
#      keeping x = x_mid. Since zo == zi, this is just y → yo.
p_post_merge = np.array([x_mid, yo, zo], dtype=float)

# Final leg: from (x_mid, yo, zo) back to outlet (xi, yo, zo) along -x
# (pure x direction, parallel to Leg 1)
# p_outlet already defined above.

# -------------------------------------------------------------
# BUILD THE STRUCTURE
# -------------------------------------------------------------

levels = 10 # ✅ your meaning: 0=1, 1=1→2→1, 2=1→2→4→2→1, ...

node_id = inlet_id

# Leg 1: inlet → inward (+x)
node_id = _grow_to_point_segmented(
    network, node_id, p_leg1_end, constraints,
    check_collisions=True,
    target_radius=ROOT_RADIUS
)

# Leg 2: turn 90° → along +y up to bifurcation station (centerline at x_mid)
node_id = _grow_to_point_segmented(
    network, node_id, p_bif_node, constraints, check_collisions=True
)
bif_node_id = node_id

# Planar braid between y_bif and y_merge at constant x_mid, centered at z=zi
merge_node_id = build_planar_braid(
    network=network,
    start_node_id=bif_node_id,
    x_mid=float(x_mid),
    y_start=float(y_bif),
    y_end=float(y_merge),
    z0=float(zi),
    dz_branch=float(dz_branch),
    levels=int(levels),
    constraints=constraints,
)

# Continue in y/z plane until y,z match outlet (in your setup zo == zi, so just y to yo)
node_id = merge_node_id
node_id = _grow_to_point_segmented(
    network, node_id, p_post_merge, constraints, check_collisions=True
)

# Finally: back along -x to the outlet point
node_id = _grow_to_point_segmented(
    network, node_id, p_outlet, constraints, check_collisions=True
)

outlet_node_id = node_id
outlet_node = network.nodes[outlet_node_id]
outlet_node.node_type = "outlet"
outlet_node.vessel_type = "arterial"

print("\n=== Final structure ===")
print(f"Outlet node id: {outlet_node_id}")
print(f"Nodes          : {len(network.nodes)}")
print(f"Segments       : {len(network.segments)}")
print(f"Inlet pos      : {network.nodes[inlet_id].position.to_array()}")
print(f"Outlet pos     : {network.nodes[outlet_node_id].position.to_array()}")


# -------------------------------------------------------------
# EXPORT: STL of this branch structure
# -------------------------------------------------------------
output_dir = REPO_ROOT / "output"
output_dir.mkdir(exist_ok=True)

stl_path = output_dir / "box_bifurcation_merge_x_constant_branches.stl"
mesh_result = export_stl(
    network,
    output_path=str(stl_path),
    mode="robust",
    repair=True,
)

print("\n=== Mesh export ===")
print("Status :", mesh_result.status)
print("Message:", mesh_result.message)
print("STL    :", mesh_result.metadata.get("output_path"))

# -------------------------------------------------------------
# VALIDATION (optional)
# -------------------------------------------------------------
report_path = output_dir / "box_bifurcation_merge_x_constant_branches_report.json"

VOXEL_PITCH = 1.0e-6
SMOOTH_ITERS = 20
DILATION_ITERS = 1

print("\n=== Validation + repair (voxel remesh) ===")
print(f"Using voxel_pitch = {VOXEL_PITCH} m")

try:
    report, G_centerline = validate_and_repair_geometry(
        input_path=str(stl_path),
        report_path=str(report_path),
        voxel_pitch=VOXEL_PITCH,
        smooth_iters=SMOOTH_ITERS,
        dilation_iters=DILATION_ITERS,
    )

    print("\n=== Validation summary (after repair) ===")
    print(f"Watertight          : {report.after_repair.watertight}")
    print(f"Volume [mm^3]       : {report.after_repair.volume}")
    print(f"Components          : {report.after_repair.num_components}")
    print(f"Flags/status        : {report.flags.status} | {report.flags.flags}")
    print(f"Centerline segments : {report.centerline_summary.get('num_segments')}")
    print(f"CFD inlet flow [m^3/s]  : {report.poiseuille_summary.get('total_inlet_flow')}")
    print(f"CFD outlet flow [m^3/s] : {report.poiseuille_summary.get('total_outlet_flow')}")
    print(f"CFD flow balance error  : {report.poiseuille_summary.get('flow_balance_error_fraction')}")

except RuntimeError as e:
    print("⚠️ Voxel remesh failed with RuntimeError:")
    print(e)
    print("Skipping voxel-based remesh/repair; raw STL is still usable.")

print("\n=== Outputs ===")
print(f"STL mesh    : {stl_path}")
print(f"JSON report : {report_path}")
mesh = trimesh.load_mesh(stl_path, process=False)
print(mesh)

# Show in notebook (or fall back if needed)
scene = trimesh.Scene(mesh)
try:
        scene.show(jupyter=True)
except Exception as e:
        print("⚠️ Inline viewer failed, trying generic viewer...")
        print(f"Error: {e}")
        scene.show()


# -------------------------------------------------------------
# EMBEDDING: embed tree as negative space in THIS box
# -------------------------------------------------------------

print("\n=== Embedding tree as negative space in box domain ===")

embed_result = embed_tree_as_negative_space(
    tree_stl_path=stl_path,
    domain=domain,          # BoxDomain: 0.02 x 0.06 x 0.03
    voxel_pitch=3.0e-6,      # 1 mm voxels; lower if you want more detail
    margin=-0.005,
    dilation_voxels=0,
    smoothing_iters=5,
    output_void=True,
    output_shell=False,
    stl_units = 'm',
    geometry_units = 'm',
)

domain_with_void = embed_result.get("domain_with_void")
void_mesh = embed_result.get("void")

embedded_domain_path = output_dir / "box_bifurcation_merge_16_constant_branches_embedded_domain.stl"
void_path = output_dir / "box_bifurcation_merge_x_constant_branches_void.stl"

if domain_with_void is not None:
    domain_with_void.export(str(embedded_domain_path))
    print(f"Domain-with-void STL: {embedded_domain_path}")

if void_mesh is not None:
    void_mesh.export(str(void_path))
    print(f"Void STL            : {void_path}")

mesh = trimesh.load_mesh(embedded_domain_path, process=False)
print(mesh)


# In[28]:


import sys
import subprocess
import pathlib
import numpy as np

# --- deps: trimesh + matplotlib + ipywidgets (for slider) ---
try:
    import trimesh
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh[all]"])
    import trimesh

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    from ipywidgets import interact, IntSlider
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("ipywidgets not installed – viewer will show a single slice only.")


def slice_viewer_embedded_domain(
    stl_path,
    voxel_pitch=1.0e-3,   # 1 mm voxels (in meters)
    axis="z",             # 'x', 'y', or 'z'
):
    """
    Voxelize an STL and view it slice-by-slice in Jupyter.

    Parameters
    ----------
    stl_path : str or Path
        Path to the embedded-domain STL (box with void).
    voxel_pitch : float
        Voxel size in meters. Smaller = more detail, more memory.
    axis : {'x','y','z'}
        Axis along which to slice.
    """
    stl_path = pathlib.Path(stl_path).resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    print(f"Loading STL: {stl_path}")
    mesh = trimesh.load_mesh(str(stl_path), process=False)
    print(mesh)
    print(f"Voxelizing with pitch = {voxel_pitch} m ...")

    # voxelize (coarse to avoid huge memory)
    vg = mesh.voxelized(pitch=voxel_pitch)
    vol = vg.matrix  # boolean 3D array

    print("Voxel grid shape (z, y, x):", vol.shape)

    # Reorder axes so that we always slice along the last dimension
    if axis == "z":
        data = vol  # (nz, ny, nx)
        axis_label = "z"
    elif axis == "y":
        data = np.swapaxes(vol, 0, 1)  # (ny, nz, nx)
        axis_label = "y"
    elif axis == "x":
        data = np.swapaxes(vol, 0, 2)  # (nx, ny, nz)
        axis_label = "x"
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    n_slices = data.shape[0]
    print(f"Number of slices along {axis_label}: {n_slices}")

    def show_slice(k=0):
        k = int(np.clip(k, 0, n_slices - 1))
        slice_2d = data[k, :, :]  # (ny, nx)
        plt.figure(figsize=(4, 4))
        plt.title(f"Slice {k} along {axis_label}")
        plt.imshow(slice_2d, origin="lower", cmap="gray")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    if HAS_WIDGETS:
        interact(
            show_slice,
            k=IntSlider(min=0, max=n_slices - 1, step=1, value=n_slices // 2),
        )
    else:
        # Fallback: just show middle slice
        show_slice(n_slices // 2)


# --- example call ---
# Adjust the path if needed:
embedded_domain_path = pathlib.Path("output") / "box_bifurcation_merge_8_constant_branches_embedded_domain.stl"

slice_viewer_embedded_domain(
    embedded_domain_path,
    voxel_pitch=1.0e-5,  # try 1e-3 first; you can go down to 3e-4 if memory allows
    axis="x",            # try 'z', then 'y'/'x'
)


# In[ ]:




