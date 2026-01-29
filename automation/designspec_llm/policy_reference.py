"""
Policy Reference for DesignSpec LLM Agent

This module provides a comprehensive text reference of all policies and their parameters
that can be included in the LLM context window. The LLM uses this to understand what
parameters are available for each policy type.

USAGE:
    from automation.designspec_llm.policy_reference import get_policy_reference_text
    policy_docs = get_policy_reference_text()
"""

POLICY_REFERENCE = '''
# DesignSpec Policy Reference

This document describes all available policies and their parameters. All geometric values are in METERS unless otherwise noted.

---

## growth - Network Generation Policy

Controls the generation backend and its parameters for growing vascular networks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable/disable network generation |
| backend | string | "space_colonization" | Generation algorithm: "space_colonization", "scaffold_topdown", or "programmatic" |
| target_terminals | int | null | Target number of terminal nodes |
| terminal_tolerance | float | 0.1 | Acceptable deviation from target (fraction, 0.1 = 10%) |
| max_iterations | int | 500 | Maximum growth iterations |
| seed | int | null | Random seed for reproducibility |
| min_segment_length | float | 0.0002 | Minimum segment length in meters (0.2mm) |
| max_segment_length | float | 0.002 | Maximum segment length in meters (2mm) |
| min_radius | float | 0.0001 | Minimum vessel radius in meters (0.1mm) |
| step_size | float | 0.0003 | Growth step size in meters (0.3mm) |
| backend_params | object | {} | Backend-specific configuration (see below) |

### backend_params for scaffold_topdown

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| primary_axis | [x,y,z] | [0,0,-1] | Global growth direction as unit vector |
| splits | int | 3 | Number of child branches at each bifurcation |
| levels | int | 6 | Maximum depth of branching (NOT "depth"!) |
| ratio | float | 0.78 | Radius taper factor at each level |
| step_length | float | 0.002 | Initial step length in meters |
| step_decay | float | 0.92 | Factor to multiply step length at each level |
| spread | float | 0.0015 | Lateral spread distance in meters |
| spread_decay | float | 0.90 | Factor to multiply spread at each level |
| cone_angle_deg | float | 70 | Maximum cone angle for child branches (degrees) |
| jitter_deg | float | 12 | Random jitter in branch angles (degrees) |
| curvature | float | 0.35 | Curvature factor (0=straight, 1=max curve) |
| curve_samples | int | 7 | Number of sample points for curved branches |
| wall_margin_m | float | 0.0001 | Minimum distance from domain boundary (meters) |
| boundary_extra_m | float | 0.0 | Extra boundary clearance (meters) |
| min_radius | float | 0.00005 | Minimum vessel radius (meters) |
| depth_adaptive_mode | string | "none" | Step length computation: "none", "uniform", "geometric" |
| branch_plane_mode | string | "global" | Branch orientation: "global" (2D), "local" (3D), "hybrid" |
| collision_online | object | {} | Online collision avoidance config (see below) |
| collision_postpass | object | {} | Post-generation collision cleanup config |

### collision_online (nested in scaffold_topdown backend_params)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable online collision avoidance |
| buffer_abs_m | float | 0.00002 | Absolute buffer distance (meters) |
| buffer_rel | float | 0.05 | Relative buffer (fraction of radius) |
| cell_size_m | float | 0.0005 | Spatial index cell size (meters) |
| rotation_attempts | int | 14 | Number of rotation attempts to avoid collision |
| reduction_factors | [float] | [0.6, 0.35] | Radius reduction factors to try |
| max_attempts_per_child | int | 18 | Max attempts per child branch |
| on_fail | string | "terminate_branch" | Action on failure: "terminate_branch" |
| merge_on_collision | bool | false | Connect to nearby branch instead of terminating |
| merge_distance_m | float | 0.0002 | Max distance to merge target (meters) |
| fail_retry_rounds | int | 0 | Number of retry rounds (0 = no retry) |
| fail_retry_mode | string | "both" | Retry mode: "shrink_radius", "increase_step", "both", "none" |
| fail_retry_shrink_factor | float | 0.85 | Radius shrink factor per retry |
| fail_retry_step_boost | float | 1.2 | Step length boost factor per retry |

### collision_postpass (nested in scaffold_topdown backend_params)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable post-generation collision cleanup |
| min_clearance_m | float | 0.00002 | Minimum clearance between segments (meters) |
| strategy_order | [string] | ["shrink", "terminate"] | Resolution strategies in order |
| shrink_factor | float | 0.9 | Radius shrink factor |
| shrink_max_iterations | int | 6 | Max shrink iterations |

### backend_params for space_colonization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| influence_radius | float | 0.001 | Attractor influence radius (meters) |
| kill_radius | float | 0.0003 | Attractor kill radius (meters) |
| perception_angle | float | 90 | Perception angle (degrees) |
| num_attraction_points | int | 5000 | Number of attractor points |

---

## tissue_sampling - Attractor Point Distribution

Controls how tissue/attractor points are distributed for space colonization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable tissue sampling |
| n_points | int | 1000 | Number of attractor points |
| seed | int | null | Random seed |
| strategy | string | "uniform" | Distribution: "uniform", "depth_biased", "radial_biased", "boundary_shell", "gaussian", "mixture" |
| depth_reference | object | {"mode": "face", "face": "top"} | Reference for depth calculation |
| depth_distribution | string | "power" | Depth distribution: "linear", "power", "exponential", "beta" |
| depth_min | float | 0.0 | Minimum depth (meters) |
| depth_max | float | null | Maximum depth (null = full domain) |
| depth_power | float | 2.0 | Power for depth-biased distribution |
| radial_distribution | string | "center_heavy" | Radial distribution: "center_heavy", "edge_heavy", "ring" |
| r_min | float | 0.0 | Minimum radius (meters) |
| r_max | float | null | Maximum radius (null = domain radius) |
| ring_r0 | float | 0.0 | Ring center radius (meters) |
| ring_sigma | float | 0.001 | Ring width (meters) |
| shell_thickness | float | 0.002 | Boundary shell thickness (meters) |
| min_distance_to_ports | float | 0.0005 | Minimum distance from ports (meters) |

---

## collision - Collision Detection Policy

Controls collision detection during network generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable collision detection |
| check_collisions | bool | true | Perform collision checks |
| collision_clearance | float | 0.0002 | Minimum clearance between segments (meters) |

---

## unified_collision - Advanced Collision Policy

Comprehensive collision detection and resolution policy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable collision handling |
| min_clearance | float | 0.0002 | Minimum clearance (meters) |
| strategy_order | [string] | ["reroute", "shrink", "terminate"] | Resolution strategies |
| min_radius | float | 0.0001 | Floor for shrink strategy (meters) |
| check_segment_segment | bool | true | Check segment-segment collisions |
| check_segment_mesh | bool | true | Check segment-mesh collisions |
| check_segment_boundary | bool | true | Check segment-boundary collisions |
| reroute_max_attempts | int | 3 | Max reroute attempts |
| shrink_factor | float | 0.9 | Radius shrink factor |
| shrink_max_iterations | int | 5 | Max shrink iterations |
| inflate_by_radius | bool | true | Inflate obstacles by vessel radius |

---

## network_cleanup - Network Post-Processing

Controls node snapping, duplicate merging, and segment pruning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enable_snap | bool | true | Enable node snapping |
| snap_tol | float | 0.0001 | Snap tolerance (meters) |
| enable_prune | bool | true | Enable segment pruning |
| min_segment_length | float | 0.0001 | Minimum segment length (meters) |
| enable_merge | bool | true | Enable duplicate merging |
| merge_tol | float | 0.0001 | Merge tolerance (meters) |

---

## mesh_synthesis - Mesh Generation from Networks

Controls how vascular networks are converted to triangle meshes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| add_node_spheres | bool | true | Add spheres at bifurcation nodes |
| cap_ends | bool | true | Cap terminal ends |
| radius_clamp_min | float | null | Minimum radius clamp (meters) |
| radius_clamp_max | float | null | Maximum radius clamp (meters) |
| voxel_repair_synthesis | bool | false | Use voxel-based repair |
| voxel_repair_pitch | float | 0.0001 | Repair voxel pitch (meters) |
| voxel_repair_auto_adjust | bool | true | Auto-adjust pitch for budget |
| voxel_repair_max_voxels | int | 100000000 | Max voxels for repair |
| segments_per_circle | int | 16 | Mesh resolution around tubes |

---

## mesh_merge - Mesh Merging Operations

Controls how multiple meshes are combined.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mode | string | "auto" | Merge mode: "auto", "voxel", "boolean" |
| voxel_pitch | float | 0.00005 | Voxel pitch for merge (meters) |
| auto_adjust_pitch | bool | true | Auto-adjust pitch for budget |
| max_pitch_steps | int | 4 | Max pitch relaxation steps |
| pitch_step_factor | float | 1.5 | Pitch increase factor per step |
| fallback_boolean | bool | true | Fall back to boolean if voxel fails |
| keep_largest_component | bool | true | Keep only largest component |
| min_component_faces | int | 100 | Minimum faces to keep component |
| fill_voxels | bool | true | Fill interior voxels |
| max_voxels | int | 100000000 | Max voxels budget |
| min_voxels_per_diameter | int | 4 | Min voxels across smallest channel |
| detail_loss_threshold | float | 0.5 | Warn if volume loss exceeds this |

---

## embedding - Embedding Voids into Domains

Controls the voxelization and carving process for creating domain-with-void meshes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| voxel_pitch | float | 0.0003 | Voxel pitch (meters) |
| shell_thickness | float | 0.002 | Shell thickness (meters) |
| auto_adjust_pitch | bool | true | Auto-adjust pitch for budget |
| max_pitch_steps | int | 4 | Max pitch relaxation steps |
| pitch_step_factor | float | 1.5 | Pitch increase factor |
| max_voxels | int | 100000000 | Max voxels budget |
| preserve_ports_enabled | bool | true | Preserve port openings |
| preserve_mode | string | "recarve" | Port preservation mode |
| carve_radius_factor | float | 1.2 | Port carving radius multiplier |
| carve_depth | float | 0.002 | Port carving depth (meters) |
| output_shell | bool | false | Output shell mesh |
| output_domain_with_void | bool | true | Output domain with void |
| output_void_mesh | bool | true | Output void mesh |

---

## validity - Validation Policy

Controls which validation checks are enabled and their thresholds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| check_watertight | bool | true | Check mesh is watertight |
| check_components | bool | true | Check component count |
| check_min_diameter | bool | true | Check minimum diameter |
| check_open_ports | bool | false | Check port connectivity |
| check_bounds | bool | true | Check bounds |
| check_void_inside_domain | bool | true | Check void is inside domain |
| allow_boundary_intersections_at_ports | bool | false | Allow surface openings at ports |
| surface_opening_tolerance | float | 0.001 | Port neighborhood tolerance (meters) |
| min_diameter_threshold | float | 0.0005 | Minimum diameter threshold (meters) |
| max_components | int | 1 | Maximum allowed components |

---

## open_ports - Open Port Validation

Controls how ports are checked for connectivity to the outside.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable open port validation |
| probe_radius_factor | float | 1.2 | Probe radius multiplier |
| probe_length | float | 0.002 | Probe length (meters) |
| min_connected_volume_voxels | int | 10 | Min connected voxels |
| validation_pitch | float | null | Validation pitch (null = adaptive) |
| local_region_size | float | 0.004 | Local ROI size (meters) |
| max_voxels_roi | int | 2000000 | Max voxels per port ROI |
| min_voxels_across_radius | int | 8 | Min voxels across port radius |
| adaptive_pitch | bool | true | Auto-compute pitch from port radius |

---

## resolution - Scale-Aware Resolution Policy

Single source of truth for all scale-dependent tolerances and pitches.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| min_channel_diameter | float | 0.00002 | Smallest channel diameter (meters, 20um) |
| min_voxels_across_feature | int | 8 | Min voxels across smallest feature |
| max_voxels | int | 100000000 | Global voxel budget |
| min_pitch | float | 0.000001 | Minimum allowed pitch (meters, 1um) |
| max_pitch | float | 0.001 | Maximum allowed pitch (meters, 1mm) |
| auto_relax_pitch | bool | true | Auto-relax pitch for budget |
| pitch_step_factor | float | 1.5 | Pitch relaxation factor |
| embed_pitch_factor | float | 1.0 | Embedding pitch multiplier |
| merge_pitch_factor | float | 1.0 | Merge pitch multiplier |
| repair_pitch_factor | float | 1.0 | Repair pitch multiplier |

---

## channels - Channel Primitive Policy

Controls the geometry of individual channel segments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable channel creation |
| profile | string | "cylinder" | Profile: "cylinder", "taper", "fang_hook" |
| length_mode | string | "explicit" | Length mode: "explicit", "to_center_fraction", "to_depth" |
| length | float | null | Channel length (meters, required for explicit/to_depth) |
| length_fraction | float | 0.5 | Fraction toward center (for to_center_fraction) |
| taper_factor | float | 0.5 | Taper factor for tapered channels |
| hook_depth | float | 0.002 | Hook depth for fang_hook (meters) |
| hook_strength | float | 0.5 | Hook curve strength |
| hook_angle_deg | float | 90 | Hook angle (degrees) |
| segments_per_curve | int | 16 | Mesh segments per curve |

---

## ports - Port Placement Policy

Controls how inlet/outlet ports are positioned on domain surfaces.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | true | Enable port placement |
| face | string | "top" | Target face: "top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z" |
| pattern | string | "circle" | Pattern: "circle", "grid", "center_rings", "explicit" |
| ridge_width | float | 0.0001 | Ridge width (meters) |
| ridge_clearance | float | 0.0001 | Ridge clearance (meters) |
| port_margin | float | 0.0005 | Port margin (meters) |
| placement_fraction | float | 0.7 | Placement radius fraction |

---

## radius - Radius Policy

Controls how radii are computed at bifurcations and along paths.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mode | string | "murray" | Mode: "constant", "taper", "murray" |
| murray_exponent | float | 3.0 | Murray's Law exponent |
| taper_factor | float | 0.8 | Taper factor per bifurcation |
| min_radius | float | 0.0001 | Minimum radius (meters) |
| max_radius | float | 0.005 | Maximum radius (meters) |

---

## output - Output Policy

Controls output file generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| output_dir | string | "./output" | Output directory |
| output_units | string | "mm" | Output units: "mm" or "m" |
| naming_convention | string | "default" | Naming: "default" or "timestamped" |
| save_intermediates | bool | false | Save intermediate files |
| save_reports | bool | true | Save reports |
| output_stl | bool | true | Output STL files |
| output_json | bool | true | Output JSON files |

---

## repair - Mesh Repair Policy

Controls mesh repair operations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| voxel_repair_enabled | bool | true | Enable voxel repair |
| voxel_pitch | float | 0.0001 | Repair voxel pitch (meters) |
| auto_adjust_pitch | bool | true | Auto-adjust pitch |
| fill_voxels | bool | true | Fill interior voxels |
| remove_small_components_enabled | bool | true | Remove small components |
| min_component_faces | int | 500 | Min faces to keep |
| fill_holes_enabled | bool | true | Fill holes |
| smooth_enabled | bool | false | Enable smoothing |
| smooth_iterations | int | 10 | Smoothing iterations |

'''


def get_policy_reference_text() -> str:
    """
    Get the complete policy reference text for inclusion in LLM context.
    
    Returns
    -------
    str
        The policy reference documentation as a formatted string.
    """
    return POLICY_REFERENCE


def get_policy_reference_compact() -> str:
    """
    Get a compact version of the policy reference (key policies only).
    
    Returns
    -------
    str
        A shorter policy reference focusing on the most commonly used policies.
    """
    compact = '''
# Key DesignSpec Policies (Compact Reference)

## growth - Network Generation
- backend: "space_colonization" | "scaffold_topdown" | "programmatic"
- target_terminals: int (target terminal count)
- backend_params: backend-specific config

### scaffold_topdown backend_params (IMPORTANT):
- levels: int (branching depth, NOT "depth"!)
- splits: int (children per bifurcation)
- primary_axis: [x,y,z] (growth direction)
- step_length: float (meters)
- spread: float (meters)
- cone_angle_deg: float (degrees)
- branch_plane_mode: "global" | "local" | "hybrid"
- collision_online: {enabled, merge_on_collision, merge_distance_m, ...}

## embedding - Void Embedding
- voxel_pitch: float (meters)
- preserve_ports_enabled: bool
- carve_depth: float (meters)

## validity - Validation
- check_watertight: bool
- check_void_inside_domain: bool
- min_diameter_threshold: float (meters)

## mesh_merge - Mesh Merging
- mode: "auto" | "voxel" | "boolean"
- voxel_pitch: float (meters)

## collision - Collision Detection
- enabled: bool
- collision_clearance: float (meters)
'''
    return compact


__all__ = ["get_policy_reference_text", "get_policy_reference_compact", "POLICY_REFERENCE"]
