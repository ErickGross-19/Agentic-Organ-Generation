#!/usr/bin/env python3
"""
Generate programmatic stress test plan for DesignSpec.

This script generates a complex set of steps for the programmatic backend stress test,
including spiral/helical trunks and binary tree branching from each of 5 inlets.

The generated steps are written to the stress_programmatic_pathfinding_dense_embed.json
spec file in examples/stress_tests/.

Usage:
    python scripts/generate_programmatic_stress_plan.py
"""

import json
import math
import os
from pathlib import Path


CYLINDER_RADIUS = 0.01
CYLINDER_HEIGHT = 0.03
CYLINDER_CENTER = [0.0, 0.0, 0.0]
WALL_MARGIN = 0.001
SAFE_RADIUS = CYLINDER_RADIUS - WALL_MARGIN

INLET_RADIUS = 0.001
INLET_Z = 0.015
INLET_POSITIONS = [
    [0.0, 0.0, INLET_Z],
    [0.006, 0.0, INLET_Z],
    [-0.006, 0.0, INLET_Z],
    [0.0, 0.006, INLET_Z],
    [0.0, -0.006, INLET_Z],
]

TAPER_FACTOR = 0.80
MIN_RADIUS = 2.5e-05
BRANCH_DEPTH = 7
TRUNK_WAYPOINTS = 10
SPIRAL_TURNS = 2.0
SPIRAL_DESCENT = 0.025


def generate_spiral_waypoints(inlet_idx: int, inlet_pos: list, num_waypoints: int) -> list:
    """Generate spiral/helical waypoints descending from an inlet."""
    waypoints = []
    
    inlet_x, inlet_y, inlet_z = inlet_pos
    
    dist_from_center = math.sqrt(inlet_x**2 + inlet_y**2)
    if dist_from_center < 0.001:
        base_angle = inlet_idx * (2 * math.pi / 5)
        spiral_radius = 0.004
    else:
        base_angle = math.atan2(inlet_y, inlet_x)
        spiral_radius = min(dist_from_center * 0.7, SAFE_RADIUS * 0.5)
    
    for i in range(num_waypoints):
        t = (i + 1) / num_waypoints
        
        angle = base_angle + SPIRAL_TURNS * 2 * math.pi * t
        
        current_radius = spiral_radius * (1.0 - 0.3 * t)
        
        z = inlet_z - SPIRAL_DESCENT * t
        z = max(z, -CYLINDER_HEIGHT/2 + WALL_MARGIN)
        
        x = current_radius * math.cos(angle)
        y = current_radius * math.sin(angle)
        
        dist = math.sqrt(x**2 + y**2)
        if dist > SAFE_RADIUS:
            scale = SAFE_RADIUS / dist
            x *= scale
            y *= scale
        
        waypoints.append({
            "name": f"I{inlet_idx}_trunk_{i}",
            "position": [round(x, 8), round(y, 8), round(z, 8)]
        })
    
    return waypoints


def generate_binary_tree_steps(
    inlet_idx: int,
    parent_name: str,
    parent_pos: list,
    parent_radius: float,
    depth: int,
    branch_idx: int,
    steps: list,
    node_counter: dict
) -> None:
    """Recursively generate binary tree branching steps."""
    if depth <= 0 or parent_radius < MIN_RADIUS:
        return
    
    child_radius = max(parent_radius * TAPER_FACTOR, MIN_RADIUS)
    
    branch_length = 0.002 * (0.85 ** (BRANCH_DEPTH - depth))
    branch_length = max(branch_length, 0.0003)
    
    parent_x, parent_y, parent_z = parent_pos
    
    base_angle = (branch_idx * 137.5) * math.pi / 180
    
    for child_idx in range(2):
        angle_offset = (math.pi / 4) if child_idx == 0 else (-math.pi / 4)
        angle = base_angle + angle_offset
        
        dx = branch_length * math.cos(angle) * 0.7
        dy = branch_length * math.sin(angle) * 0.7
        dz = -branch_length * 0.7
        
        child_x = parent_x + dx
        child_y = parent_y + dy
        child_z = parent_z + dz
        
        dist = math.sqrt(child_x**2 + child_y**2)
        if dist > SAFE_RADIUS:
            scale = SAFE_RADIUS / dist
            child_x *= scale
            child_y *= scale
        
        child_z = max(child_z, -CYLINDER_HEIGHT/2 + WALL_MARGIN)
        
        child_name = f"I{inlet_idx}_d{depth}_b{node_counter['count']}"
        node_counter['count'] += 1
        
        steps.append({
            "op": "ADD_NODE",
            "params": {
                "name": child_name,
                "position": [round(child_x, 8), round(child_y, 8), round(child_z, 8)],
                "node_type": "junction" if depth > 1 else "terminal",
                "radius": child_radius
            }
        })
        
        steps.append({
            "op": "ROUTE",
            "params": {
                "from": parent_name,
                "to": child_name,
                "algorithm": "astar_voxel",
                "radius": child_radius,
                "clearance": 5e-05
            }
        })
        
        if depth > 1 and child_radius > MIN_RADIUS:
            generate_binary_tree_steps(
                inlet_idx=inlet_idx,
                parent_name=child_name,
                parent_pos=[child_x, child_y, child_z],
                parent_radius=child_radius,
                depth=depth - 1,
                branch_idx=node_counter['count'],
                steps=steps,
                node_counter=node_counter
            )


def generate_steps_for_inlet(inlet_idx: int, inlet_pos: list) -> list:
    """Generate all steps for a single inlet: trunk + binary tree branches."""
    steps = []
    node_counter = {'count': 0}
    
    inlet_name = f"I{inlet_idx}"
    steps.append({
        "op": "ADD_NODE",
        "params": {
            "name": inlet_name,
            "position": inlet_pos,
            "node_type": "inlet",
            "radius": INLET_RADIUS
        }
    })
    
    waypoints = generate_spiral_waypoints(inlet_idx, inlet_pos, TRUNK_WAYPOINTS)
    
    current_radius = INLET_RADIUS
    prev_name = inlet_name
    
    branch_points = [2, 5, 8]
    
    for i, wp in enumerate(waypoints):
        wp_name = wp["name"]
        wp_pos = wp["position"]
        
        is_branch_point = i in branch_points
        node_type = "junction" if is_branch_point else "junction"
        
        current_radius = max(current_radius * 0.95, MIN_RADIUS * 2)
        
        steps.append({
            "op": "ADD_NODE",
            "params": {
                "name": wp_name,
                "position": wp_pos,
                "node_type": node_type,
                "radius": current_radius
            }
        })
        
        steps.append({
            "op": "ROUTE",
            "params": {
                "from": prev_name,
                "to": wp_name,
                "algorithm": "astar_voxel",
                "radius": current_radius,
                "clearance": 5e-05
            }
        })
        
        if is_branch_point:
            branch_radius = current_radius * TAPER_FACTOR
            generate_binary_tree_steps(
                inlet_idx=inlet_idx,
                parent_name=wp_name,
                parent_pos=wp_pos,
                parent_radius=branch_radius,
                depth=BRANCH_DEPTH,
                branch_idx=node_counter['count'],
                steps=steps,
                node_counter=node_counter
            )
        
        prev_name = wp_name
    
    return steps


def generate_all_steps() -> list:
    """Generate steps for all inlets."""
    all_steps = []
    
    for inlet_idx, inlet_pos in enumerate(INLET_POSITIONS):
        inlet_steps = generate_steps_for_inlet(inlet_idx, inlet_pos)
        all_steps.extend(inlet_steps)
    
    return all_steps


def create_spec_with_steps(steps: list) -> dict:
    """Create the full DesignSpec with generated steps."""
    return {
        "schema": {
            "name": "aog_designspec",
            "version": "1.0.0",
            "compatible_with": ["1.0.x"]
        },
        "meta": {
            "name": "stress_programmatic_pathfinding_dense_embed",
            "description": "Stress test: Cylinder R=10mm H=30mm with ridge. 5 inlets (center + 4 @ r=6mm). Programmatic backend with A* pathfinding, spiral trunks, binary tree branching (depth 7), and voxel_merge_fallback collision strategy. WARNING: This spec is intentionally extreme and may exceed memory/time limits.",
            "seed": 42,
            "input_units": "m",
            "tags": ["stress_test", "programmatic", "pathfinding", "astar", "dense", "voxel_merge_fallback", "embed", "extreme"]
        },
        "policies": {
            "resolution": {
                "input_units": "m",
                "min_channel_diameter": 5e-05,
                "min_voxels_across_feature": 50,
                "max_voxels": 10000000000000,
                "min_pitch": 1e-06,
                "max_pitch": 1e-06,
                "auto_relax_pitch": False,
                "pitch_step_factor": 1.0,
                "max_voxels_embed": 10000000000000,
                "max_voxels_merge": 10000000000000,
                "max_voxels_repair": 10000000000000
            },
            "domain_meshing": {
                "cache_meshes": True,
                "emit_warnings": True,
                "target_face_count": 200000,
                "voxel_pitch": None,
                "primitive_policy": {
                    "sections_radial": 128,
                    "sections_axial": 64,
                    "sections_angular": 128,
                    "subdivisions": 2
                },
                "mesh_policy": {
                    "validate_watertight": True,
                    "repair_if_needed": True,
                    "repair_voxel_pitch": 5e-05,
                    "max_faces": 1000000,
                    "simplify_if_over_max": True,
                    "simplify_target_ratio": 0.6
                }
            },
            "ridge": {
                "enabled": True,
                "face": "top",
                "height": 0.005,
                "thickness": 0.0005,
                "inset": 0.0,
                "overlap": 0.00025,
                "resolution": 128
            },
            "ports": {
                "enabled": True,
                "projection_mode": "clamp_to_face",
                "ridge_constraint_enabled": True,
                "disk_constraint_enabled": True
            },
            "growth": {
                "enabled": True,
                "backend": "programmatic",
                "target_terminals": None,
                "terminal_tolerance": 0.15,
                "max_iterations": 5000,
                "min_segment_length": 1e-05,
                "max_segment_length": 0.005,
                "min_radius": 2.5e-05,
                "step_size": 0.001,
                "backend_params": {
                    "mode": "network",
                    "path_algorithm": "astar_voxel",
                    "waypoint_policy": {
                        "allow_skip": True,
                        "max_skip_count": 999,
                        "emit_warnings": True,
                        "fallback_direct": True
                    },
                    "retry_policy": {
                        "max_retries": 20,
                        "retry_with_larger_clearance": True,
                        "clearance_increase_factor": 1.2
                    },
                    "radius_policy": {
                        "mode": "taper",
                        "taper_factor": 0.80,
                        "min_radius": 2.5e-05,
                        "max_radius": 0.001
                    },
                    "collision_policy": {
                        "enabled": True,
                        "min_clearance": 5e-05,
                        "strategy_order": ["reroute", "voxel_merge_fallback", "shrink", "terminate"],
                        "min_radius": 2.5e-05,
                        "check_segment_segment": True,
                        "check_segment_boundary": True,
                        "check_segment_mesh": False,
                        "inflate_by_radius": True
                    },
                    "steps": steps
                }
            },
            "radius": {
                "mode": "taper",
                "murray_exponent": 3.0,
                "taper_factor": 0.80,
                "min_radius": 2.5e-05,
                "max_radius": 0.001
            },
            "collision": {
                "enabled": True,
                "check_collisions": True,
                "collision_clearance": 5e-05
            },
            "composition": {
                "repair_enabled": False,
                "union_before_embed": True,
                "keep_largest_component": False,
                "merge_policy": {
                    "keep_largest_component": False
                }
            },
            "repair": {
                "voxel_repair_enabled": False,
                "fill_voxels": False
            },
            "embedding": {
                "voxel_pitch": 1e-06,
                "shell_thickness": 0.0,
                "use_resolution_policy": False,
                "auto_adjust_pitch": False,
                "max_pitch_steps": 1,
                "pitch_step_factor": 1.0,
                "max_voxels": 10000000000000,
                "fallback": "auto",
                "preserve_ports_enabled": True,
                "preserve_mode": "recarve",
                "carve_radius_factor": 1.4,
                "carve_depth": 0.002
            },
            "open_port": {
                "enabled": False
            },
            "validity": {
                "check_watertight": False,
                "check_components": False,
                "check_min_diameter": False,
                "check_open_ports": False,
                "check_bounds": True,
                "check_void_inside_domain": False
            },
            "output": {
                "output_dir": "./out/stress_programmatic_pathfinding_dense_embed",
                "output_units": "m",
                "save_intermediates": True,
                "save_reports": True,
                "output_stl": True,
                "output_json": True
            }
        },
        "domains": {
            "main_domain": {
                "type": "cylinder",
                "center": [0.0, 0.0, 0.0],
                "radius": 0.01,
                "height": 0.03
            }
        },
        "components": [
            {
                "id": "stress_programmatic",
                "domain_ref": "main_domain",
                "ports": {
                    "inlets": [
                        {
                            "name": "inlet_center",
                            "position": [0.0, 0.0, 0.015],
                            "direction": [0.0, 0.0, -1.0],
                            "radius": 0.001
                        },
                        {
                            "name": "inlet_e",
                            "position": [0.006, 0.0, 0.015],
                            "direction": [0.0, 0.0, -1.0],
                            "radius": 0.001
                        },
                        {
                            "name": "inlet_w",
                            "position": [-0.006, 0.0, 0.015],
                            "direction": [0.0, 0.0, -1.0],
                            "radius": 0.001
                        },
                        {
                            "name": "inlet_n",
                            "position": [0.0, 0.006, 0.015],
                            "direction": [0.0, 0.0, -1.0],
                            "radius": 0.001
                        },
                        {
                            "name": "inlet_s",
                            "position": [0.0, -0.006, 0.015],
                            "direction": [0.0, 0.0, -1.0],
                            "radius": 0.001
                        }
                    ],
                    "outlets": []
                },
                "build": {
                    "type": "backend_network",
                    "backend": "programmatic",
                    "backend_params": {
                        "mode": "network",
                        "path_algorithm": "astar_voxel",
                        "waypoint_policy": {
                            "allow_skip": True,
                            "max_skip_count": 999,
                            "emit_warnings": True,
                            "fallback_direct": True
                        },
                        "retry_policy": {
                            "max_retries": 20,
                            "retry_with_larger_clearance": True,
                            "clearance_increase_factor": 1.2
                        },
                        "radius_policy": {
                            "mode": "taper",
                            "taper_factor": 0.80,
                            "min_radius": 2.5e-05,
                            "max_radius": 0.001
                        },
                        "collision_policy": {
                            "enabled": True,
                            "min_clearance": 5e-05,
                            "strategy_order": ["reroute", "voxel_merge_fallback", "shrink", "terminate"],
                            "min_radius": 2.5e-05,
                            "check_segment_segment": True,
                            "check_segment_boundary": True,
                            "check_segment_mesh": False,
                            "inflate_by_radius": True
                        },
                        "steps": steps
                    }
                },
                "save_artifacts": {
                    "network": "artifacts/stress_programmatic_network.json",
                    "void_mesh": "artifacts/stress_programmatic_void.stl"
                }
            }
        ],
        "embedding": {
            "enable": True,
            "outputs": {
                "domain_with_void": "domain_with_void.stl",
                "void_mesh": "void_union.stl"
            }
        },
        "validity": {
            "enable": False,
            "save_report": "validity_report.json"
        },
        "outputs": {
            "artifacts_dir": "artifacts",
            "named": {
                "domain_mesh": "artifacts/domain_mesh_with_ridge.stl",
                "run_report": "run_report.json"
            }
        }
    }


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    output_path = repo_root / "examples" / "stress_tests" / "stress_programmatic_pathfinding_dense_embed.json"
    
    print("Generating programmatic stress test steps...")
    steps = generate_all_steps()
    print(f"Generated {len(steps)} steps")
    
    spec = create_spec_with_steps(steps)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(spec, f, indent=2)
    
    print(f"Wrote spec to: {output_path}")
    print(f"Total steps: {len(steps)}")
    
    add_node_count = sum(1 for s in steps if s['op'] == 'ADD_NODE')
    route_count = sum(1 for s in steps if s['op'] == 'ROUTE')
    print(f"  ADD_NODE steps: {add_node_count}")
    print(f"  ROUTE steps: {route_count}")


if __name__ == "__main__":
    main()
