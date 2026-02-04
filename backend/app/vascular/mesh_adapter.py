"""
Convert AOG VascularNetwork to manifold3d objects.

This module bridges the gap between AOG's trimesh-based output
and MorphoStruct's manifold3d-based geometry pipeline.
"""

import sys
import os

# Add AOG repo to Python path
AOG_REPO_PATH = r"C:\Users\Erick\organ-agent-generation\repo"
if AOG_REPO_PATH not in sys.path:
    sys.path.insert(0, AOG_REPO_PATH)

import manifold3d as m3d
import trimesh
import numpy as np
from typing import Tuple
import logging

from generation.core.network import VascularNetwork
from generation.adapters.mesh_adapter import to_trimesh
from aog_policies import MeshSynthesisPolicy

logger = logging.getLogger(__name__)


def network_to_manifold(
    network: VascularNetwork,
    segments_per_circle: int = 12,
    add_node_spheres: bool = False,
    cap_ends: bool = True,
) -> Tuple[m3d.Manifold, dict]:
    """
    Convert VascularNetwork to manifold3d Manifold.

    Steps:
    1. Use AOG to_trimesh() to get trimesh
    2. Convert trimesh to manifold3d
    3. Return manifold for MorphoStruct pipeline

    Parameters
    ----------
    network : VascularNetwork
        The vascular network to convert
    segments_per_circle : int
        Number of segments around vessel circumference
    add_node_spheres : bool
        Whether to add spheres at junction nodes
    cap_ends : bool
        Whether to cap terminal ends

    Returns
    -------
    manifold : m3d.Manifold
        The converted manifold geometry
    stats : dict
        Statistics about the conversion
    """
    # Create mesh synthesis policy
    policy = MeshSynthesisPolicy(
        add_node_spheres=add_node_spheres,
        cap_ends=cap_ends,
        segments_per_circle=segments_per_circle,
        mutate_network_in_place=False,
        voxel_repair_synthesis=False,  # Disable for performance
    )

    # Convert network to trimesh using AOG adapter
    try:
        mesh_result = to_trimesh(network, policy)

        # Handle both tuple and OperationResult returns
        if hasattr(mesh_result, 'mesh'):
            # It's an OperationResult
            mesh_trimesh = mesh_result.mesh
        elif isinstance(mesh_result, tuple):
            # It's a (mesh, report) tuple
            mesh_trimesh, report = mesh_result
        else:
            # Direct mesh return
            mesh_trimesh = mesh_result

    except Exception as e:
        logger.error(f"Failed to convert network to trimesh: {e}")
        raise ValueError(f"Network to trimesh conversion failed: {e}")

    if mesh_trimesh is None:
        raise ValueError("to_trimesh returned None - network may be empty")

    # Ensure mesh is watertight and has correct normals
    if not mesh_trimesh.is_watertight:
        logger.warning("Trimesh is not watertight, attempting to fix")
        try:
            mesh_trimesh.fill_holes()
            mesh_trimesh.fix_normals()
        except Exception as e:
            logger.warning(f"Could not repair mesh: {e}")

    # Convert trimesh to manifold3d
    try:
        vertices = mesh_trimesh.vertices
        faces = mesh_trimesh.faces

        # manifold3d expects flattened vertex data and triangular face indices
        manifold = m3d.Manifold(
            mesh=m3d.Mesh(
                vert_properties=vertices.flatten().tolist(),
                tri_verts=faces.flatten().tolist(),
                num_prop=3,  # x, y, z coordinates
            )
        )

        # Calculate stats
        stats = {
            "network_nodes": len(network.nodes),
            "network_segments": len(network.segments),
            "mesh_vertices": len(vertices),
            "mesh_faces": len(faces),
            "is_manifold": manifold.status() == m3d.Manifold.Error.NoError,
        }

        return manifold, stats

    except Exception as e:
        logger.error(f"Failed to convert trimesh to manifold3d: {e}")
        raise ValueError(f"Trimesh to manifold3d conversion failed: {e}")


def calculate_network_stats(network: VascularNetwork) -> dict:
    """
    Calculate statistics about a vascular network.

    Parameters
    ----------
    network : VascularNetwork
        The network to analyze

    Returns
    -------
    stats : dict
        Network statistics
    """
    total_length = 0.0
    min_radius = float('inf')
    max_radius = 0.0

    for segment in network.segments.values():
        if hasattr(segment, 'geometry') and segment.geometry is not None:
            geom = segment.geometry
            # Get length from segment
            if hasattr(segment, 'length'):
                total_length += segment.length

            # Get radius from geometry
            if hasattr(geom, 'radius_start'):
                min_radius = min(min_radius, geom.radius_start)
                max_radius = max(max_radius, geom.radius_start)
            if hasattr(geom, 'radius_end'):
                min_radius = min(min_radius, geom.radius_end)
                max_radius = max(max_radius, geom.radius_end)

    # Count terminals
    terminal_count = sum(
        1 for node in network.nodes.values()
        if hasattr(node, 'node_type') and node.node_type == 'terminal'
    )

    return {
        "total_length_m": total_length,
        "min_radius_m": min_radius if min_radius != float('inf') else 0.0,
        "max_radius_m": max_radius,
        "terminal_count": terminal_count,
    }
