import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize
try:
    from skimage.morphology import skeletonize_3d
    HAS_SKELETONIZE_3D = True
except ImportError:
    HAS_SKELETONIZE_3D = False
from typing import Dict, Tuple, Any


def extract_centerline_graph(
    fluid_mask: np.ndarray,
    bbox_min: np.ndarray,
    spacing: np.ndarray,
    keep_largest_component: bool = True,
):
    """
    Extract a centerline graph from a voxelized fluid domain.

    - Skeletonize the fluid.
    - Distance transform -> local radius.
    - Build a graph with one node per skeleton voxel, radius, and coord.

    Coordinates are mapped from voxel indices (i,j,k) to world coords via:
      coord = bbox_min + spacing * (i+0.5, j+0.5, k+0.5)

    Parameters
    ----------
    fluid_mask : (nx, ny, nz) bool
        Boolean array of voxel occupancy
    bbox_min : (3,) float
        Lower corner of physical domain
    spacing : (3,) float
        Voxel spacing (dx, dy, dz)

    Returns
    -------
    G : nx.Graph
        Centerline graph with node attributes: coord, radius, ijk
    centerline_meta : dict
        Metadata including bbox_min, spacing, num_nodes, num_edges
    """
    bbox_min = np.asarray(bbox_min, dtype=float)
    spacing = np.asarray(spacing, dtype=float)

    labeled, num_components = ndimage.label(fluid_mask)
    if num_components > 1:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        largest_component = np.argmax(component_sizes)
        fluid_mask_filtered = (labeled == largest_component)
    else:
        fluid_mask_filtered = fluid_mask

    skeleton = None
    if HAS_SKELETONIZE_3D:
        try:
            skeleton = skeletonize_3d(fluid_mask_filtered)
        except Exception:
            pass
    
    if skeleton is None or not skeleton.any():
        try:
            skeleton = skeletonize(fluid_mask_filtered, method='lee')
        except Exception:
            skeleton = None

    # Distance transform gives radius in voxel units -> convert to world
    dist_voxel = ndimage.distance_transform_edt(fluid_mask_filtered)
    dist_world = dist_voxel * np.mean(spacing)

    if skeleton is None or not skeleton.any():
        idx_i, idx_j, idx_k = np.array([]), np.array([]), np.array([])
        num_skel_voxels = 0
    else:
        idx_i, idx_j, idx_k = np.where(skeleton)
        num_skel_voxels = len(idx_i)
    
    if num_skel_voxels == 0:
        raise RuntimeError("Skeletonization produced zero voxels.")

    G = nx.Graph()
    ijk_to_id: Dict[Tuple[int, int, int], int] = {}

    # Nodes
    for node_id, (i, j, k) in enumerate(zip(idx_i, idx_j, idx_k)):
        ijk = (int(i), int(j), int(k))
        ijk_to_id[ijk] = node_id

        # world coordinates from bounding box + spacing (voxel centers)
        coord = bbox_min + spacing * np.array([i + 0.5, j + 0.5, k + 0.5], dtype=float)
        radius = dist_world[i, j, k]

        G.add_node(
            node_id,
            ijk=ijk,
            coord=coord,
            radius=float(radius),
        )

    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    skel_set = set(ijk_to_id.keys())
    for ijk, node_id in ijk_to_id.items():
        i, j, k = ijk
        for di, dj, dk in neighbor_offsets:
            nb = (i + di, j + dj, k + dk)
            if nb in skel_set:
                nb_id = ijk_to_id[nb]
                if not G.has_edge(node_id, nb_id):
                    G.add_edge(node_id, nb_id)

    if keep_largest_component:
        components = list(nx.connected_components(G))
        if len(components) > 1:
            largest = max(components, key=len)
            G = G.subgraph(largest).copy()

    for u, v in G.edges:
        cu = np.asarray(G.nodes[u]["coord"], dtype=float)
        cv = np.asarray(G.nodes[v]["coord"], dtype=float)
        length = float(np.linalg.norm(cu - cv))
        G.edges[u, v]["length"] = length

    centerline_meta = {
        "bbox_min": bbox_min.tolist(),
        "spacing": spacing.tolist(),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "grid_shape": list(fluid_mask.shape),
    }

    return G, centerline_meta
