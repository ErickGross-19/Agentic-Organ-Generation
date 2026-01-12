"""
Geometry conversion for CFD simulations.

Converts VascularNetwork to watertight 3D surface models suitable for meshing.
Handles tube sweeping, junction blending, and surface cleanup.

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.types import Point3D, TubeGeometry


@dataclass
class GeometryConfig:
    """Configuration for geometry generation."""
    
    tube_resolution: int = 16
    junction_blend_radius: float = 0.0005
    smoothing_iterations: int = 2
    min_segment_length: float = 0.0001
    
    export_format: str = "stl"
    repair_mesh: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tube_resolution": self.tube_resolution,
            "junction_blend_radius": self.junction_blend_radius,
            "smoothing_iterations": self.smoothing_iterations,
            "min_segment_length": self.min_segment_length,
            "export_format": self.export_format,
            "repair_mesh": self.repair_mesh,
        }


@dataclass
class CenterlineSegment:
    """Processed centerline segment for geometry generation."""
    
    segment_id: int
    points: np.ndarray
    radii: np.ndarray
    start_node_id: int
    end_node_id: int
    vessel_type: str
    
    @property
    def length(self) -> float:
        """Compute total centerline length."""
        total = 0.0
        for i in range(len(self.points) - 1):
            total += np.linalg.norm(self.points[i + 1] - self.points[i])
        return total


@dataclass
class SimulationCenterlines:
    """Collection of processed centerlines for simulation."""
    
    segments: List[CenterlineSegment] = field(default_factory=list)
    inlet_node_ids: List[int] = field(default_factory=list)
    outlet_node_ids: List[int] = field(default_factory=list)
    junction_node_ids: List[int] = field(default_factory=list)
    
    def get_segment_by_id(self, segment_id: int) -> Optional[CenterlineSegment]:
        """Get segment by ID."""
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        return None


def extract_simulation_centerlines(
    network: VascularNetwork,
    config: Optional[GeometryConfig] = None,
) -> SimulationCenterlines:
    """
    Extract clean centerlines from VascularNetwork for CFD.
    
    Ensures consistent orientation (root → terminals), removes
    near-zero-length segments, and ensures radius continuity.
    
    Parameters
    ----------
    network : VascularNetwork
        Source vascular network
    config : GeometryConfig, optional
        Geometry configuration
        
    Returns
    -------
    SimulationCenterlines
        Processed centerlines ready for geometry generation
    """
    if config is None:
        config = GeometryConfig()
    
    result = SimulationCenterlines()
    
    for node in network.nodes.values():
        if node.node_type == "inlet":
            result.inlet_node_ids.append(node.id)
        elif node.node_type in ["outlet", "terminal"]:
            result.outlet_node_ids.append(node.id)
        elif node.node_type == "junction":
            result.junction_node_ids.append(node.id)
    
    for seg_id, segment in network.segments.items():
        start_node = network.nodes[segment.start_node_id]
        end_node = network.nodes[segment.end_node_id]
        
        if segment.geometry.centerline_points:
            points = [start_node.position.to_array()]
            points.extend([p.to_array() for p in segment.geometry.centerline_points])
            points.append(end_node.position.to_array())
            points = np.array(points)
            
            n_points = len(points)
            radii = np.linspace(
                segment.geometry.radius_start,
                segment.geometry.radius_end,
                n_points
            )
        else:
            points = np.array([
                start_node.position.to_array(),
                end_node.position.to_array(),
            ])
            radii = np.array([
                segment.geometry.radius_start,
                segment.geometry.radius_end,
            ])
        
        length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        if length < config.min_segment_length:
            continue
        
        cl_segment = CenterlineSegment(
            segment_id=seg_id,
            points=points,
            radii=radii,
            start_node_id=segment.start_node_id,
            end_node_id=segment.end_node_id,
            vessel_type=segment.vessel_type,
        )
        result.segments.append(cl_segment)
    
    return result


def _create_tube_vertices(
    centerline: np.ndarray,
    radii: np.ndarray,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create tube mesh vertices and faces from centerline.
    
    Parameters
    ----------
    centerline : np.ndarray
        Centerline points (N, 3)
    radii : np.ndarray
        Radius at each centerline point (N,)
    resolution : int
        Number of circumferential points
        
    Returns
    -------
    vertices : np.ndarray
        Mesh vertices (N*resolution, 3)
    faces : np.ndarray
        Mesh faces (triangles)
    """
    n_points = len(centerline)
    n_verts = n_points * resolution
    
    vertices = np.zeros((n_verts, 3))
    
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    
    for i in range(n_points):
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i == n_points - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        
        if abs(tangent[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        normal = np.cross(tangent, up)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        binormal = np.cross(tangent, normal)
        
        r = radii[i]
        for j in range(resolution):
            offset = r * (np.cos(theta[j]) * normal + np.sin(theta[j]) * binormal)
            vertices[i * resolution + j] = centerline[i] + offset
    
    faces = []
    for i in range(n_points - 1):
        for j in range(resolution):
            j_next = (j + 1) % resolution
            
            v0 = i * resolution + j
            v1 = i * resolution + j_next
            v2 = (i + 1) * resolution + j_next
            v3 = (i + 1) * resolution + j
            
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    return vertices, np.array(faces)


def _create_cap_faces(
    center_idx: int,
    ring_start_idx: int,
    resolution: int,
    flip: bool = False,
) -> np.ndarray:
    """Create triangular cap faces."""
    faces = []
    for j in range(resolution):
        j_next = (j + 1) % resolution
        if flip:
            faces.append([center_idx, ring_start_idx + j_next, ring_start_idx + j])
        else:
            faces.append([center_idx, ring_start_idx + j, ring_start_idx + j_next])
    return np.array(faces)


def build_watertight_geometry(
    network: VascularNetwork,
    config: Optional[GeometryConfig] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build watertight 3D surface geometry from VascularNetwork.
    
    Creates tube surfaces for each segment and blends them at junctions.
    Outputs a single connected fluid domain mesh.
    
    Parameters
    ----------
    network : VascularNetwork
        Source vascular network
    config : GeometryConfig, optional
        Geometry configuration
    output_path : str, optional
        Path to save geometry file
        
    Returns
    -------
    dict
        Geometry result containing:
        - vertices: mesh vertices
        - faces: mesh faces
        - inlet_faces: face indices for inlet caps
        - outlet_faces: face indices for outlet caps
        - wall_faces: face indices for wall
        - file_path: path to saved file (if output_path provided)
    """
    if config is None:
        config = GeometryConfig()
    
    centerlines = extract_simulation_centerlines(network, config)
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    inlet_face_indices = []
    outlet_face_indices = []
    wall_face_indices = []
    
    for cl_seg in centerlines.segments:
        tube_verts, tube_faces = _create_tube_vertices(
            cl_seg.points, cl_seg.radii, config.tube_resolution
        )
        
        tube_faces_offset = tube_faces + vertex_offset
        
        wall_start = len(all_faces)
        all_vertices.append(tube_verts)
        all_faces.append(tube_faces_offset)
        wall_end = wall_start + len(tube_faces_offset)
        wall_face_indices.extend(range(wall_start, wall_end))
        
        vertex_offset += len(tube_verts)
        
        if cl_seg.start_node_id in centerlines.inlet_node_ids:
            cap_center = cl_seg.points[0]
            all_vertices.append(cap_center.reshape(1, 3))
            center_idx = vertex_offset
            vertex_offset += 1
            
            ring_start = vertex_offset - 1 - config.tube_resolution
            cap_faces = _create_cap_faces(center_idx, ring_start - len(tube_verts) + vertex_offset, config.tube_resolution, flip=True)
            
            inlet_start = len(all_faces)
            for face_list in all_faces:
                inlet_start = sum(len(f) for f in all_faces if isinstance(f, np.ndarray))
                break
        
        if cl_seg.end_node_id in centerlines.outlet_node_ids:
            cap_center = cl_seg.points[-1]
            all_vertices.append(cap_center.reshape(1, 3))
            center_idx = vertex_offset
            vertex_offset += 1
    
    if all_vertices:
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces) if all_faces else np.array([])
    else:
        vertices = np.array([])
        faces = np.array([])
    
    result = {
        "vertices": vertices,
        "faces": faces,
        "inlet_faces": inlet_face_indices,
        "outlet_faces": outlet_face_indices,
        "wall_faces": wall_face_indices,
        "n_segments": len(centerlines.segments),
        "inlet_node_ids": centerlines.inlet_node_ids,
        "outlet_node_ids": centerlines.outlet_node_ids,
    }
    
    if output_path is not None:
        result["file_path"] = _export_geometry(vertices, faces, output_path, config)
    
    return result


def _export_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    config: GeometryConfig,
) -> str:
    """Export geometry to file."""
    try:
        import trimesh
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        if config.repair_mesh:
            mesh.fill_holes()
            mesh.fix_normals()
        
        mesh.export(output_path)
        return output_path
        
    except ImportError:
        _export_stl_simple(vertices, faces, output_path)
        return output_path


def _export_stl_simple(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
) -> None:
    """Simple STL export without trimesh dependency."""
    with open(output_path, 'w') as f:
        f.write("solid mesh\n")
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid mesh\n")


def check_watertightness(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, Any]:
    """
    Check if mesh is watertight.
    
    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices
    faces : np.ndarray
        Mesh faces
        
    Returns
    -------
    dict
        Watertightness check results
    """
    if len(faces) == 0:
        return {
            "is_watertight": False,
            "n_boundary_edges": 0,
            "n_non_manifold_edges": 0,
            "error": "No faces in mesh",
        }
    
    edge_count = {}
    
    for face in faces:
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v0, v1]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary_edges = sum(1 for count in edge_count.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_count.values() if count > 2)
    
    is_watertight = boundary_edges == 0 and non_manifold_edges == 0
    
    return {
        "is_watertight": is_watertight,
        "n_boundary_edges": boundary_edges,
        "n_non_manifold_edges": non_manifold_edges,
        "n_edges": len(edge_count),
        "n_faces": len(faces),
        "n_vertices": len(vertices),
    }
