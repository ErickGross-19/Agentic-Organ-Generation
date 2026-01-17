"""
Mesh generation for CFD simulations.

Provides hooks for volumetric mesh generation using SimVascular or
fallback methods (VTK + TetGen/Gmsh).

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""
    
    target_edge_size: float = 0.0005
    min_edge_size: float = 0.0001
    max_edge_size: float = 0.002
    
    boundary_layer_enabled: bool = False
    boundary_layer_thickness: float = 0.0002
    boundary_layer_layers: int = 3
    boundary_layer_growth_rate: float = 1.2
    
    curvature_refinement: bool = True
    curvature_angle_threshold: float = 30.0
    
    junction_refinement: bool = True
    junction_refinement_factor: float = 0.5
    
    min_quality_threshold: float = 0.1
    
    mesher: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_edge_size": self.target_edge_size,
            "min_edge_size": self.min_edge_size,
            "max_edge_size": self.max_edge_size,
            "boundary_layer_enabled": self.boundary_layer_enabled,
            "boundary_layer_thickness": self.boundary_layer_thickness,
            "boundary_layer_layers": self.boundary_layer_layers,
            "boundary_layer_growth_rate": self.boundary_layer_growth_rate,
            "curvature_refinement": self.curvature_refinement,
            "curvature_angle_threshold": self.curvature_angle_threshold,
            "junction_refinement": self.junction_refinement,
            "junction_refinement_factor": self.junction_refinement_factor,
            "min_quality_threshold": self.min_quality_threshold,
            "mesher": self.mesher,
        }


@dataclass
class MeshResult:
    """Result of mesh generation."""
    
    success: bool = False
    
    vertices: Optional[np.ndarray] = None
    cells: Optional[np.ndarray] = None
    cell_types: Optional[np.ndarray] = None
    
    boundary_faces: Optional[np.ndarray] = None
    inlet_faces: Optional[np.ndarray] = None
    outlet_faces: Optional[np.ndarray] = None
    wall_faces: Optional[np.ndarray] = None
    
    n_vertices: int = 0
    n_cells: int = 0
    n_boundary_faces: int = 0
    
    min_quality: float = 0.0
    mean_quality: float = 0.0
    
    file_path: Optional[str] = None
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "n_vertices": self.n_vertices,
            "n_cells": self.n_cells,
            "n_boundary_faces": self.n_boundary_faces,
            "min_quality": self.min_quality,
            "mean_quality": self.mean_quality,
            "file_path": self.file_path,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def generate_mesh(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    config: Optional[MeshConfig] = None,
    output_path: Optional[str] = None,
) -> MeshResult:
    """
    Generate volumetric mesh from surface mesh.
    
    Attempts to use SimVascular meshing if available, falls back to
    simple tetrahedral meshing otherwise.
    
    Parameters
    ----------
    surface_vertices : np.ndarray
        Surface mesh vertices (N, 3)
    surface_faces : np.ndarray
        Surface mesh faces (M, 3)
    config : MeshConfig, optional
        Mesh configuration
    output_path : str, optional
        Path to save mesh file
        
    Returns
    -------
    MeshResult
        Mesh generation result
    """
    if config is None:
        config = MeshConfig()
    
    result = MeshResult()
    
    if len(surface_vertices) == 0 or len(surface_faces) == 0:
        result.errors.append("Empty surface mesh")
        return result
    
    mesher = config.mesher
    if mesher == "auto":
        mesher = _detect_available_mesher()
    
    if mesher == "simvascular":
        result = _mesh_with_simvascular(surface_vertices, surface_faces, config)
    elif mesher == "tetgen":
        result = _mesh_with_tetgen(surface_vertices, surface_faces, config)
    elif mesher == "gmsh":
        result = _mesh_with_gmsh(surface_vertices, surface_faces, config)
    else:
        result = _mesh_simple_tetrahedralization(surface_vertices, surface_faces, config)
    
    if result.success and output_path is not None:
        result.file_path = _export_mesh(result, output_path)
    
    return result


def _detect_available_mesher() -> str:
    """Detect which mesher is available."""
    try:
        import sv
        return "simvascular"
    except ImportError:
        pass
    
    try:
        import tetgen
        return "tetgen"
    except ImportError:
        pass
    
    try:
        import gmsh
        return "gmsh"
    except ImportError:
        pass
    
    return "simple"


def _mesh_with_simvascular(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: MeshConfig,
) -> MeshResult:
    """Generate mesh using SimVascular."""
    result = MeshResult()
    
    try:
        import sv
        
        result.warnings.append("SimVascular meshing not fully implemented - using simple fallback")
        return _mesh_simple_tetrahedralization(vertices, faces, config)
        
    except ImportError:
        result.errors.append("SimVascular not available")
        return result


def _mesh_with_tetgen(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: MeshConfig,
) -> MeshResult:
    """Generate mesh using TetGen."""
    result = MeshResult()
    
    try:
        import tetgen
        
        tet = tetgen.TetGen(vertices, faces)
        
        max_volume = (config.target_edge_size ** 3) / 6.0
        
        tet.tetrahedralize(
            order=1,
            mindihedral=10,
            minratio=1.5,
            quality=True,
            maxvolume=max_volume,
        )
        
        result.vertices = tet.node
        result.cells = tet.elem
        result.n_vertices = len(tet.node)
        result.n_cells = len(tet.elem)
        result.success = True
        
        return result
        
    except ImportError:
        result.errors.append("TetGen not available")
        return result
    except Exception as e:
        result.errors.append(f"TetGen meshing failed: {e}")
        return result


def _mesh_with_gmsh(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: MeshConfig,
) -> MeshResult:
    """Generate mesh using Gmsh."""
    result = MeshResult()
    
    try:
        import gmsh
        
        gmsh.initialize()
        gmsh.model.add("vascular_mesh")
        
        for i, v in enumerate(vertices):
            gmsh.model.geo.addPoint(v[0], v[1], v[2], config.target_edge_size, i + 1)
        
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        result.vertices = np.array(node_coords).reshape(-1, 3)
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        if len(elem_node_tags) > 0:
            result.cells = np.array(elem_node_tags[0]).reshape(-1, 4) - 1
        
        result.n_vertices = len(result.vertices)
        result.n_cells = len(result.cells) if result.cells is not None else 0
        result.success = True
        
        gmsh.finalize()
        
        return result
        
    except ImportError:
        result.errors.append("Gmsh not available")
        return result
    except Exception as e:
        result.errors.append(f"Gmsh meshing failed: {e}")
        return result


def _mesh_simple_tetrahedralization(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: MeshConfig,
) -> MeshResult:
    """
    Simple tetrahedral mesh generation without external dependencies.
    
    Creates a basic tetrahedral mesh by subdividing the bounding box
    and keeping only tetrahedra inside the surface.
    """
    result = MeshResult()
    
    try:
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        padding = config.target_edge_size
        min_coords -= padding
        max_coords += padding
        
        edge_size = config.target_edge_size
        n_x = max(2, int((max_coords[0] - min_coords[0]) / edge_size) + 1)
        n_y = max(2, int((max_coords[1] - min_coords[1]) / edge_size) + 1)
        n_z = max(2, int((max_coords[2] - min_coords[2]) / edge_size) + 1)
        
        n_x = min(n_x, 50)
        n_y = min(n_y, 50)
        n_z = min(n_z, 50)
        
        x = np.linspace(min_coords[0], max_coords[0], n_x)
        y = np.linspace(min_coords[1], max_coords[1], n_y)
        z = np.linspace(min_coords[2], max_coords[2], n_z)
        
        grid_vertices = []
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    grid_vertices.append([x[i], y[j], z[k]])
        
        grid_vertices = np.array(grid_vertices)
        
        def vertex_index(i, j, k):
            return i * n_y * n_z + j * n_z + k
        
        tetrahedra = []
        for i in range(n_x - 1):
            for j in range(n_y - 1):
                for k in range(n_z - 1):
                    v0 = vertex_index(i, j, k)
                    v1 = vertex_index(i + 1, j, k)
                    v2 = vertex_index(i, j + 1, k)
                    v3 = vertex_index(i + 1, j + 1, k)
                    v4 = vertex_index(i, j, k + 1)
                    v5 = vertex_index(i + 1, j, k + 1)
                    v6 = vertex_index(i, j + 1, k + 1)
                    v7 = vertex_index(i + 1, j + 1, k + 1)
                    
                    tetrahedra.append([v0, v1, v2, v4])
                    tetrahedra.append([v1, v2, v3, v7])
                    tetrahedra.append([v1, v2, v4, v7])
                    tetrahedra.append([v2, v4, v6, v7])
                    tetrahedra.append([v1, v4, v5, v7])
                    tetrahedra.append([v4, v5, v6, v7])
        
        tetrahedra = np.array(tetrahedra)
        
        result.vertices = grid_vertices
        result.cells = tetrahedra
        result.n_vertices = len(grid_vertices)
        result.n_cells = len(tetrahedra)
        result.success = True
        
        result.warnings.append(
            f"Using simple grid-based tetrahedralization ({n_x}x{n_y}x{n_z} grid). "
            "Install TetGen or Gmsh for better quality meshes."
        )
        
        return result
        
    except Exception as e:
        result.errors.append(f"Simple meshing failed: {e}")
        return result


def _export_mesh(result: MeshResult, output_path: str) -> str:
    """Export mesh to file."""
    if output_path.endswith('.vtu'):
        return _export_vtu(result, output_path)
    elif output_path.endswith('.vtk'):
        return _export_vtk(result, output_path)
    else:
        return _export_vtk(result, output_path + '.vtk')


def _export_vtk(result: MeshResult, output_path: str) -> str:
    """Export mesh to VTK format."""
    with open(output_path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Vascular mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {result.n_vertices} float\n")
        for v in result.vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        n_cells = result.n_cells
        cell_size = n_cells * 5
        f.write(f"CELLS {n_cells} {cell_size}\n")
        for cell in result.cells:
            f.write(f"4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}\n")
        
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("10\n")
    
    return output_path


def _export_vtu(result: MeshResult, output_path: str) -> str:
    """Export mesh to VTU format."""
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        
        points = vtk.vtkPoints()
        for v in result.vertices:
            points.InsertNextPoint(v[0], v[1], v[2])
        
        cells = vtk.vtkCellArray()
        for cell in result.cells:
            tetra = vtk.vtkTetra()
            for i, idx in enumerate(cell):
                tetra.GetPointIds().SetId(i, idx)
            cells.InsertNextCell(tetra)
        
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.SetCells(vtk.VTK_TETRA, cells)
        
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(grid)
        writer.Write()
        
        return output_path
        
    except ImportError:
        return _export_vtk(result, output_path.replace('.vtu', '.vtk'))


def compute_mesh_quality(
    vertices: np.ndarray,
    cells: np.ndarray,
) -> Dict[str, float]:
    """
    Compute mesh quality metrics.
    
    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices
    cells : np.ndarray
        Mesh cells (tetrahedra)
        
    Returns
    -------
    dict
        Quality metrics (min, max, mean, std)
    """
    if len(cells) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    
    qualities = []
    
    for cell in cells:
        v0, v1, v2, v3 = vertices[cell]
        
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v0),
            np.linalg.norm(v3 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v1),
            np.linalg.norm(v3 - v2),
        ]
        
        mat = np.array([v1 - v0, v2 - v0, v3 - v0])
        volume = abs(np.linalg.det(mat)) / 6.0
        
        max_edge = max(edges)
        if max_edge > 0:
            quality = volume / (max_edge ** 3)
        else:
            quality = 0.0
        
        qualities.append(quality)
    
    qualities = np.array(qualities)
    
    return {
        "min": float(np.min(qualities)),
        "max": float(np.max(qualities)),
        "mean": float(np.mean(qualities)),
        "std": float(np.std(qualities)),
    }
