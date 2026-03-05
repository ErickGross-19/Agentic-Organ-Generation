from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class MeshDiagnostics:
    watertight: bool
    euler_number: int
    num_vertices: int
    num_faces: int
    num_components: int
    non_manifold_edges: int
    degenerate_faces: int
    bounding_box_extents: List[float]
    volume: Optional[float]
    volume_source: str


@dataclass
class SurfaceQuality:
    min_face_area: float
    max_face_area: float
    mean_face_area: float
    min_edge_length: float
    max_edge_length: float
    mean_edge_length: float
    max_aspect_ratio: float
    frac_aspect_ratio_over_10: float


@dataclass
class ValidationFlags:
    status: str
    flags: List[str]


@dataclass
class ValidationReport:
    input_file: str
    intermediate_stl: Optional[str]
    cleaned_stl: str
    scafold_stl: str

    before: MeshDiagnostics
    after_basic_clean: MeshDiagnostics
    after_voxel: MeshDiagnostics
    after_repair: MeshDiagnostics

    flags: ValidationFlags

    surface_before: SurfaceQuality
    surface_after: SurfaceQuality

    connectivity: Dict[str, Any]
    centerline_summary: Dict[str, Any]
    poiseuille_summary: Dict[str, Any]
    drift_metrics: Optional[Dict[str, Any]] = None
