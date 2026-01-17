"""
CFD (Computational Fluid Dynamics) coupling for vascular networks.

This module provides an automated pipeline for:
1. Converting VascularNetwork to watertight 3D geometry
2. Generating volumetric meshes for simulation
3. Assigning boundary conditions (inlet/outlet)
4. Running multifidelity simulations (0D/1D/3D)
5. Computing hemodynamic metrics

Based on Sexton et al.'s automated modeling pipeline for synthetic vasculature.

Note: The library uses METERS internally for all geometry.
"""

from .pipeline import run_cfd_pipeline, CFDConfig
from .results import CFDResult, CFDMetrics
from .geometry import build_watertight_geometry, GeometryConfig
from .meshing import generate_mesh, MeshConfig
from .bcs import BoundaryConditions, InletBC, OutletBC

__all__ = [
    "run_cfd_pipeline",
    "CFDConfig",
    "CFDResult",
    "CFDMetrics",
    "build_watertight_geometry",
    "GeometryConfig",
    "generate_mesh",
    "MeshConfig",
    "BoundaryConditions",
    "InletBC",
    "OutletBC",
]
