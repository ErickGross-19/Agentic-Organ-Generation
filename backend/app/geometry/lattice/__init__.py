"""
Lattice scaffold generators.

This module provides basic and advanced lattice generators for tissue engineering:

Basic:
- Cubic: Simple cubic lattice with 12 edge struts per cell
- BCC: Body-centered cubic with center node and diagonal connections

Advanced TPMS (Triply Periodic Minimal Surfaces):
- Gyroid: TPMS with excellent permeability and mechanical properties
- Schwarz P: TPMS with cubic symmetry topology

Strut-Based:
- Octet Truss: High strength-to-weight ratio lattice
- Voronoi: Organic, randomized cellular structure
- Honeycomb: Hexagonal cells for load-bearing applications
"""

# Basic lattice (cubic, BCC)
from .basic import (
    LatticeParams,
    make_strut,
    get_cubic_struts,
    get_bcc_struts,
    generate_lattice,
    generate_lattice_from_dict,
)

# Advanced TPMS
from .gyroid import (
    GyroidParams,
    generate_gyroid,
    generate_gyroid_from_dict,
)
from .schwarz_p import (
    SchwarzPParams,
    generate_schwarz_p,
    generate_schwarz_p_from_dict,
)

# Advanced strut-based
from .octet_truss import (
    OctetTrussParams,
    generate_octet_truss,
    generate_octet_truss_from_dict,
)
from .voronoi import (
    VoronoiParams,
    generate_voronoi,
    generate_voronoi_from_dict,
)
from .honeycomb import (
    HoneycombParams,
    generate_honeycomb,
    generate_honeycomb_from_dict,
)

__all__ = [
    # Basic lattice
    "LatticeParams",
    "make_strut",
    "get_cubic_struts",
    "get_bcc_struts",
    "generate_lattice",
    "generate_lattice_from_dict",
    # Gyroid TPMS
    "GyroidParams",
    "generate_gyroid",
    "generate_gyroid_from_dict",
    # Schwarz P TPMS
    "SchwarzPParams",
    "generate_schwarz_p",
    "generate_schwarz_p_from_dict",
    # Octet Truss
    "OctetTrussParams",
    "generate_octet_truss",
    "generate_octet_truss_from_dict",
    # Voronoi
    "VoronoiParams",
    "generate_voronoi",
    "generate_voronoi_from_dict",
    # Honeycomb
    "HoneycombParams",
    "generate_honeycomb",
    "generate_honeycomb_from_dict",
]
