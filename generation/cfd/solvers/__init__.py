"""
CFD solver implementations for vascular networks.

Provides multifidelity solvers:
- 0D: Lumped parameter model (fast, average pressures/flows)
- 1D: Wave propagation model (intermediate, distributed properties)
- 3D: Full FE CFD (detailed velocity/pressure/WSS fields)

Note: The library uses METERS internally for all geometry.
Pressures are in Pascals (Pa), flows in m^3/s.
"""

from .base import BaseSolver, SolverConfig
from .sv_0d import Solver0D
from .sv_1d import Solver1D
from .sv_3d import Solver3D

__all__ = [
    "BaseSolver",
    "SolverConfig",
    "Solver0D",
    "Solver1D",
    "Solver3D",
]
