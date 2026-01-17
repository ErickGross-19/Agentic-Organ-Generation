"""
Base solver interface for CFD simulations.

Defines the abstract interface that all solver implementations must follow.

Note: The library uses METERS internally for all geometry.
Pressures are in Pascals (Pa), flows in m^3/s.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal

from ..bcs import BoundaryConditions
from ..results import CFDResult


@dataclass
class SolverConfig:
    """Configuration for CFD solver."""
    
    n_time_steps: int = 100
    time_step_size: float = 0.001
    steady_state: bool = True
    
    solver_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    linear_solver: str = "direct"
    preconditioner: str = "ilu"
    
    output_frequency: int = 10
    output_vtk: bool = True
    output_csv: bool = True
    
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_time_steps": self.n_time_steps,
            "time_step_size": self.time_step_size,
            "steady_state": self.steady_state,
            "solver_tolerance": self.solver_tolerance,
            "max_iterations": self.max_iterations,
            "linear_solver": self.linear_solver,
            "preconditioner": self.preconditioner,
            "output_frequency": self.output_frequency,
            "output_vtk": self.output_vtk,
            "output_csv": self.output_csv,
            "verbose": self.verbose,
        }


class BaseSolver(ABC):
    """
    Abstract base class for CFD solvers.
    
    All solver implementations (0D, 1D, 3D) must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize solver.
        
        Parameters
        ----------
        config : SolverConfig, optional
            Solver configuration
        """
        self.config = config or SolverConfig()
        self._is_initialized = False
    
    @property
    @abstractmethod
    def fidelity(self) -> Literal["0D", "1D", "3D"]:
        """Return solver fidelity level."""
        pass
    
    @property
    @abstractmethod
    def requires_mesh(self) -> bool:
        """Return whether solver requires volumetric mesh."""
        pass
    
    @abstractmethod
    def setup(
        self,
        network: "VascularNetwork",
        bcs: BoundaryConditions,
        mesh_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Set up solver with network, boundary conditions, and optional mesh.
        
        Parameters
        ----------
        network : VascularNetwork
            Vascular network to simulate
        bcs : BoundaryConditions
            Boundary conditions
        mesh_data : dict, optional
            Mesh data (required for 3D solver)
            
        Returns
        -------
        bool
            True if setup successful
        """
        pass
    
    @abstractmethod
    def solve(self) -> CFDResult:
        """
        Run the simulation.
        
        Returns
        -------
        CFDResult
            Simulation results
        """
        pass
    
    @abstractmethod
    def get_node_pressures(self) -> Dict[int, float]:
        """
        Get pressure at each node.
        
        Returns
        -------
        dict
            Node ID -> pressure (Pa)
        """
        pass
    
    @abstractmethod
    def get_segment_flows(self) -> Dict[int, float]:
        """
        Get flow through each segment.
        
        Returns
        -------
        dict
            Segment ID -> flow (m^3/s)
        """
        pass
    
    def validate_setup(self) -> List[str]:
        """
        Validate solver setup.
        
        Returns
        -------
        list
            List of validation errors (empty if valid)
        """
        errors = []
        if not self._is_initialized:
            errors.append("Solver not initialized - call setup() first")
        return errors
    
    def reset(self) -> None:
        """Reset solver state."""
        self._is_initialized = False


def compute_poiseuille_resistance(
    length: float,
    radius: float,
    viscosity: float = 0.0035,
) -> float:
    """
    Compute Poiseuille resistance for a cylindrical vessel.
    
    R = 8 * mu * L / (pi * r^4)
    
    Parameters
    ----------
    length : float
        Vessel length (m)
    radius : float
        Vessel radius (m)
    viscosity : float
        Dynamic viscosity (Pa.s), default blood viscosity
        
    Returns
    -------
    float
        Hydraulic resistance (Pa.s/m^3)
    """
    import math
    
    if radius <= 0:
        return float('inf')
    
    return (8.0 * viscosity * length) / (math.pi * radius ** 4)


def compute_segment_resistance(
    segment: "VesselSegment",
    viscosity: float = 0.0035,
) -> float:
    """
    Compute Poiseuille resistance for a vessel segment.
    
    Uses mean radius for tapered segments.
    
    Parameters
    ----------
    segment : VesselSegment
        Vessel segment
    viscosity : float
        Dynamic viscosity (Pa.s)
        
    Returns
    -------
    float
        Hydraulic resistance (Pa.s/m^3)
    """
    length = segment.geometry.length()
    mean_radius = segment.geometry.mean_radius()
    
    return compute_poiseuille_resistance(length, mean_radius, viscosity)


def build_resistance_network(
    network: "VascularNetwork",
    viscosity: float = 0.0035,
) -> Dict[int, float]:
    """
    Build resistance values for all segments in network.
    
    Parameters
    ----------
    network : VascularNetwork
        Vascular network
    viscosity : float
        Dynamic viscosity (Pa.s)
        
    Returns
    -------
    dict
        Segment ID -> resistance (Pa.s/m^3)
    """
    resistances = {}
    
    for seg_id, segment in network.segments.items():
        resistances[seg_id] = compute_segment_resistance(segment, viscosity)
    
    return resistances
