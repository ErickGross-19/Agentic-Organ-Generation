"""
Base interface for vascular network generation backends.

This module defines the abstract interface that all generation backends must implement,
providing a unified API for different generation methods (CCO, space colonization, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from ..core.network import VascularNetwork
from ..core.domain import DomainSpec


@dataclass
class BackendConfig:
    """Base configuration for generation backends."""
    
    seed: Optional[int] = None
    min_segment_length: float = 0.0005  # meters
    max_segment_length: float = 0.020  # meters
    min_radius: float = 0.0001  # meters
    min_terminal_separation: float = 0.0005  # meters
    check_collisions: bool = True
    collision_clearance: float = 0.0002  # meters


@dataclass
class GenerationState:
    """State object for incremental generation."""
    
    network: VascularNetwork
    iteration: int = 0
    remaining_outlets: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationAction:
    """Action for step-based generation."""
    
    action_type: str  # "add_outlet", "force_outlet", "close_loop", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)


class GenerationBackend(ABC):
    """
    Abstract base class for vascular network generation backends.
    
    All backends must implement the generate() method and declare their capabilities
    via the supports_dual_tree and supports_closed_loop properties.
    
    Output must always be a VascularNetwork instance.
    """
    
    @property
    @abstractmethod
    def supports_dual_tree(self) -> bool:
        """Whether this backend can generate dual arterial-venous trees."""
        pass
    
    @property
    @abstractmethod
    def supports_closed_loop(self) -> bool:
        """Whether this backend can generate closed-loop (A-V connected) networks."""
        pass
    
    @abstractmethod
    def generate(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlet_position: np.ndarray,
        inlet_radius: float,
        vessel_type: str = "arterial",
        config: Optional[BackendConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a vascular network within the given domain.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        num_outlets : int
            Target number of terminal outlets
        inlet_position : np.ndarray
            Position of the inlet node (x, y, z) in meters
        inlet_radius : float
            Radius of the inlet vessel in meters
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : BackendConfig, optional
            Backend configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network
        """
        pass
    
    def step(
        self,
        state: GenerationState,
        action: GenerationAction,
    ) -> GenerationState:
        """
        Perform a single generation step (optional, for agentic iteration).
        
        This method allows incremental generation where an agent can control
        each step of the process.
        
        Parameters
        ----------
        state : GenerationState
            Current generation state
        action : GenerationAction
            Action to perform
            
        Returns
        -------
        GenerationState
            Updated generation state
            
        Notes
        -----
        Default implementation raises NotImplementedError.
        Backends that support step-based generation should override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support step-based generation"
        )
    
    def generate_dual_tree(
        self,
        domain: DomainSpec,
        arterial_outlets: int,
        venous_outlets: int,
        arterial_inlet: np.ndarray,
        venous_outlet: np.ndarray,
        arterial_radius: float,
        venous_radius: float,
        config: Optional[BackendConfig] = None,
        rng_seed: Optional[int] = None,
        create_anastomoses: bool = False,
        num_anastomoses: int = 0,
    ) -> VascularNetwork:
        """
        Generate a dual arterial-venous network.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        arterial_outlets : int
            Number of arterial terminal outlets
        venous_outlets : int
            Number of venous terminal outlets
        arterial_inlet : np.ndarray
            Position of arterial inlet (x, y, z) in meters
        venous_outlet : np.ndarray
            Position of venous outlet (x, y, z) in meters
        arterial_radius : float
            Radius of arterial inlet in meters
        venous_radius : float
            Radius of venous outlet in meters
        config : BackendConfig, optional
            Backend configuration
        rng_seed : int, optional
            Random seed for reproducibility
        create_anastomoses : bool
            Whether to create A-V anastomoses
        num_anastomoses : int
            Number of anastomoses to create (if create_anastomoses is True)
            
        Returns
        -------
        VascularNetwork
            Generated dual-tree vascular network
            
        Raises
        ------
        NotImplementedError
            If backend does not support dual tree generation
        """
        if not self.supports_dual_tree:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support dual tree generation"
            )
        raise NotImplementedError("Subclass must implement generate_dual_tree")
