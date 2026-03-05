"""
Orchestration entrypoint for network refinement.

Provides high-level functions for geometry and topology optimization
that can be called from the generation pipeline.

Refinement stages:
1. Geometry optimization (NLP) - optimizes node positions, radii, pressures
2. Topology optimization (SA) - swaps subtrees to reduce objective

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal

from ..core.network import VascularNetwork
from .nlp_geometry import NLPConfig, NLPResult, optimize_geometry
from .topology_swaps import TopologyConfig, TopologyResult, optimize_topology


@dataclass
class RefinementConfig:
    """Configuration for network refinement."""
    
    enable_geometry_optimization: bool = True
    enable_topology_optimization: bool = True
    
    geometry_rounds: int = 1
    topology_rounds: int = 1
    
    nlp_config: Optional[NLPConfig] = None
    topology_config: Optional[TopologyConfig] = None
    
    interleave_rounds: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_geometry_optimization": self.enable_geometry_optimization,
            "enable_topology_optimization": self.enable_topology_optimization,
            "geometry_rounds": self.geometry_rounds,
            "topology_rounds": self.topology_rounds,
            "interleave_rounds": self.interleave_rounds,
        }


@dataclass
class RefinementResult:
    """Result of network refinement."""
    
    success: bool = False
    
    initial_volume: float = 0.0
    final_volume: float = 0.0
    total_volume_reduction: float = 0.0
    
    geometry_results: List[NLPResult] = field(default_factory=list)
    topology_results: List[TopologyResult] = field(default_factory=list)
    
    total_geometry_rounds: int = 0
    total_topology_rounds: int = 0
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "initial_volume": self.initial_volume,
            "final_volume": self.final_volume,
            "total_volume_reduction": self.total_volume_reduction,
            "total_geometry_rounds": self.total_geometry_rounds,
            "total_topology_rounds": self.total_topology_rounds,
            "geometry_results": [r.to_dict() for r in self.geometry_results],
            "topology_results": [r.to_dict() for r in self.topology_results],
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _compute_total_volume(network: VascularNetwork) -> float:
    """Compute total vessel volume."""
    import math
    total = 0.0
    for segment in network.segments.values():
        r = segment.geometry.mean_radius()
        L = segment.geometry.length()
        total += math.pi * r ** 2 * L
    return total


def refine_geometry(
    network: VascularNetwork,
    config: Optional[NLPConfig] = None,
    rounds: int = 1,
) -> List[NLPResult]:
    """
    Refine network geometry using NLP optimization.
    
    Optimizes node positions, radii, and pressures to minimize
    total vessel volume while satisfying physiological constraints.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to refine (modified in place)
    config : NLPConfig, optional
        NLP configuration
    rounds : int
        Number of optimization rounds
        
    Returns
    -------
    list
        List of NLPResult for each round
    """
    if config is None:
        config = NLPConfig()
    
    results = []
    
    for i in range(rounds):
        result = optimize_geometry(network, config)
        results.append(result)
        
        if not result.success:
            break
    
    return results


def refine_topology(
    network: VascularNetwork,
    config: Optional[TopologyConfig] = None,
    rounds: int = 1,
) -> List[TopologyResult]:
    """
    Refine network topology using subtree swapping.
    
    Swaps subtrees and accepts/rejects using simulated annealing
    to reduce total vessel volume.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to refine (modified in place)
    config : TopologyConfig, optional
        Topology optimization configuration
    rounds : int
        Number of optimization rounds
        
    Returns
    -------
    list
        List of TopologyResult for each round
    """
    if config is None:
        config = TopologyConfig()
    
    results = []
    
    for i in range(rounds):
        result = optimize_topology(network, config)
        results.append(result)
        
        if not result.success:
            break
    
    return results


def refine_network(
    network: VascularNetwork,
    config: Optional[RefinementConfig] = None,
) -> RefinementResult:
    """
    Refine network using both geometry and topology optimization.
    
    This is the main entry point for network refinement. It runs
    geometry optimization (NLP) and/or topology optimization (SA)
    based on the configuration.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to refine (modified in place)
    config : RefinementConfig, optional
        Refinement configuration
        
    Returns
    -------
    RefinementResult
        Combined refinement result
    """
    if config is None:
        config = RefinementConfig()
    
    result = RefinementResult()
    result.initial_volume = _compute_total_volume(network)
    
    nlp_config = config.nlp_config or NLPConfig()
    topo_config = config.topology_config or TopologyConfig()
    
    if config.interleave_rounds:
        max_rounds = max(config.geometry_rounds, config.topology_rounds)
        
        for i in range(max_rounds):
            if config.enable_geometry_optimization and i < config.geometry_rounds:
                geo_results = refine_geometry(network, nlp_config, rounds=1)
                result.geometry_results.extend(geo_results)
                result.total_geometry_rounds += 1
                
                if geo_results and not geo_results[0].success:
                    result.warnings.append(f"Geometry round {i+1} did not converge")
            
            if config.enable_topology_optimization and i < config.topology_rounds:
                topo_results = refine_topology(network, topo_config, rounds=1)
                result.topology_results.extend(topo_results)
                result.total_topology_rounds += 1
                
                if topo_results and not topo_results[0].success:
                    result.warnings.append(f"Topology round {i+1} did not converge")
    else:
        if config.enable_geometry_optimization:
            geo_results = refine_geometry(network, nlp_config, rounds=config.geometry_rounds)
            result.geometry_results = geo_results
            result.total_geometry_rounds = len(geo_results)
            
            for i, geo_result in enumerate(geo_results):
                if not geo_result.success:
                    result.warnings.append(f"Geometry round {i+1} did not converge")
        
        if config.enable_topology_optimization:
            topo_results = refine_topology(network, topo_config, rounds=config.topology_rounds)
            result.topology_results = topo_results
            result.total_topology_rounds = len(topo_results)
            
            for i, topo_result in enumerate(topo_results):
                if not topo_result.success:
                    result.warnings.append(f"Topology round {i+1} did not converge")
    
    result.final_volume = _compute_total_volume(network)
    
    if result.initial_volume > 0:
        result.total_volume_reduction = (
            (result.initial_volume - result.final_volume) / result.initial_volume
        )
    
    result.success = True
    
    return result


def quick_refine(
    network: VascularNetwork,
    mode: Literal["geometry", "topology", "both"] = "both",
) -> RefinementResult:
    """
    Quick refinement with default settings.
    
    Convenience function for simple refinement without configuration.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to refine (modified in place)
    mode : str
        Refinement mode: "geometry", "topology", or "both"
        
    Returns
    -------
    RefinementResult
        Refinement result
    """
    config = RefinementConfig(
        enable_geometry_optimization=(mode in ["geometry", "both"]),
        enable_topology_optimization=(mode in ["topology", "both"]),
        geometry_rounds=1,
        topology_rounds=1,
    )
    
    return refine_network(network, config)


def iterative_refine(
    network: VascularNetwork,
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    config: Optional[RefinementConfig] = None,
) -> RefinementResult:
    """
    Iteratively refine network until convergence.
    
    Runs refinement rounds until volume reduction falls below
    the convergence threshold or max iterations is reached.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to refine (modified in place)
    max_iterations : int
        Maximum number of iterations
    convergence_threshold : float
        Stop when volume reduction < threshold
    config : RefinementConfig, optional
        Base refinement configuration
        
    Returns
    -------
    RefinementResult
        Combined refinement result
    """
    if config is None:
        config = RefinementConfig()
    
    config.geometry_rounds = 1
    config.topology_rounds = 1
    
    combined_result = RefinementResult()
    combined_result.initial_volume = _compute_total_volume(network)
    
    prev_volume = combined_result.initial_volume
    
    for iteration in range(max_iterations):
        round_result = refine_network(network, config)
        
        combined_result.geometry_results.extend(round_result.geometry_results)
        combined_result.topology_results.extend(round_result.topology_results)
        combined_result.total_geometry_rounds += round_result.total_geometry_rounds
        combined_result.total_topology_rounds += round_result.total_topology_rounds
        combined_result.warnings.extend(round_result.warnings)
        
        current_volume = _compute_total_volume(network)
        
        if prev_volume > 0:
            volume_reduction = (prev_volume - current_volume) / prev_volume
        else:
            volume_reduction = 0.0
        
        if volume_reduction < convergence_threshold:
            break
        
        prev_volume = current_volume
    
    combined_result.final_volume = _compute_total_volume(network)
    
    if combined_result.initial_volume > 0:
        combined_result.total_volume_reduction = (
            (combined_result.initial_volume - combined_result.final_volume) 
            / combined_result.initial_volume
        )
    
    combined_result.success = True
    
    return combined_result
