"""
Open-port validation for embedded vascular networks.

This module provides validation checks to ensure ports (inlets/outlets) are
properly open and connected to the void space after embedding.

"Open port" means there exists a continuous tunnel from outside the domain
into the void at each inlet/outlet location.

Algorithm (voxel connectivity):
1. Voxelize a small local region around each port at validation_pitch
2. Identify outside voxels (known outside domain) and void voxels (inside void)
3. Flood fill from a seed point outside the domain near the port
4. Check connectivity into void region
5. Return pass/fail with diagnostics

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging

from aog_policies.validity import OpenPortPolicy

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


@dataclass
class PortCheckResult:
    """Result of checking a single port."""
    port_id: str
    port_type: str  # "inlet" or "outlet"
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    radius: float
    is_open: bool
    connected_volume_voxels: int = 0
    outside_seed_found: bool = False
    void_reached: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "port_id": self.port_id,
            "port_type": self.port_type,
            "position": self.position,
            "direction": self.direction,
            "radius": self.radius,
            "is_open": self.is_open,
            "connected_volume_voxels": self.connected_volume_voxels,
            "outside_seed_found": self.outside_seed_found,
            "void_reached": self.void_reached,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class OpenPortValidationResult:
    """Result of open-port validation for all ports."""
    success: bool
    all_ports_open: bool
    ports_checked: int
    ports_open: int
    ports_closed: int
    port_results: List[PortCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "all_ports_open": self.all_ports_open,
            "ports_checked": self.ports_checked,
            "ports_open": self.ports_open,
            "ports_closed": self.ports_closed,
            "port_results": [r.to_dict() for r in self.port_results],
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class LocalVoxelPatch:
    """
    Voxelized local region around a port for connectivity checking.
    
    Only voxelizes a small region around the port to avoid memory issues.
    Supports ROI voxel budgeting to ensure deterministic performance.
    """
    
    def __init__(
        self,
        center: np.ndarray,
        size: float,
        pitch: float,
        max_voxels: Optional[int] = None,
        auto_relax_pitch: bool = True,
    ):
        """
        Initialize local voxel patch.
        
        Parameters
        ----------
        center : np.ndarray
            Center of the local region.
        size : float
            Size of the local region (cube side length).
        pitch : float
            Voxel pitch (resolution) in meters.
        max_voxels : int, optional
            Maximum number of voxels allowed in the ROI. If exceeded and
            auto_relax_pitch is True, pitch will be increased.
        auto_relax_pitch : bool
            If True, automatically relax pitch to fit within max_voxels budget.
        """
        self.center = center
        self.size = size
        self.pitch = pitch
        self.pitch_was_relaxed = False
        
        # Compute initial shape
        half_size = size / 2
        self.min_bound = center - half_size
        self.max_bound = center + half_size
        
        shape = np.ceil(size / pitch).astype(int)
        shape = np.maximum(shape, 1)
        total_voxels = int(np.prod(shape))
        
        # ROI voxel budgeting: relax pitch if needed
        if max_voxels is not None and total_voxels > max_voxels and auto_relax_pitch:
            pitch_factor = 1.5
            while total_voxels > max_voxels:
                pitch *= pitch_factor
                shape = np.ceil(size / pitch).astype(int)
                shape = np.maximum(shape, 1)
                total_voxels = int(np.prod(shape))
            self.pitch = pitch
            self.pitch_was_relaxed = True
            logger.warning(
                f"Relaxed open-port validation pitch to {pitch:.6f}m "
                f"to fit within ROI budget ({max_voxels:,} voxels)"
            )
        
        if isinstance(shape, np.ndarray):
            self.shape = tuple(shape.tolist())
        else:
            self.shape = (int(shape), int(shape), int(shape))
        
        self._outside_domain = np.zeros(self.shape, dtype=bool)
        self._void = np.zeros(self.shape, dtype=bool)
        self._solid = np.zeros(self.shape, dtype=bool)
    
    def world_to_voxel(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world position to voxel indices."""
        voxel = np.floor((pos - self.min_bound) / self.pitch).astype(int)
        voxel = np.clip(voxel, 0, np.array(self.shape) - 1)
        return tuple(voxel)
    
    def voxel_to_world(self, voxel: Tuple[int, int, int]) -> np.ndarray:
        """Convert voxel indices to world position (center of voxel)."""
        return self.min_bound + (np.array(voxel) + 0.5) * self.pitch
    
    def is_valid_voxel(self, voxel: Tuple[int, int, int]) -> bool:
        """Check if voxel indices are within bounds."""
        return all(0 <= v < s for v, s in zip(voxel, self.shape))
    
    def classify_from_mesh(
        self,
        domain_mesh: "trimesh.Trimesh",
        void_mesh: Optional["trimesh.Trimesh"] = None,
    ) -> None:
        """
        Classify voxels as outside domain, void, or solid.
        
        Parameters
        ----------
        domain_mesh : trimesh.Trimesh
            The domain mesh (outer boundary).
        void_mesh : trimesh.Trimesh, optional
            The void mesh (embedded channels). If None, uses domain_mesh.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    pos = self.voxel_to_world((i, j, k))
                    
                    in_domain = self._point_in_mesh(pos, domain_mesh)
                    
                    if not in_domain:
                        self._outside_domain[i, j, k] = True
                    elif void_mesh is not None:
                        in_void = self._point_in_mesh(pos, void_mesh)
                        if in_void:
                            self._void[i, j, k] = True
                        else:
                            self._solid[i, j, k] = True
                    else:
                        self._solid[i, j, k] = True
    
    def classify_from_domain_with_void(
        self,
        domain_with_void_mesh: "trimesh.Trimesh",
        original_domain_mesh: "trimesh.Trimesh",
    ) -> None:
        """
        Classify voxels using domain-with-void mesh.
        
        Parameters
        ----------
        domain_with_void_mesh : trimesh.Trimesh
            The domain mesh with void carved out.
        original_domain_mesh : trimesh.Trimesh
            The original domain mesh (before void embedding).
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    pos = self.voxel_to_world((i, j, k))
                    
                    in_original = self._point_in_mesh(pos, original_domain_mesh)
                    
                    if not in_original:
                        self._outside_domain[i, j, k] = True
                    else:
                        in_solid = self._point_in_mesh(pos, domain_with_void_mesh)
                        if in_solid:
                            self._solid[i, j, k] = True
                        else:
                            self._void[i, j, k] = True
    
    def _point_in_mesh(self, point: np.ndarray, mesh: "trimesh.Trimesh") -> bool:
        """Check if a point is inside a mesh."""
        try:
            return bool(mesh.contains([point])[0])
        except Exception:
            return False
    
    def flood_fill_from_outside(
        self,
        seed_direction: np.ndarray,
        seed_distance: float,
        port_position: np.ndarray,
    ) -> Tuple[bool, int, bool]:
        """
        Flood fill from outside the domain toward the port.
        
        Parameters
        ----------
        seed_direction : np.ndarray
            Direction to place the seed (outward from port).
        seed_distance : float
            Distance from port to place the seed.
        port_position : np.ndarray
            Position of the port.
        
        Returns
        -------
        tuple
            (outside_seed_found, connected_voxels, void_reached)
        """
        seed_pos = port_position + seed_direction * seed_distance
        seed_voxel = self.world_to_voxel(seed_pos)
        
        if not self.is_valid_voxel(seed_voxel):
            return False, 0, False
        
        if not self._outside_domain[seed_voxel]:
            for offset in range(1, 10):
                test_pos = port_position + seed_direction * (seed_distance + offset * self.pitch)
                test_voxel = self.world_to_voxel(test_pos)
                if self.is_valid_voxel(test_voxel) and self._outside_domain[test_voxel]:
                    seed_voxel = test_voxel
                    break
            else:
                return False, 0, False
        
        visited = np.zeros(self.shape, dtype=bool)
        queue = deque([seed_voxel])  # Use deque for O(1) popleft
        visited[seed_voxel] = True
        connected_count = 0
        void_reached = False
        
        neighbors = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
        
        while queue:
            current = queue.popleft()  # O(1) instead of O(n) with list.pop(0)
            
            if self._outside_domain[current] or self._void[current]:
                connected_count += 1
                
                if self._void[current]:
                    void_reached = True
            
            for delta in neighbors:
                neighbor = (
                    current[0] + delta[0],
                    current[1] + delta[1],
                    current[2] + delta[2],
                )
                
                if not self.is_valid_voxel(neighbor):
                    continue
                
                if visited[neighbor]:
                    continue
                
                if self._solid[neighbor]:
                    continue
                
                visited[neighbor] = True
                queue.append(neighbor)
        
        return True, connected_count, void_reached


def check_port_open(
    port_position: np.ndarray,
    port_direction: np.ndarray,
    port_radius: float,
    domain_with_void_mesh: "trimesh.Trimesh",
    original_domain_mesh: "trimesh.Trimesh",
    policy: OpenPortPolicy,
    port_id: str = "unknown",
    port_type: str = "unknown",
) -> PortCheckResult:
    """
    Check if a single port is open.
    
    Parameters
    ----------
    port_position : np.ndarray
        Position of the port center.
    port_direction : np.ndarray
        Outward direction of the port.
    port_radius : float
        Radius of the port.
    domain_with_void_mesh : trimesh.Trimesh
        The domain mesh with void carved out.
    original_domain_mesh : trimesh.Trimesh
        The original domain mesh.
    policy : OpenPortPolicy
        Validation policy.
    port_id : str
        Identifier for the port.
    port_type : str
        Type of port ("inlet" or "outlet").
    
    Returns
    -------
    PortCheckResult
        Result of the port check.
    """
    port_direction = port_direction / np.linalg.norm(port_direction)
    
    pitch = policy.validation_pitch
    if pitch is None:
        pitch = port_radius / 4
    
    patch = LocalVoxelPatch(
        center=port_position,
        size=policy.local_region_size,
        pitch=pitch,
        max_voxels=policy.max_voxels_roi,
        auto_relax_pitch=policy.auto_relax_pitch,
    )
    
    patch.classify_from_domain_with_void(
        domain_with_void_mesh=domain_with_void_mesh,
        original_domain_mesh=original_domain_mesh,
    )
    
    outside_seed_found, connected_voxels, void_reached = patch.flood_fill_from_outside(
        seed_direction=port_direction,
        seed_distance=policy.probe_length,
        port_position=port_position,
    )
    
    is_open = (
        outside_seed_found and
        void_reached and
        connected_voxels >= policy.min_connected_volume_voxels
    )
    
    warnings = []
    errors = []
    
    if not outside_seed_found:
        errors.append(f"Could not find outside seed point for port {port_id}")
    elif not void_reached:
        errors.append(f"Port {port_id} does not connect to void space")
    elif connected_voxels < policy.min_connected_volume_voxels:
        warnings.append(
            f"Port {port_id} has small connected volume ({connected_voxels} voxels, "
            f"minimum {policy.min_connected_volume_voxels})"
        )
    
    return PortCheckResult(
        port_id=port_id,
        port_type=port_type,
        position=tuple(port_position),
        direction=tuple(port_direction),
        radius=port_radius,
        is_open=is_open,
        connected_volume_voxels=connected_voxels,
        outside_seed_found=outside_seed_found,
        void_reached=void_reached,
        diagnostics={
            "pitch_used": pitch,
            "local_region_size": policy.local_region_size,
            "probe_length": policy.probe_length,
        },
        warnings=warnings,
        errors=errors,
    )


def check_open_ports(
    ports: List[Dict[str, Any]],
    domain_with_void_mesh: "trimesh.Trimesh",
    original_domain_mesh: "trimesh.Trimesh",
    policy: Optional[OpenPortPolicy] = None,
) -> OpenPortValidationResult:
    """
    Check if all ports are open.
    
    Parameters
    ----------
    ports : list of dict
        List of port specifications with keys:
        - position: (x, y, z) tuple
        - direction: (dx, dy, dz) tuple (outward direction)
        - radius: float
        - type: "inlet" or "outlet" (optional)
        - id: str (optional)
    domain_with_void_mesh : trimesh.Trimesh
        The domain mesh with void carved out.
    original_domain_mesh : trimesh.Trimesh
        The original domain mesh.
    policy : OpenPortPolicy, optional
        Validation policy.
    
    Returns
    -------
    OpenPortValidationResult
        Result of the validation.
    """
    if policy is None:
        policy = OpenPortPolicy()
    
    if not policy.enabled:
        return OpenPortValidationResult(
            success=True,
            all_ports_open=True,
            ports_checked=0,
            ports_open=0,
            ports_closed=0,
            metadata={"skipped": True, "reason": "policy.enabled=False"},
        )
    
    port_results = []
    all_warnings = []
    all_errors = []
    
    for i, port in enumerate(ports):
        port_id = port.get("id", f"port_{i}")
        port_type = port.get("type", "unknown")
        position = np.array(port["position"])
        direction = np.array(port.get("direction", [0, 0, 1]))
        radius = port["radius"]
        
        result = check_port_open(
            port_position=position,
            port_direction=direction,
            port_radius=radius,
            domain_with_void_mesh=domain_with_void_mesh,
            original_domain_mesh=original_domain_mesh,
            policy=policy,
            port_id=port_id,
            port_type=port_type,
        )
        
        port_results.append(result)
        all_warnings.extend(result.warnings)
        all_errors.extend(result.errors)
    
    ports_open = sum(1 for r in port_results if r.is_open)
    ports_closed = len(port_results) - ports_open
    all_ports_open = ports_closed == 0
    
    return OpenPortValidationResult(
        success=all_ports_open,
        all_ports_open=all_ports_open,
        ports_checked=len(port_results),
        ports_open=ports_open,
        ports_closed=ports_closed,
        port_results=port_results,
        warnings=all_warnings,
        errors=all_errors,
        metadata={
            "policy": policy.to_dict(),
        },
    )


__all__ = [
    "OpenPortPolicy",
    "PortCheckResult",
    "OpenPortValidationResult",
    "LocalVoxelPatch",
    "check_port_open",
    "check_open_ports",
]
