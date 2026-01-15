"""
Multi-component compositor for merging vascular network components.

This module provides functionality to compose multiple components (networks,
meshes, channels) into a single unified void mesh for embedding.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import logging

from ..core.network import VascularNetwork
from ..policies import MeshSynthesisPolicy, MeshMergePolicy

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


@dataclass
class ComposePolicy:
    """
    Policy for multi-component composition.
    
    Controls how multiple components are merged into a single void mesh.
    
    JSON Schema:
    {
        "synthesis_policy": MeshSynthesisPolicy,
        "merge_policy": MeshMergePolicy,
        "repair_enabled": bool,
        "repair_voxel_pitch": float (meters),
        "keep_largest_component": bool,
        "min_component_volume": float (cubic meters)
    }
    """
    synthesis_policy: Optional[MeshSynthesisPolicy] = None
    merge_policy: Optional[MeshMergePolicy] = None
    repair_enabled: bool = True
    repair_voxel_pitch: float = 5e-5  # 50um
    keep_largest_component: bool = True
    min_component_volume: float = 1e-12  # 1 cubic mm
    
    def __post_init__(self):
        if self.synthesis_policy is None:
            self.synthesis_policy = MeshSynthesisPolicy()
        if self.merge_policy is None:
            self.merge_policy = MeshMergePolicy()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "synthesis_policy": self.synthesis_policy.to_dict() if self.synthesis_policy else None,
            "merge_policy": self.merge_policy.to_dict() if self.merge_policy else None,
            "repair_enabled": self.repair_enabled,
            "repair_voxel_pitch": self.repair_voxel_pitch,
            "keep_largest_component": self.keep_largest_component,
            "min_component_volume": self.min_component_volume,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComposePolicy":
        synthesis_policy = None
        merge_policy = None
        
        if d.get("synthesis_policy"):
            synthesis_policy = MeshSynthesisPolicy.from_dict(d["synthesis_policy"])
        if d.get("merge_policy"):
            merge_policy = MeshMergePolicy.from_dict(d["merge_policy"])
        
        return cls(
            synthesis_policy=synthesis_policy,
            merge_policy=merge_policy,
            repair_enabled=d.get("repair_enabled", True),
            repair_voxel_pitch=d.get("repair_voxel_pitch", 5e-5),
            keep_largest_component=d.get("keep_largest_component", True),
            min_component_volume=d.get("min_component_volume", 1e-12),
        )


@dataclass
class ComponentSpec:
    """
    Specification for a single component in the composition.
    
    A component can be a VascularNetwork, a trimesh.Trimesh, or a path
    to an STL file.
    """
    type: Literal["network", "mesh", "file"]
    data: Union[VascularNetwork, "trimesh.Trimesh", str]
    name: str = ""
    transform: Optional[np.ndarray] = None
    
    @classmethod
    def from_network(
        cls,
        network: VascularNetwork,
        name: str = "network",
    ) -> "ComponentSpec":
        """Create a component from a VascularNetwork."""
        return cls(type="network", data=network, name=name)
    
    @classmethod
    def from_mesh(
        cls,
        mesh: "trimesh.Trimesh",
        name: str = "mesh",
    ) -> "ComponentSpec":
        """Create a component from a trimesh.Trimesh."""
        return cls(type="mesh", data=mesh, name=name)
    
    @classmethod
    def from_file(
        cls,
        path: str,
        name: str = "file",
    ) -> "ComponentSpec":
        """Create a component from an STL file path."""
        return cls(type="file", data=path, name=name)


@dataclass
class ComposeReport:
    """Report from a composition operation."""
    success: bool
    components_processed: int = 0
    meshes_merged: int = 0
    vertex_count: int = 0
    face_count: int = 0
    volume: float = 0.0
    is_watertight: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    component_reports: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "components_processed": self.components_processed,
            "meshes_merged": self.meshes_merged,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "volume": self.volume,
            "is_watertight": self.is_watertight,
            "warnings": self.warnings,
            "errors": self.errors,
            "component_reports": self.component_reports,
            "metadata": self.metadata,
        }


def compose_components(
    components: List[Union[ComponentSpec, VascularNetwork, "trimesh.Trimesh"]],
    policy: Optional[ComposePolicy] = None,
) -> Tuple["trimesh.Trimesh", ComposeReport]:
    """
    Compose multiple components into a single void mesh.
    
    This function takes a list of components (networks, meshes, or file paths)
    and merges them into a single unified mesh suitable for embedding.
    
    Parameters
    ----------
    components : list
        List of components to compose. Each can be:
        - ComponentSpec: Full specification with type and options
        - VascularNetwork: Converted to mesh using synthesis policy
        - trimesh.Trimesh: Used directly
    policy : ComposePolicy, optional
        Policy controlling composition behavior
        
    Returns
    -------
    void_mesh : trimesh.Trimesh
        Merged void mesh
    report : ComposeReport
        Report with composition statistics
    """
    import trimesh
    
    if policy is None:
        policy = ComposePolicy()
    
    warnings = []
    errors = []
    component_reports = []
    meshes_to_merge = []
    
    # Process each component
    for i, component in enumerate(components):
        comp_report = {"index": i, "success": False}
        
        try:
            # Normalize to ComponentSpec
            if isinstance(component, ComponentSpec):
                spec = component
            elif isinstance(component, VascularNetwork):
                spec = ComponentSpec.from_network(component, name=f"network_{i}")
            elif hasattr(component, 'vertices') and hasattr(component, 'faces'):
                spec = ComponentSpec.from_mesh(component, name=f"mesh_{i}")
            else:
                warnings.append(f"Unknown component type at index {i}: {type(component)}")
                continue
            
            comp_report["name"] = spec.name
            comp_report["type"] = spec.type
            
            # Convert to mesh
            if spec.type == "network":
                mesh = _network_to_mesh(spec.data, policy.synthesis_policy)
                comp_report["source"] = "network_synthesis"
            elif spec.type == "mesh":
                mesh = spec.data.copy()
                comp_report["source"] = "direct_mesh"
            elif spec.type == "file":
                mesh = trimesh.load(spec.data)
                comp_report["source"] = f"file:{spec.data}"
            else:
                warnings.append(f"Unknown component type: {spec.type}")
                continue
            
            # Apply transform if specified
            if spec.transform is not None:
                mesh.apply_transform(spec.transform)
                comp_report["transform_applied"] = True
            
            # Validate mesh
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                warnings.append(f"Component {spec.name} produced empty mesh")
                continue
            
            comp_report["vertex_count"] = len(mesh.vertices)
            comp_report["face_count"] = len(mesh.faces)
            comp_report["volume"] = float(abs(mesh.volume))
            comp_report["success"] = True
            
            meshes_to_merge.append(mesh)
            
        except Exception as e:
            logger.exception(f"Error processing component {i}")
            comp_report["error"] = str(e)
            errors.append(f"Component {i}: {e}")
        
        component_reports.append(comp_report)
    
    # Check if we have any meshes to merge
    if not meshes_to_merge:
        return trimesh.Trimesh(), ComposeReport(
            success=False,
            errors=["No valid meshes to merge"] + errors,
            warnings=warnings,
            component_reports=component_reports,
        )
    
    # Merge meshes
    if len(meshes_to_merge) == 1:
        merged = meshes_to_merge[0].copy()
    else:
        merged, merge_report = _merge_meshes(meshes_to_merge, policy.merge_policy)
        
        if merge_report.warnings:
            warnings.extend(merge_report.warnings)
    
    # Apply repair if enabled
    if policy.repair_enabled:
        merged = _repair_mesh(merged, policy.repair_voxel_pitch)
    
    # Keep largest component if enabled
    if policy.keep_largest_component and not merged.is_watertight:
        merged = _keep_largest_component(merged, policy.min_component_volume)
    
    # Final cleanup
    merged.merge_vertices()
    merged.remove_unreferenced_vertices()
    
    if merged.volume < 0:
        merged.invert()
    
    trimesh.repair.fix_normals(merged)
    
    # Build report
    report = ComposeReport(
        success=True,
        components_processed=len(components),
        meshes_merged=len(meshes_to_merge),
        vertex_count=len(merged.vertices),
        face_count=len(merged.faces),
        volume=float(abs(merged.volume)),
        is_watertight=merged.is_watertight,
        warnings=warnings,
        errors=errors,
        component_reports=component_reports,
        metadata={
            "policy": policy.to_dict(),
        },
    )
    
    return merged, report


def _network_to_mesh(
    network: VascularNetwork,
    policy: Optional[MeshSynthesisPolicy] = None,
) -> "trimesh.Trimesh":
    """Convert a VascularNetwork to a mesh."""
    from .mesh.synthesis import synthesize_mesh
    
    if policy is None:
        policy = MeshSynthesisPolicy()
    
    mesh, _ = synthesize_mesh(network, policy)
    return mesh


def _merge_meshes(
    meshes: List["trimesh.Trimesh"],
    policy: Optional[MeshMergePolicy] = None,
) -> Tuple["trimesh.Trimesh", Any]:
    """Merge multiple meshes using the merge policy."""
    from .mesh.merge import merge_meshes
    
    if policy is None:
        policy = MeshMergePolicy()
    
    return merge_meshes(meshes, policy)


def _repair_mesh(
    mesh: "trimesh.Trimesh",
    voxel_pitch: float,
) -> "trimesh.Trimesh":
    """Apply voxel-based repair to a mesh."""
    import trimesh
    
    try:
        # Voxelize
        voxels = mesh.voxelized(voxel_pitch)
        voxels = voxels.fill()
        
        # Reconstruct
        repaired = voxels.marching_cubes
        
        # Check for coordinate system issues
        in_extent = float(np.max(mesh.extents))
        out_extent = float(np.max(repaired.extents))
        
        if in_extent > 0 and out_extent / in_extent > 50:
            repaired.apply_transform(voxels.transform)
        
        repaired.merge_vertices()
        repaired.remove_unreferenced_vertices()
        
        if repaired.volume < 0:
            repaired.invert()
        
        trimesh.repair.fix_normals(repaired)
        
        return repaired
        
    except Exception as e:
        logger.warning(f"Voxel repair failed: {e}, returning original mesh")
        return mesh


def _keep_largest_component(
    mesh: "trimesh.Trimesh",
    min_volume: float,
) -> "trimesh.Trimesh":
    """Keep only the largest connected component of a mesh."""
    import trimesh
    
    try:
        # Split into connected components
        components = mesh.split(only_watertight=False)
        
        if len(components) <= 1:
            return mesh
        
        # Find largest by volume
        largest = None
        largest_volume = 0
        
        for comp in components:
            vol = abs(comp.volume)
            if vol > largest_volume:
                largest_volume = vol
                largest = comp
        
        if largest is not None and largest_volume >= min_volume:
            return largest
        
        return mesh
        
    except Exception as e:
        logger.warning(f"Component filtering failed: {e}")
        return mesh


def compose_network_and_channels(
    network: VascularNetwork,
    channel_meshes: List["trimesh.Trimesh"],
    policy: Optional[ComposePolicy] = None,
) -> Tuple["trimesh.Trimesh", ComposeReport]:
    """
    Convenience function to compose a network with channel meshes.
    
    This is a common use case where a generated network is combined
    with additional channel geometries (e.g., fang hooks, path channels).
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to include
    channel_meshes : list of trimesh.Trimesh
        Additional channel meshes to merge
    policy : ComposePolicy, optional
        Composition policy
        
    Returns
    -------
    void_mesh : trimesh.Trimesh
        Merged void mesh
    report : ComposeReport
        Composition report
    """
    components = [ComponentSpec.from_network(network, name="network")]
    
    for i, mesh in enumerate(channel_meshes):
        components.append(ComponentSpec.from_mesh(mesh, name=f"channel_{i}"))
    
    return compose_components(components, policy)


def compose_dual_tree(
    arterial_network: VascularNetwork,
    venous_network: VascularNetwork,
    policy: Optional[ComposePolicy] = None,
) -> Tuple["trimesh.Trimesh", ComposeReport]:
    """
    Convenience function to compose arterial and venous networks.
    
    Parameters
    ----------
    arterial_network : VascularNetwork
        Arterial tree network
    venous_network : VascularNetwork
        Venous tree network
    policy : ComposePolicy, optional
        Composition policy
        
    Returns
    -------
    void_mesh : trimesh.Trimesh
        Merged void mesh containing both trees
    report : ComposeReport
        Composition report
    """
    components = [
        ComponentSpec.from_network(arterial_network, name="arterial"),
        ComponentSpec.from_network(venous_network, name="venous"),
    ]
    
    return compose_components(components, policy)


__all__ = [
    "compose_components",
    "compose_network_and_channels",
    "compose_dual_tree",
    "ComposePolicy",
    "ComponentSpec",
    "ComposeReport",
]
