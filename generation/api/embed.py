"""
Unified embedding API for vascular network voids.

This module provides the main entry point for embedding void meshes
into domain volumes as negative space.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, Tuple, Dict, Any, Union, List, TYPE_CHECKING
from pathlib import Path
import logging

# Import policies from aog_policies (canonical source for runner contract)
from aog_policies import EmbeddingPolicy, OperationReport
from aog_policies.resolution import ResolutionPolicy
from ..specs.design_spec import DomainSpec
from ..specs.compile import compile_domain
from ..core.domain import (
    BoxDomain,
    EllipsoidDomain,
    CylinderDomain,
    DomainSpec as RuntimeDomainSpec,
    domain_from_dict,
)

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def _coerce_domain(domain: Union[Dict[str, Any], DomainSpec, "RuntimeDomainSpec"]) -> "RuntimeDomainSpec":
    """
    Coerce domain input to a runtime Domain object.
    
    This helper supports three input types:
    1. Dict domain spec (runner/JSON): {"type": "box", "x_min": ..., ...}
    2. Runtime Domain objects (already compiled): BoxDomain, CylinderDomain, etc.
    3. Legacy spec dataclasses: DomainSpec from specs.design_spec
    
    Parameters
    ----------
    domain : dict, DomainSpec, or RuntimeDomainSpec
        Domain specification in any supported format
        
    Returns
    -------
    RuntimeDomainSpec
        Compiled runtime domain object
        
    Raises
    ------
    ValueError
        If domain type is not recognized
    """
    if isinstance(domain, dict):
        return domain_from_dict(domain)
    
    if isinstance(domain, RuntimeDomainSpec):
        return domain
    
    if hasattr(domain, 'type') and hasattr(domain, 'to_dict'):
        return compile_domain(domain)
    
    raise ValueError(
        f"Cannot coerce domain of type {type(domain).__name__}. "
        "Expected dict, RuntimeDomainSpec, or legacy DomainSpec."
    )


def embed_void(
    domain: Union[DomainSpec, "Domain"],
    void_mesh: Union["trimesh.Trimesh", str, Path],
    embedding_policy: Optional[EmbeddingPolicy] = None,
    ports: Optional[List[Dict[str, Any]]] = None,
    resolution_policy: Optional[ResolutionPolicy] = None,
    domain_mesh: Optional["trimesh.Trimesh"] = None,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", "trimesh.Trimesh", OperationReport]:
    """
    Embed a void mesh into a domain as negative space.
    
    This is the unified entry point for embedding operations, supporting
    both in-memory meshes and file paths.
    
    B2/B3 FIX: Now supports resolution policy for unified pitch selection:
    - If use_resolution_policy=True, uses the resolution resolver for pitch derivation
    - Respects max_voxels budget with deterministic pitch relaxation
    - Reports effective pitch and warnings in metadata
    
    Parameters
    ----------
    domain : DomainSpec or Domain
        Domain specification or compiled domain object
    void_mesh : trimesh.Trimesh or str or Path
        Void mesh to embed, either as a mesh object or path to STL file
    embedding_policy : EmbeddingPolicy, optional
        Policy controlling embedding parameters
    ports : list of dict, optional
        Port specifications for port preservation. Each dict should have:
        - position: (x, y, z) tuple
        - direction: (dx, dy, dz) tuple
        - radius: float
    resolution_policy : ResolutionPolicy, optional
        Resolution policy for pitch derivation when use_resolution_policy=True.
    domain_mesh : trimesh.Trimesh, optional
        Pre-built domain mesh to use instead of generating from domain spec.
        Use this to preserve ridge-augmented or other modified domain meshes.
        
    Returns
    -------
    solid : trimesh.Trimesh
        Domain with void carved out (for 3D printing)
    void : trimesh.Trimesh
        The void mesh (for visualization)
    shell : trimesh.Trimesh
        Shell mesh around the void (optional, may be same as solid)
    report : OperationReport
        Report with requested/effective policy and metadata
    """
    import trimesh
    from ..ops.embedding import embed_tree_as_negative_space
    
    if embedding_policy is None:
        embedding_policy = EmbeddingPolicy()
    
    compiled_domain = _coerce_domain(domain)
    
    # Load void mesh if path provided
    if isinstance(void_mesh, (str, Path)):
        void_mesh_obj = trimesh.load(str(void_mesh))
        if not isinstance(void_mesh_obj, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh, got {type(void_mesh_obj)}")
        void_mesh = void_mesh_obj
    
    warnings = []
    metadata = {
        "voxel_pitch": embedding_policy.voxel_pitch,
        "shell_thickness": embedding_policy.shell_thickness,
    }
    
    # Track pitch adjustments
    pitch_adjustments = []
    effective_pitch = embedding_policy.voxel_pitch
    
    # Perform embedding with retry logic
    try:
        solid, void_out, shell, embed_metadata = _embed_with_retry(
            compiled_domain,
            void_mesh,
            embedding_policy,
            pitch_adjustments,
            resolution_policy,
            domain_mesh=domain_mesh,
        )
        
        if pitch_adjustments:
            effective_pitch = pitch_adjustments[-1]
            warnings.append(
                f"Voxel pitch adjusted from {embedding_policy.voxel_pitch:.6f} "
                f"to {effective_pitch:.6f} due to memory constraints"
            )
        
        # B2/B3 FIX: Include resolution resolver warnings and metadata
        if embed_metadata.get("resolution_warnings"):
            warnings.extend(embed_metadata["resolution_warnings"])
        if embed_metadata.get("resolution_result"):
            metadata["resolution_result"] = embed_metadata["resolution_result"]
            # Update effective pitch from resolution result
            effective_pitch = embed_metadata["resolution_result"].get("effective_pitch", effective_pitch)
        
        # VOXEL RECARVE: Apply port preservation if enabled and ports provided
        # Uses voxel-based carving (no boolean backend dependency)
        ports_preserved = 0
        recarve_report_dict = None
        if embedding_policy.preserve_ports_enabled and ports:
            solid, ports_preserved, port_warnings, recarve_report_dict = _preserve_ports(
                solid,
                ports,
                embedding_policy,
            )
            warnings.extend(port_warnings)
            metadata["ports_preserved"] = ports_preserved
            if recarve_report_dict:
                metadata["recarve_report"] = recarve_report_dict
        
        # Check for shrink warnings
        shrink_warnings = _check_shrink_warnings(void_mesh, void_out)
        warnings.extend(shrink_warnings)
        
        metadata["pitch_adjustments"] = pitch_adjustments
        metadata["effective_pitch"] = effective_pitch
        metadata["use_resolution_policy"] = embedding_policy.use_resolution_policy
        metadata["solid_volume"] = float(solid.volume) if solid.is_watertight else None
        metadata["void_volume"] = float(void_out.volume) if void_out.is_watertight else None
        
        success = True
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Return empty meshes on failure
        solid = trimesh.Trimesh()
        void_out = void_mesh
        shell = trimesh.Trimesh()
        success = False
        metadata["error"] = str(e)
    
    # Build effective policy
    effective_policy = embedding_policy.to_dict()
    effective_policy["voxel_pitch"] = effective_pitch
    
    report = OperationReport(
        operation="embed_void",
        success=success,
        requested_policy=embedding_policy.to_dict(),
        effective_policy=effective_policy,
        warnings=warnings,
        metadata=metadata,
    )
    
    return solid, void_out, shell, report


def _embed_with_retry(
    domain: "Domain",
    void_mesh: "trimesh.Trimesh",
    policy: EmbeddingPolicy,
    pitch_adjustments: list,
    resolution_policy: Optional[ResolutionPolicy] = None,
    domain_mesh: Optional["trimesh.Trimesh"] = None,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", "trimesh.Trimesh", Dict[str, Any]]:
    """
    B2/B3 FIX: Perform embedding using canonical mesh-based embedding directly.
    
    Now calls embed_void_mesh_as_negative_space directly with in-memory meshes,
    eliminating the need for temp STL files.
    
    Supports resolution policy for unified pitch selection:
    - If use_resolution_policy=True, uses the resolution resolver for pitch derivation
    - Respects max_voxels budget with deterministic pitch relaxation
    
    If domain_mesh is provided, uses it directly instead of generating from domain spec.
    This preserves ridge-augmented or other modified domain meshes.
    """
    from ..ops.embedding.enhanced_embedding import embed_void_mesh_as_negative_space as enhanced_embed
    import trimesh
    
    current_pitch = policy.voxel_pitch
    
    try:
        # B2/B3 FIX: Call canonical mesh-based embedding function with resolution policy
        solid, void_out, shell, metadata = enhanced_embed(
            void_mesh=void_mesh,
            domain=domain,
            voxel_pitch=current_pitch,
            output_void=True,
            output_shell=True,
            shell_thickness=policy.shell_thickness,
            auto_adjust_pitch=policy.auto_adjust_pitch,
            max_pitch_steps=policy.max_pitch_steps,
            max_voxels=policy.max_voxels,
            resolution_policy=resolution_policy,
            use_resolution_policy=policy.use_resolution_policy,
            domain_mesh=domain_mesh,
        )
        
        # Track pitch adjustments from metadata
        if metadata.get("pitch_adjustments", 0) > 0:
            effective_pitch = metadata.get("voxel_pitch_used", current_pitch)
            pitch_adjustments.append(effective_pitch)
        
        # Also track resolution policy relaxation
        if metadata.get("budget_first_relaxed", False):
            effective_pitch = metadata.get("voxel_pitch", current_pitch)
            if effective_pitch not in pitch_adjustments:
                pitch_adjustments.append(effective_pitch)
        
        # Handle None values
        if solid is None:
            solid = trimesh.Trimesh()
        if void_out is None:
            void_out = void_mesh
        if shell is None:
            shell = solid
        
        return solid, void_out, shell, metadata
        
    except Exception as e:
        logger.error(f"Mesh-based embedding failed: {e}")
        raise


def _preserve_ports(
    mesh: "trimesh.Trimesh",
    ports: List[Dict[str, Any]],
    policy: EmbeddingPolicy,
) -> Tuple["trimesh.Trimesh", int, List[str], Optional[Dict[str, Any]]]:
    """
    Preserve port geometry in the embedded mesh via voxel-based recarving.
    
    VOXEL RECARVE: Now uses voxel grid operations instead of boolean backends,
    eliminating dependency on Blender/Cork/etc. Works consistently across all
    environments.
    
    Uses the EmbeddingPolicy port preservation fields:
    - carve_radius_factor: Factor to multiply port radius for carving
    - carve_depth: Depth of carving cylinder
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh to preserve ports in
    ports : list of dict
        Port specifications with position, direction, radius
    policy : EmbeddingPolicy
        Policy with port preservation settings
        
    Returns
    -------
    result_mesh : trimesh.Trimesh
        Mesh with ports preserved
    ports_preserved : int
        Number of ports successfully preserved
    warnings : list of str
        Any warnings generated during preservation
    recarve_report_dict : dict or None
        Detailed recarve report as dict for metadata
    """
    from ..ops.embedding.enhanced_embedding import voxel_recarve_ports
    
    if not ports:
        return mesh, 0, [], None
    
    # Use voxel-based recarving (no boolean backend dependency)
    result_mesh, recarve_report = voxel_recarve_ports(
        mesh=mesh,
        ports=ports,
        voxel_pitch=policy.voxel_pitch,
        carve_radius_factor=policy.carve_radius_factor,
        carve_depth=policy.carve_depth,
        carve_shape="cylinder",
    )
    
    # Collect warnings from report
    warnings = list(recarve_report.warnings)
    
    # Count successfully carved ports (those with voxels_carved > 0)
    ports_preserved = sum(
        1 for pr in recarve_report.port_results if pr.voxels_carved > 0
    )
    
    return result_mesh, ports_preserved, warnings, recarve_report.to_dict()


def _check_shrink_warnings(
    original_void: "trimesh.Trimesh",
    embedded_void: "trimesh.Trimesh",
) -> list:
    """
    Check for shrinkage during embedding and generate warnings.
    """
    warnings = []
    
    try:
        orig_vol = float(original_void.volume) if original_void.is_watertight else None
        embed_vol = float(embedded_void.volume) if embedded_void.is_watertight else None
        
        if orig_vol and embed_vol:
            shrink_ratio = (orig_vol - embed_vol) / orig_vol
            if shrink_ratio > 0.1:  # More than 10% shrinkage
                warnings.append(
                    f"Void volume shrunk by {shrink_ratio*100:.1f}% during embedding "
                    f"(original: {orig_vol:.9e}, embedded: {embed_vol:.9e})"
                )
        
        # Check for diameter changes via bounding box
        orig_extents = original_void.extents
        embed_extents = embedded_void.extents
        
        for i, (orig, embed) in enumerate(zip(orig_extents, embed_extents)):
            if orig > 0:
                change = (orig - embed) / orig
                if change > 0.1:  # More than 10% change
                    axis = ["X", "Y", "Z"][i]
                    warnings.append(
                        f"{axis}-axis extent shrunk by {change*100:.1f}%"
                    )
                    
    except Exception as e:
        logger.warning(f"Could not compute shrink warnings: {e}")
    
    return warnings


def embed_void_mesh_as_negative_space(
    domain: Union[DomainSpec, "Domain"],
    void_mesh: Union["trimesh.Trimesh", str, Path],
    voxel_pitch: float = 3e-4,
    shell_thickness: float = 2e-3,
    fallback: str = "auto",
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Convenience wrapper for embed_void with simpler interface.
    
    This is the primary embedding function, accepting in-memory meshes.
    
    Parameters
    ----------
    domain : DomainSpec or Domain
        Domain specification
    void_mesh : trimesh.Trimesh or str or Path
        Void mesh to embed
    voxel_pitch : float
        Voxel pitch in meters (default: 0.3mm)
    shell_thickness : float
        Shell thickness in meters (default: 2mm)
    fallback : str
        Fallback strategy: "auto", "voxel_subtraction", "none"
        
    Returns
    -------
    solid : trimesh.Trimesh
        Domain with void carved out
    report : OperationReport
        Report with metadata
    """
    policy = EmbeddingPolicy(
        voxel_pitch=voxel_pitch,
        shell_thickness=shell_thickness,
        fallback=fallback,
    )
    
    solid, void_out, shell, report = embed_void(domain, void_mesh, policy)
    
    return solid, report


__all__ = [
    "embed_void",
    "embed_void_mesh_as_negative_space",
]
