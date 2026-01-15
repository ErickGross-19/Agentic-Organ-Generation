"""
Unified embedding API for vascular network voids.

This module provides the main entry point for embedding void meshes
into domain volumes as negative space.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, Tuple, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path
import logging

from ..policies import EmbeddingPolicy, OperationReport
from ..specs.design_spec import DomainSpec
from ..core.domain import BoxDomain, EllipsoidDomain, CylinderDomain

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def embed_void(
    domain: Union[DomainSpec, "Domain"],
    void_mesh: Union["trimesh.Trimesh", str, Path],
    embedding_policy: Optional[EmbeddingPolicy] = None,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", "trimesh.Trimesh", OperationReport]:
    """
    Embed a void mesh into a domain as negative space.
    
    This is the unified entry point for embedding operations, supporting
    both in-memory meshes and file paths.
    
    Parameters
    ----------
    domain : DomainSpec or Domain
        Domain specification or compiled domain object
    void_mesh : trimesh.Trimesh or str or Path
        Void mesh to embed, either as a mesh object or path to STL file
    embedding_policy : EmbeddingPolicy, optional
        Policy controlling embedding parameters
        
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
    from ..specs.compile import compile_domain
    
    if embedding_policy is None:
        embedding_policy = EmbeddingPolicy()
    
    # Compile domain if needed
    if hasattr(domain, 'type'):
        compiled_domain = compile_domain(domain)
    else:
        compiled_domain = domain
    
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
        solid, void_out, shell = _embed_with_retry(
            compiled_domain,
            void_mesh,
            embedding_policy,
            pitch_adjustments,
        )
        
        if pitch_adjustments:
            effective_pitch = pitch_adjustments[-1]
            warnings.append(
                f"Voxel pitch adjusted from {embedding_policy.voxel_pitch:.6f} "
                f"to {effective_pitch:.6f} due to memory constraints"
            )
        
        # Check for shrink warnings
        shrink_warnings = _check_shrink_warnings(void_mesh, void_out)
        warnings.extend(shrink_warnings)
        
        metadata["pitch_adjustments"] = pitch_adjustments
        metadata["effective_pitch"] = effective_pitch
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
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", "trimesh.Trimesh"]:
    """
    Perform embedding with automatic pitch adjustment on memory errors.
    
    Note: The existing embed_tree_as_negative_space function only accepts
    file paths, so we save the in-memory mesh to a temp file. This is a
    Phase 1 workaround; Phase 2 will refactor the underlying function to
    accept in-memory meshes directly.
    """
    from ..ops.embedding import embed_tree_as_negative_space
    import trimesh
    import tempfile
    import os
    
    current_pitch = policy.voxel_pitch
    max_attempts = policy.max_pitch_steps if policy.auto_adjust_pitch else 1
    
    # Save mesh to temp file (existing function only accepts paths)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "void_mesh.stl")
    
    try:
        void_mesh.export(temp_path)
        
        for attempt in range(max_attempts):
            try:
                # Call the existing embedding function with file path
                result = embed_tree_as_negative_space(
                    tree_stl_path=temp_path,
                    domain=domain,
                    voxel_pitch=current_pitch,
                    shell_thickness=policy.shell_thickness,
                    output_void=True,
                    output_shell=True,
                    auto_adjust_pitch=policy.auto_adjust_pitch,
                )
                
                # Unpack result dictionary
                if isinstance(result, dict):
                    solid = result.get('domain_with_void', trimesh.Trimesh())
                    void_out = result.get('void', void_mesh)
                    shell = result.get('shell', solid)
                    
                    # Handle None values
                    if solid is None:
                        solid = trimesh.Trimesh()
                    if void_out is None:
                        void_out = void_mesh
                    if shell is None:
                        shell = solid
                else:
                    solid = result
                    void_out = void_mesh
                    shell = solid
                
                return solid, void_out, shell
                
            except MemoryError:
                if attempt < max_attempts - 1:
                    current_pitch *= 1.5
                    pitch_adjustments.append(current_pitch)
                    logger.warning(
                        f"Memory error during embedding, increasing pitch to {current_pitch:.6f}"
                    )
                else:
                    raise
        
        raise RuntimeError("Embedding failed after all retry attempts")
        
    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


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
