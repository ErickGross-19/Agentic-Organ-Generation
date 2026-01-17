"""
RunnerContext and ArtifactStore for DesignSpec execution.

This module provides caching and artifact management for the
DesignSpecRunner pipeline.

CACHING
-------
RunnerContext caches expensive computations using stable content hashes:
- Compiled domains
- Domain meshes
- Component outputs (networks, void meshes)
- Unioned void
- Embedded outputs

ARTIFACTS
---------
ArtifactStore manages named intermediate artifacts:
- Registers artifacts by stage + name
- Saves to disk based on outputs policy
- Builds manifest entries with metadata
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from pathlib import Path
import json
import hashlib
import logging
import time

if TYPE_CHECKING:
    import trimesh
    from generation.core.domain import DomainSpec
    from generation.core.network import VascularNetwork

logger = logging.getLogger(__name__)


def _compute_hash(obj: Any) -> str:
    """
    Compute a stable hash for an object.
    
    Parameters
    ----------
    obj : Any
        Object to hash (must be JSON-serializable or have to_dict method)
        
    Returns
    -------
    str
        Hex digest of the hash (first 16 characters)
    """
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    elif isinstance(obj, dict):
        data = obj
    else:
        data = str(obj)
    
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@dataclass
class CacheEntry:
    """A cached computation result."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class RunnerContext:
    """
    Context for caching expensive computations during pipeline execution.
    
    Provides a cache contract for storing and retrieving intermediate
    results keyed by stable content hashes.
    
    Cached items include:
    - compiled_domain:<hash> - Compiled domain objects
    - domain_mesh:<hash> - Domain meshes
    - component_network:<id> - Generated networks
    - component_void:<id> - Component void meshes
    - union_void - Unioned void mesh
    - embedded_solid - Embedded domain with void
    - embedded_shell - Shell mesh
    
    Attributes
    ----------
    spec_hash : str
        Hash of the spec being executed
    seed : int or None
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        spec_hash: str,
        seed: Optional[int] = None,
    ):
        self.spec_hash = spec_hash
        self.seed = seed
        self._cache: Dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a cached value by key.
        
        Parameters
        ----------
        key : str
            Cache key
        default : Any
            Default value if not found
            
        Returns
        -------
        Any
            Cached value or default
        """
        entry = self._cache.get(key)
        if entry is not None:
            self._hits += 1
            entry.hit_count += 1
            return entry.value
        self._misses += 1
        return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a cached value by key.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        self._cache[key] = CacheEntry(key=key, value=value)
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache
    
    def get_or_compute(self, key: str, compute_fn: callable) -> Any:
        """
        Get a cached value or compute and cache it.
        
        Parameters
        ----------
        key : str
            Cache key
        compute_fn : callable
            Function to compute the value if not cached
            
        Returns
        -------
        Any
            Cached or computed value
        """
        if key in self._cache:
            self._hits += 1
            self._cache[key].hit_count += 1
            return self._cache[key].value
        
        self._misses += 1
        value = compute_fn()
        self._cache[key] = CacheEntry(key=key, value=value)
        return value
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached entry.
        
        Parameters
        ----------
        key : str
            Cache key to invalidate
            
        Returns
        -------
        bool
            True if entry was removed, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Cache statistics including hits, misses, and size
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }
    
    def make_domain_key(self, domain_dict: Dict[str, Any]) -> str:
        """Make a cache key for a domain."""
        return f"compiled_domain:{_compute_hash(domain_dict)}"
    
    def make_policy_key(self, policy_name: str, policy_dict: Dict[str, Any]) -> str:
        """Make a cache key for a policy."""
        return f"compiled_policy:{policy_name}:{_compute_hash(policy_dict)}"
    
    def make_component_key(self, component_id: str, stage: str) -> str:
        """Make a cache key for a component stage."""
        return f"component_{stage}:{component_id}"


@dataclass
class ArtifactEntry:
    """A registered artifact."""
    name: str
    stage: str
    path: Optional[Path] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    saved: bool = False


class ArtifactStore:
    """
    Store for named intermediate artifacts.
    
    Manages artifact registration, saving, and manifest generation
    for the DesignSpecRunner pipeline.
    
    Attributes
    ----------
    output_dir : Path
        Base output directory
    artifacts_dir : Path
        Directory for intermediate artifacts
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        artifacts_subdir: str = "artifacts",
    ):
        self.output_dir = Path(output_dir)
        self.artifacts_dir = self.output_dir / artifacts_subdir
        self._artifacts: Dict[str, ArtifactEntry] = {}
        self._requested: Dict[str, str] = {}
    
    def request_artifact(self, name: str, path: str) -> None:
        """
        Request that an artifact be saved to a specific path.
        
        Parameters
        ----------
        name : str
            Artifact name (e.g., "net_1_network")
        path : str
            Relative path for saving (e.g., "artifacts/net_1_network.json")
        """
        self._requested[name] = path
    
    def register(
        self,
        name: str,
        stage: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactEntry:
        """
        Register an artifact.
        
        Parameters
        ----------
        name : str
            Artifact name
        stage : str
            Stage that produced the artifact
        value : Any
            The artifact value (mesh, network, etc.)
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        ArtifactEntry
            The registered artifact entry
        """
        content_hash = None
        if hasattr(value, "to_dict"):
            content_hash = _compute_hash(value.to_dict())
        elif isinstance(value, dict):
            content_hash = _compute_hash(value)
        
        entry = ArtifactEntry(
            name=name,
            stage=stage,
            content_hash=content_hash,
            metadata=metadata or {},
        )
        
        self._artifacts[name] = entry
        
        if name in self._requested:
            entry.path = self.output_dir / self._requested[name]
        
        return entry
    
    def get(self, name: str) -> Optional[ArtifactEntry]:
        """Get an artifact entry by name."""
        return self._artifacts.get(name)
    
    def save_mesh(
        self,
        name: str,
        mesh: "trimesh.Trimesh",
        format: str = "stl",
    ) -> Optional[Path]:
        """
        Save a mesh artifact to disk.
        
        Parameters
        ----------
        name : str
            Artifact name
        mesh : trimesh.Trimesh
            Mesh to save
        format : str
            Output format (stl, ply, obj)
            
        Returns
        -------
        Path or None
            Path where mesh was saved, or None if not requested
        """
        entry = self._artifacts.get(name)
        if entry is None:
            return None
        
        if entry.path is None:
            return None
        
        entry.path.parent.mkdir(parents=True, exist_ok=True)
        
        mesh.export(str(entry.path))
        entry.saved = True
        
        entry.metadata["face_count"] = len(mesh.faces)
        entry.metadata["vertex_count"] = len(mesh.vertices)
        entry.metadata["is_watertight"] = mesh.is_watertight
        if mesh.is_watertight:
            entry.metadata["volume"] = float(mesh.volume)
        
        # Handle empty/degenerate meshes gracefully
        bbox = mesh.bounds
        if bbox is not None and len(mesh.vertices) > 0:
            entry.metadata["bbox"] = {
                "min": bbox[0].tolist(),
                "max": bbox[1].tolist(),
            }
        else:
            entry.metadata["bbox"] = None
        
        return entry.path
    
    def save_json(
        self,
        name: str,
        data: Union[Dict[str, Any], Any],
    ) -> Optional[Path]:
        """
        Save a JSON artifact to disk.
        
        Parameters
        ----------
        name : str
            Artifact name
        data : dict or object with to_dict method
            Data to save
            
        Returns
        -------
        Path or None
            Path where JSON was saved, or None if not requested
        """
        entry = self._artifacts.get(name)
        if entry is None:
            return None
        
        if entry.path is None:
            return None
        
        entry.path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        
        with open(entry.path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        entry.saved = True
        
        return entry.path
    
    def build_manifest(self) -> Dict[str, Any]:
        """
        Build a manifest of all registered artifacts.
        
        Returns
        -------
        dict
            Manifest with artifact entries
        """
        manifest = {}
        
        for name, entry in self._artifacts.items():
            manifest[name] = {
                "stage": entry.stage,
                "path": str(entry.path) if entry.path else None,
                "content_hash": entry.content_hash,
                "saved": entry.saved,
                "metadata": entry.metadata,
            }
        
        return manifest
    
    def save_manifest(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the artifact manifest to disk.
        
        Parameters
        ----------
        path : str or Path, optional
            Path for manifest file (default: output_dir/artifact_manifest.json)
            
        Returns
        -------
        Path
            Path where manifest was saved
        """
        if path is None:
            path = self.output_dir / "artifact_manifest.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest = self.build_manifest()
        
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return path


__all__ = [
    "RunnerContext",
    "ArtifactStore",
    "ArtifactEntry",
    "CacheEntry",
]
